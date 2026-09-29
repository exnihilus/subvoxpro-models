"""Prompt on the CPU with the original graph, decode on DirectML with the rewritten graph.

The original graph's prompt cache already has the rewritten layout (prompt in the first slots, padding masked), so
it can seed the fixed-capacity cache. DirectML keeps its fast fused graph only for the shapes it compiled first,
so the decode session is warmed up on its canonical shape (fixed row count and capacity) before any real work.

Checks: greedy tokens match the all-CPU original over a long decode, then decode speed over several prompts.

Usage: python validate_handoff.py <onnx directory> [steps]
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

from validate import Batch, empty_past, outputs_of, session
from validate_greedy import STOP_SPEECH, embed, padded, prompts

CANONICAL_ROWS = 12
CAPACITY = 512
HIDDEN = 1024


class GpuDecoder:
    def __init__(self, path):
        self.session = session(path, "DmlExecutionProvider")
        self.past_names = [i.name for i in self.session.get_inputs() if i.name.startswith("past_")]
        self.present_names = [n.replace("past_key_values.", "present.") for n in self.past_names]
        self.caches = [[ort.OrtValue.ortvalue_from_shape_and_type((CANONICAL_ROWS, 16, CAPACITY, 64), np.float32, "dml")
                        for _ in self.past_names] for _ in range(2)]
        self.logits = ort.OrtValue.ortvalue_from_shape_and_type((CANONICAL_ROWS, 1, 8194), np.float32, "cpu")
        self.latest = None
        self.warm_up()

    def warm_up(self):
        zero_past = [np.zeros((CANONICAL_ROWS, 16, CAPACITY, 64), np.float32)] * len(self.past_names)
        mask = np.zeros((CANONICAL_ROWS, CAPACITY), np.int64)
        mask[:, 0] = 1
        self.run(np.zeros((CANONICAL_ROWS, 1, HIDDEN), np.float32), mask, 0, zero_past, target=0)

    def run(self, embeds, mask, write_index, past, target):
        binding = self.session.io_binding()
        binding.bind_cpu_input("inputs_embeds", embeds)
        binding.bind_cpu_input("attention_mask", mask)
        binding.bind_cpu_input("cache_write_index", np.array([write_index], np.int64))
        for name, value in zip(self.past_names, past):
            if isinstance(value, np.ndarray):
                binding.bind_cpu_input(name, value)
            else:
                binding.bind_ortvalue_input(name, value)
        for name, value in zip(self.present_names, self.caches[target]):
            binding.bind_ortvalue_output(name, value)
        binding.bind_ortvalue_output("logits", self.logits)
        self.session.run_with_iobinding(binding)
        self.latest = target
        return self.logits.numpy()[:, 0]

    def step(self, embeds, mask, write_index, prompt_past=None):
        past = prompt_past if prompt_past is not None else self.caches[self.latest]
        target = 0 if self.latest != 0 else 1
        return self.run(embeds, mask, write_index, past, target)


def pad_rows(array, rows):
    padded_array = np.zeros((rows,) + array.shape[1:], array.dtype)
    padded_array[:array.shape[0]] = array
    return padded_array


def handoff_decode(cpu, gpu, embed_session, prompt_rows, steps):
    lengths = [len(r) for r in prompt_rows] + [1] * (CANONICAL_ROWS - len(prompt_rows))
    batch = Batch(lengths, 0)
    embeds = pad_rows(padded(prompt_rows, batch.width), CANONICAL_ROWS)
    logits, past = outputs_of(cpu, {"inputs_embeds": embeds, "attention_mask": batch.mask(0), **empty_past(cpu, CANONICAL_ROWS)})
    step_logits = np.stack([logits[row, length - 1] for row, length in enumerate(lengths)])
    prompt_past = [past[name] for name in gpu.past_names]
    tokens = []
    decode_time = 0.0
    for step in range(1, steps + 1):
        next_tokens = step_logits[:len(prompt_rows), :STOP_SPEECH + 1].argmax(-1)
        tokens.append(next_tokens)
        step_embeds = np.zeros((CANONICAL_ROWS, 1, HIDDEN), np.float32)
        for row, token in enumerate(next_tokens):
            step_embeds[row] = embed(embed_session, [int(token)], [step])
        start = time.perf_counter()
        step_logits = gpu.step(step_embeds, batch.mask(step, CAPACITY), batch.width + step - 1, prompt_past if step == 1 else None)
        decode_time += time.perf_counter() - start
    return np.array(tokens), 1000 * decode_time / steps


def cpu_reference(cpu, embed_session, prompt_rows, steps):
    batch = Batch([len(r) for r in prompt_rows], 0)
    logits, past = outputs_of(cpu, {"inputs_embeds": padded(prompt_rows, batch.width), "attention_mask": batch.mask(0), **empty_past(cpu, batch.rows)})
    step_logits = np.stack([logits[row, length - 1] for row, length in enumerate(batch.valid_lengths)])
    tokens = []
    for step in range(1, steps + 1):
        next_tokens = step_logits[:, :STOP_SPEECH + 1].argmax(-1)
        tokens.append(next_tokens)
        embeds = np.stack([embed(embed_session, [int(t)], [step]) for t in next_tokens])
        logits, past = outputs_of(cpu, {"inputs_embeds": embeds, "attention_mask": batch.mask(step), **past})
        step_logits = logits[:, 0]
    return np.array(tokens)


def main():
    directory = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    embed_session = session(os.path.join(directory, "embed_tokens.onnx"), "CPUExecutionProvider")
    cpu = session(os.path.join(directory, "language_model.onnx"), "CPUExecutionProvider")
    gpu = GpuDecoder(os.path.join(directory, "language_model_dml.onnx"))
    for seed, lengths in ((3, [240, 180]), (5, [230, 230, 200, 200, 190, 190]), (9, [260, 210, 210, 150])):
        prompt_rows = prompts(embed_session, lengths, seed)
        reference = cpu_reference(cpu, embed_session, prompt_rows, steps)
        tokens, ms = handoff_decode(cpu, gpu, embed_session, prompt_rows, steps)
        same = tokens == reference
        divergence = [int(np.argmin(same[:, r])) if not same[:, r].all() else steps for r in range(same.shape[1])]
        print(f"{len(lengths)} rows: identical tokens until {divergence} of {steps}; GPU decode {ms:.1f} ms/step")


if __name__ == "__main__":
    main()
