"""Checks that the rewritten language model matches the original, then times it.

Equivalence: a right-padded batch (two phrases of different lengths, two CFG rows each) runs a prefill and
several decode steps on both graphs, each with its own cache protocol. The rewritten cache grows in small steps
here so the growth path is exercised too. The logits the runtime samples from (last valid prefill position, then
every decode step) must match.

Usage: python validate.py <language_model.onnx> <language_model_dml.onnx> [--speed]
"""
import sys
import time

import numpy as np
import onnxruntime as ort

HIDDEN = 1024
HEADS = 16
HEAD_SIZE = 64
DECODE_STEPS = 12
TEST_CAPACITY_STEP = 16
SPEED_CAPACITY_STEP = 128


def session(path, provider):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=[provider])


def past_names(sess):
    return [i.name for i in sess.get_inputs() if i.name.startswith("past_key_values.")]


def empty_past(sess, rows):
    return {name: np.zeros((rows, HEADS, 0, HEAD_SIZE), np.float32) for name in past_names(sess)}


class Batch:
    """Right-padded prompts followed by appended tokens, as the runtime feeds them."""

    def __init__(self, valid_lengths, seed):
        self.valid_lengths = valid_lengths
        self.rows = len(valid_lengths)
        self.width = max(valid_lengths)
        self.rng = np.random.default_rng(seed)
        self.prompt_valid = np.zeros((self.rows, self.width), np.int64)
        for row, length in enumerate(valid_lengths):
            self.prompt_valid[row, :length] = 1

    def prompt_embeds(self):
        embeds = self.rng.standard_normal((self.rows, self.width, HIDDEN)).astype(np.float32) * 0.1
        return embeds * self.prompt_valid[:, :, None].astype(np.float32)

    def step_embeds(self):
        return self.rng.standard_normal((self.rows, 1, HIDDEN)).astype(np.float32) * 0.1

    def mask(self, generated, capacity=None):
        used = self.width + generated
        mask = np.zeros((self.rows, capacity or used), np.int64)
        mask[:, :self.width] = self.prompt_valid
        mask[:, self.width:used] = 1
        return mask


def outputs_of(sess, feeds):
    names = [o.name for o in sess.get_outputs()]
    values = dict(zip(names, sess.run(names, feeds)))
    past = {n.replace("present.", "past_key_values."): v for n, v in values.items() if n.startswith("present.")}
    return values["logits"], past


def original_logits(sess, batch):
    logits, past = outputs_of(sess, {"inputs_embeds": batch.prompt_embeds(), "attention_mask": batch.mask(0), **empty_past(sess, batch.rows)})
    collected = [np.stack([logits[row, length - 1] for row, length in enumerate(batch.valid_lengths)])]
    for step in range(1, DECODE_STEPS + 1):
        logits, past = outputs_of(sess, {"inputs_embeds": batch.step_embeds(), "attention_mask": batch.mask(step), **past})
        collected.append(logits[:, 0])
    return np.stack(collected)


def capacity_for(used, capacity_step):
    return -(-used // capacity_step) * capacity_step


def rewritten_logits(sess, batch):
    capacity = capacity_for(batch.width, TEST_CAPACITY_STEP)
    feeds = {"inputs_embeds": batch.prompt_embeds(), "attention_mask": batch.mask(0, capacity),
             "cache_write_index": np.array([0], np.int64), **empty_past(sess, batch.rows)}
    logits, past = outputs_of(sess, feeds)
    collected = [np.stack([logits[row, length - 1] for row, length in enumerate(batch.valid_lengths)])]
    for step in range(1, DECODE_STEPS + 1):
        slot = batch.width + step - 1
        capacity = capacity_for(slot + 1, TEST_CAPACITY_STEP)
        feeds = {"inputs_embeds": batch.step_embeds(), "attention_mask": batch.mask(step, capacity),
                 "cache_write_index": np.array([slot], np.int64), **past}
        logits, past = outputs_of(sess, feeds)
        collected.append(logits[:, 0])
    return np.stack(collected)


def compare(label, reference, candidate):
    difference = np.abs(reference - candidate)
    same_top = np.mean(np.argmax(reference, -1) == np.argmax(candidate, -1))
    print(f"{label}: max |diff| {difference.max():.2e}, mean |diff| {difference.mean():.2e}, "
          f"logit range {np.abs(reference).max():.1f}, same top token {same_top:.0%}")
    return difference.max()


def check_equivalence(original_path, rewritten_path):
    valid_lengths = [40, 40, 23, 23]
    reference = original_logits(session(original_path, "CPUExecutionProvider"), Batch(valid_lengths, 7))
    worst = 0.0
    for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
        candidate = rewritten_logits(session(rewritten_path, provider), Batch(valid_lengths, 7))
        worst = max(worst, compare(f"rewritten on {provider}", reference, candidate))
    return worst


def time_rewritten_on_gpu(path, rows, context, steps):
    sess = session(path, "DmlExecutionProvider")
    rng = np.random.default_rng(0)
    names = [o.name for o in sess.get_outputs()]
    past = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in empty_past(sess, rows).items()}
    embeds = rng.standard_normal((rows, context, HIDDEN)).astype(np.float32) * 0.1
    keep_alive = None
    start = None
    for step in range(steps + 1):
        if step == 1:
            start = time.perf_counter()
        slot = 0 if step == 0 else context + step - 1
        used = context + step
        capacity = capacity_for(used, SPEED_CAPACITY_STEP)
        mask = np.zeros((rows, capacity), np.int64)
        mask[:, :used] = 1
        binding = sess.io_binding()
        binding.bind_cpu_input("inputs_embeds", embeds)
        binding.bind_cpu_input("attention_mask", mask)
        binding.bind_cpu_input("cache_write_index", np.array([slot], np.int64))
        for name, value in past.items():
            binding.bind_ortvalue_input(name, value)
        for name in names:
            binding.bind_output(name, "cpu" if name == "logits" else "dml")
        sess.run_with_iobinding(binding)
        keep_alive = binding
        outputs = dict(zip(names, keep_alive.get_outputs()))
        past = {n.replace("present.", "past_key_values."): v for n, v in outputs.items() if n.startswith("present.")}
        embeds = rng.standard_normal((rows, 1, HIDDEN)).astype(np.float32) * 0.1
    elapsed = time.perf_counter() - start
    print(f"rewritten on DirectML rows={rows:2d}: {steps / elapsed:6.1f} steps/s ({1000 * elapsed / steps:.1f} ms/step)")


def time_original_on_cpu(path, rows, context, steps):
    sess = session(path, "CPUExecutionProvider")
    batch = Batch([context] * rows, 0)
    logits, past = outputs_of(sess, {"inputs_embeds": batch.prompt_embeds(), "attention_mask": batch.mask(0), **empty_past(sess, rows)})
    start = time.perf_counter()
    for step in range(1, steps + 1):
        logits, past = outputs_of(sess, {"inputs_embeds": batch.step_embeds(), "attention_mask": batch.mask(step), **past})
    elapsed = time.perf_counter() - start
    print(f"original on CPU       rows={rows:2d}: {steps / elapsed:6.1f} steps/s ({1000 * elapsed / steps:.1f} ms/step)")


def main():
    original_path, rewritten_path = sys.argv[1], sys.argv[2]
    print(f"worst difference {check_equivalence(original_path, rewritten_path):.2e}")
    if "--speed" in sys.argv:
        for rows in (2, 12):
            time_original_on_cpu(original_path, rows, 250, 40)
            time_rewritten_on_gpu(rewritten_path, rows, 250, 200)


if __name__ == "__main__":
    main()
