"""Long greedy decode on realistic embeddings: original graph on the CPU vs rewritten graph on DirectML.

Prompts are real token embeddings from embed_tokens.onnx (speech tokens standing in for the voice conditioning,
then text tokens), right-padded as the runtime batches them. Each graph decodes greedily, feeding back its own
tokens; the report shows where the token sequences first diverge and how far the logits drift.

Usage: python validate_greedy.py <onnx directory> [steps]
"""
import os
import sys

import numpy as np
import onnxruntime as ort

from validate import Batch, capacity_for, empty_past, outputs_of, session

START_SPEECH = 6561
STOP_SPEECH = 6562
TEXT_VOCABULARY = 2000
CAPACITY_STEP = 128


def embed(embed_session, ids, positions, exaggeration=0.5):
    feeds = {"input_ids": np.array([ids], np.int64), "position_ids": np.array([positions], np.int64),
             "exaggeration": np.array([exaggeration], np.float32)}
    return embed_session.run(None, feeds)[0][0]


def prompts(embed_session, lengths, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for length in lengths:
        speech = list(rng.integers(0, 6000, length // 2))
        text = list(rng.integers(10, TEXT_VOCABULARY, length - length // 2 - 1)) + [START_SPEECH]
        ids = speech + text
        positions = [0 if token >= START_SPEECH else i for i, token in enumerate(ids)]
        rows.append(embed(embed_session, ids, positions))
    return rows


def padded(rows, width):
    embeds = np.zeros((len(rows), width, rows[0].shape[1]), np.float32)
    for row, values in enumerate(rows):
        embeds[row, :len(values)] = values
    return embeds


def decode(sess, embed_session, prompt_rows, steps, static_cache):
    batch = Batch([len(r) for r in prompt_rows], 0)
    feeds = {"inputs_embeds": padded(prompt_rows, batch.width), **empty_past(sess, batch.rows)}
    if static_cache:
        feeds["attention_mask"] = batch.mask(0, capacity_for(batch.width + CAPACITY_STEP, CAPACITY_STEP))
        feeds["cache_write_index"] = np.array([0], np.int64)
    else:
        feeds["attention_mask"] = batch.mask(0)
    logits, past = outputs_of(sess, feeds)
    step_logits = np.stack([logits[row, length - 1] for row, length in enumerate(batch.valid_lengths)])
    tokens, history = [], []
    for step in range(1, steps + 1):
        next_tokens = step_logits[:, :STOP_SPEECH + 1].argmax(-1)
        tokens.append(next_tokens)
        history.append(step_logits)
        embeds = np.stack([embed(embed_session, [int(t)], [step]) for t in next_tokens])
        feeds = {"inputs_embeds": embeds, **past}
        if static_cache:
            used = batch.width + step
            feeds["attention_mask"] = batch.mask(step, capacity_for(max(used, batch.width + CAPACITY_STEP), CAPACITY_STEP))
            feeds["cache_write_index"] = np.array([used - 1], np.int64)
        else:
            feeds["attention_mask"] = batch.mask(step)
        logits, past = outputs_of(sess, feeds)
        step_logits = logits[:, 0]
    return np.array(tokens), np.array(history)


def main():
    directory = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    embed_session = session(os.path.join(directory, "embed_tokens.onnx"), "CPUExecutionProvider")
    prompt_rows = prompts(embed_session, [240, 180], 3)
    reference_tokens, reference_logits = decode(
        session(os.path.join(directory, "language_model.onnx"), "CPUExecutionProvider"), embed_session, prompt_rows, steps, False)
    for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
        tokens, logits = decode(
            session(os.path.join(directory, "language_model_dml.onnx"), provider), embed_session, prompt_rows, steps, True)
        same = (tokens == reference_tokens)
        first_divergence = [int(np.argmin(same[:, row])) if not same[:, row].all() else steps for row in range(same.shape[1])]
        drift = np.abs(logits - reference_logits).max(axis=(1, 2))
        print(f"{provider}: identical tokens until step {first_divergence} of {steps}; "
              f"max logit drift at steps 1/10/50/last: {drift[0]:.1e} {drift[min(9, steps - 1)]:.1e} {drift[min(49, steps - 1)]:.1e} {drift[-1]:.1e}")


if __name__ == "__main__":
    main()
