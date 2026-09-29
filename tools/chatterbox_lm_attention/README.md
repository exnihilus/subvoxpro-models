# Chatterbox language model for DirectML

`chatterbox-multilingual/language_model_dml.onnx` is the upstream `language_model.onnx` graph with its 30
`com.microsoft.GroupQueryAttention` nodes replaced by standard ONNX operators, which DirectML runs. It references the
unchanged upstream `language_model.onnx_data` (fp32 weights), so no weight is duplicated or altered.

The attention keeps a fixed-capacity KV cache and takes an extra `cache_write_index` input, because DirectML
re-prepares a graph whenever a shape changes. The protocol is described in `attention.py`.

## Rebuild

```
python rewrite.py <dir>/language_model.onnx <dir>/language_model_dml.onnx
```

## Checks (need `onnxruntime-directml`)

- `validate.py <original> <rewritten> [--speed]`: right-padded batch, prompt + decode steps with cache growth; the
  logits match the original on the CPU and on DirectML (max difference ≈ 6e-5 for logits ≈ 14).
- `validate_greedy.py <dir> [steps]`: long greedy decode on real token embeddings; identical tokens to the original.
- `validate_handoff.py <dir> [steps]`: the runtime's split (prompt on the CPU with the original graph, decoding on
  DirectML with a fixed shape warmed up first); identical tokens and the decode speed.
