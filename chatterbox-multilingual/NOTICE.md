# Chatterbox Multilingual — split conditional decoder graphs

These 24 ONNX graph files are a mechanical split of `onnx/conditional_decoder.onnx` from
[onnx-community/chatterbox-multilingual-ONNX](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX)
at revision `452d3f434aa592098f1eedac9099f33642ab2da5`, produced with
[`tools/split_chatterbox_decoder.py`](../tools/split_chatterbox_decoder.py) (`python split_chatterbox_decoder.py 24`).

The six `conditional_decoder.groupNN.onnx` files, which SubVox Pro downloads, chain these 24 graphs four by four,
unchanged, with [`tools/merge_chatterbox_decoder.py`](../tools/merge_chatterbox_decoder.py)
(`python merge_chatterbox_decoder.py . 6`).

They contain only graph structure: every weight still lives in the unmodified upstream
`conditional_decoder.onnx_data`, downloaded from Hugging Face next to them. Splitting keeps ONNX Runtime
session creation to seconds instead of more than a minute for the 24k-node graph; six groups rather than 24
graphs keep the GPU memory of the DirectML sessions low.

- **Chatterbox** — Copyright (c) 2025 Resemble AI — MIT License — <https://github.com/resemble-ai/chatterbox>
- **ONNX export** — onnx-community — MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
