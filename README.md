# SubVox Pro — Model Catalog

Online catalog used by the **SubVox Pro** Unity asset. It keeps model lists, prices and
download links up to date without a new release of the package. SubVox Pro falls back to
a bundled copy when offline.

## Files

- [`subvoxpro-models.json`](subvoxpro-models.json) — cloud AI models and their prices
  (translation, text-to-speech, speech-to-text).
- [`subvoxpro-local-models.json`](subvoxpro-local-models.json) — download links and
  checksums for the optional local models.

## Hosted data

- [`chatterbox-multilingual/`](chatterbox-multilingual/NOTICE.md) — Chatterbox Multilingual
  decoder graphs, split for faster loading (no weights).
- [`ipadic/`](ipadic/NOTICE.md) — mecab-ipadic 2.7.0 dictionary converted to UTF-8, used for
  Japanese readings.
- [`qwen3-voices/`](qwen3-voices/NOTICE.md) — twenty synthetic reference voices (CC0 1.0),
  generated locally without any human recordings.
- [`tools/`](tools/) — the scripts used to produce the files above.

Each folder's `NOTICE.md` gives its source and license.
