# SubVox Pro — Remote Model Catalog

Live catalogue of AI models (names + pricing) used by the **SubVox Pro** Unity asset.
It lets the model dropdowns be updated **without republishing the package**: SubVox Pro
fetches this file at launch (and on a manual *Refresh*), caches the result on disk, and
falls back to a bundled copy when offline.

**Raw URL consumed by the package:**

```
https://raw.githubusercontent.com/exnihilus/subvoxpro-models/main/subvoxpro-models.json
```

## How to update

Edit [`subvoxpro-models.json`](subvoxpro-models.json), bump `updatedAt` (and `version` only
on a schema change), commit to `main`. Clients pick it up on their next launch/refresh.

## Format

```jsonc
{
  "version": 1,              // schema version — bump only when the shape changes
  "updatedAt": "2026-07-22", // free-form date, shown in the editor as "updated on…"

  "translation": [
    { "provider": "OpenAI", "models": [ { "id": "gpt-4o-mini", "priceIn": 0.15, "priceOut": 0.60 } ] }
  ],
  "tts": [ { "provider": "InWorld",  "models": [ { "id": "inworld-tts-2", "priceIn": 25 } ] } ],
  "stt": [ { "provider": "Deepgram", "models": [ { "id": "nova-3", "priceIn": 0.0043 } ] } ]
}
```

Each section is an **array** of `{ "provider": ..., "models": [ ... ] }`. To add a model to a
provider, append an entry to its `models` array; to add a provider, append a new object.

- **`id`** (required) — the exact model identifier sent to the provider API.
- **`priceIn`** (optional) — price shown in the dropdown. For LLMs it is USD per **1M input
  tokens**; for DeepL/Google Translate USD per **1M characters**; for TTS USD per **1M
  characters**; for STT USD per **audio hour**. The editor appends the right unit.
- **`priceOut`** (optional) — USD per **1M output tokens** (LLM providers only).

Prices are optional — a model with no price simply shows its name. They exist to make the
dropdowns informative.

## Notes

- **Provider keys must match** the enum names in the package (`OpenAI`, `Claude`, `Gemini`,
  `DeepSeek`, `DeepL`, `GoogleTranslate`, `InWorld`, `ElevenLabs`, `GoogleCloud`, `Azure`,
  `AssemblyAI`, `Gladia`, `Deepgram`, `Speechmatics`, `RevAI`).
- **Local providers** (Local Whisper, Kokoro) are intentionally **absent** — they need no
  internet and are defined inside the package as a permanent fallback.
- A provider absent from this file keeps the package's bundled defaults for that provider.
- Only the first model of a provider that exposes no dropdown (DeepL, Google Translate,
  TTS/STT today) is used for pricing/display.
