# SubVox Pro — Remote Model Catalog

Live catalogue of AI models (names + pricing) used by the **SubVox Pro** Unity asset.
It lets the model dropdowns be updated **without republishing the package**: SubVox Pro
fetches this file at launch (and on a manual *Refresh*), caches the result on disk, and
falls back to a bundled copy when offline.

The repository also owns the download manifest for optional local models. This keeps every
large binary URL replaceable without republishing SubVox Pro while retaining a bundled,
known-good fallback in the package.

**Raw URL consumed by the package:**

```
https://raw.githubusercontent.com/exnihilus/subvoxpro-models/main/subvoxpro-models.json
```

**Local-model manifest:**

```
https://raw.githubusercontent.com/exnihilus/subvoxpro-models/main/subvoxpro-local-models.json
```

## How to update

Edit [`subvoxpro-models.json`](subvoxpro-models.json), bump `updatedAt` (and `version` only
on a schema change), commit to `main`. Clients pick it up on their next launch/refresh.

For a local download, edit [`subvoxpro-local-models.json`](subvoxpro-local-models.json).
Keep package IDs and asset paths stable when only replacing a dead mirror. Update `version`,
`size` and `sha256` together when changing the underlying binary. SubVox Pro fetches this
file whenever the user explicitly installs a local model, caches the last valid response in
AppData, and falls back to its bundled copy when offline.

## Local-model format

```jsonc
{
  "version": 1,
  "updatedAt": "2026-07-23",
  "packages": [
    {
      "id": "whisper-large-v3",
      "displayName": "Faster Whisper Large V3",
      "version": "pinned upstream revision",
      "assets": [
        {
          "id": "model",
          "path": "model.bin",
          "url": "https://download-host/model.bin",
          "size": 3087284237,
          "sha256": "..."
        }
      ]
    }
  ]
}
```

- **`id`** — stable identifier consumed by the Unity installer.
- **`version`** — pinned upstream revision or release tag.
- **`path`** — destination relative to that model's AppData directory; traversal outside it is rejected.
- **`url`** — replaceable HTTPS download URL.
- **`size` / `sha256`** — exact integrity checks; partial downloads are resumed and verified before activation.

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
- Local binary URLs belong in `subvoxpro-local-models.json`, not in the pricing catalogue.
- A provider absent from this file keeps the package's bundled defaults for that provider.
- Only the first model of a provider that exposes no dropdown (DeepL, Google Translate,
  TTS/STT today) is used for pricing/display.
