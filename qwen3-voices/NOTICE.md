# Qwen3 synthetic reference voices

ExNihilus dedicates these twenty synthetic WAV recordings and the original texts and voice descriptions
in voices.json to the public domain under CC0 1.0 Universal, to the extent any rights apply.
Commercial use, redistribution, editing and voice cloning are permitted without attribution.
The complete CC0 legal code is in LICENSE.txt.

The references were generated locally with Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign from original descriptions,
without supplying a human reference recording or requesting the imitation of a named person.
No Gemini API or other hosted speech provider was used. These voices do not identify actual speakers.

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian each have
one female and one male reference. Regional targets: Mandarin Chinese, General American English,
standard German/French/Russian/Italian, Spain Spanish, Brazilian Portuguese, Tokyo Japanese and Seoul Korean.
Gender and accent labels describe the intended synthetic performance.

Provenance: voices.json records model/tokenizer hashes, upstream conversion and native engine revisions,
generation descriptions, original transcripts, output SHA-256 hashes and durations. The model weights
are Apache 2.0 (QWEN3-TTS-MODEL-LICENSE.txt), and the native engine is MIT; CC0 applies to this reference pack.
The original model source is https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign.

quality.json records local Whisper large-v3-turbo transcript and waveform checks of every reference and
its Base clone. Those automated checks establish usable speech and expected text; they do not certify
subjective accent or performance quality in every language. References remain available for audition.

Generated with SubVox Pro's developer tool Tools/Qwen3Tts/VoicePack/Program.cs and requests.json.
Users only download this small pack into AppData; no generator, Python or external server is required.
