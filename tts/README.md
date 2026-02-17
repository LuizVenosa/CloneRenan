# TTS

Quick TTS utilities for the Renan clone.

## Install

```bash
pip install -r requirements-tts.txt
```

## Streaming orchestrator (RealtimeTTS default)

```bash
python tts/run_streaming_chat.py
```

Environment toggles:

- `USE_REALTIMETTS=1` use RealtimeTTS backend (default)
- `RTTTS_ENGINE=system` RealtimeTTS engine class name (ex: `SystemEngine`, `AzureEngine`, `OpenAIEngine`, if installed)
- `RTTTS_ENGINE_KWARGS={"api_key":"...","voice":"..."}`
- `RTTTS_VOICE=<voice-id-or-name>` preferred voice for RealtimeTTS backend
- `RTTTS_RATE=<number>` rate for RealtimeTTS backend (engine-dependent)
- `RTTTS_PITCH=<number>` pitch for RealtimeTTS backend (engine-dependent)
- `USE_REALTIMETTS=0` use Edge fallback backend
- `EDGE_VOICE=pt-BR-AntonioNeural` Edge voice id (when `USE_REALTIMETTS=0`)
- `EDGE_RATE=+10%` Edge speech rate (when `USE_REALTIMETTS=0`)
- `EDGE_PITCH=+0Hz` Edge pitch (when `USE_REALTIMETTS=0`)
- `TTS_OUTPUT_DEVICE=CABLE Input` output device name substring (empty = default device)
- `TWITCH_ENABLED=1` enable Twitch adapter
- `TWITCH_PREFIX=!ask` required ask prefix
- `OBS_ENABLED=1` enable OBS captions/scenes (default ws port `4455`)
- `VTS_ENABLED=1` enable VTube Studio states (default ws port `8001`)
- `PRINT_LOCAL_RESPONSE=1` print final assistant text in local terminal mode

Recommended for more natural PT-BR speech:

```bash
USE_REALTIMETTS=0
EDGE_VOICE=pt-BR-FranciscaNeural
EDGE_RATE=+5%
EDGE_PITCH=+0Hz
```

If you keep `USE_REALTIMETTS=1`, try:

```bash
RTTTS_VOICE=Microsoft Francisca Online (Natural) - Portuguese (Brazil)
RTTTS_RATE=170
RTTTS_PITCH=0
```

Note: RealtimeTTS supports multiple engines; available voices and which settings work depend on the active engine.

List available RTTS engines and local system voices:

```bash
python -m tts.list_realtimetts_options
```

## Qwen3-TTS local test

```bash
python tts/run_qwen_tts.py --text "Olá! Teste de voz."
```

Common options:

- `--model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` model id
- `--mode custom_voice|voice_clone|voice_design`
- `--language Portuguese`
- `--speaker <name>` use a specific CustomVoice speaker
- `--instruct "..."` voice design instruction
- `--ref-audio /path/ref.wav` reference audio for voice clone
- `--ref-text "..."` reference transcript for voice clone
- `--x-vector-only` run voice clone in x-vector-only mode
- `--device-map cuda:0` GPU device map
- `--dtype float16|float32|bfloat16`
- `--attn flash_attention_2` attention implementation (if installed)
- `--output /path/out.wav` save wav to disk
- `--no-play` skip playback (only generate)
