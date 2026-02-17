import argparse
import os
import sys

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tts.tts_qwen_engine import QwenTTSEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS locally")
    parser.add_argument("--text", default="Olá! Este é um teste de voz.")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--mode", choices=["custom_voice", "voice_clone", "voice_design"], default="custom_voice")
    parser.add_argument("--language", default="Portuguese")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--ref-audio", dest="ref_audio", default=None)
    parser.add_argument("--ref-text", dest="ref_text", default=None)
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--attn", default=None)
    parser.add_argument("--output", default=None, help="Optional output wav path")
    parser.add_argument("--device", default="CABLE Input", help="Audio output device name")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--no-play", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    engine = QwenTTSEngine(
        output_device_name=args.device,
        enable_monitor=not args.no_monitor,
        model_id=args.model,
        mode=args.mode,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )

    if args.output:
        engine.save_wav(args.text, args.output)
        print(f"Saved: {args.output}")

    if not args.no_play:
        engine.speak(args.text)


if __name__ == "__main__":
    main()
