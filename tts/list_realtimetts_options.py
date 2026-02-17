import inspect


def main() -> None:
    try:
        import RealtimeTTS  # type: ignore
    except Exception as exc:
        print(f"RealtimeTTS import failed: {exc}")
        return

    print("RealtimeTTS engines:")
    engine_names = []
    for attr in sorted(dir(RealtimeTTS)):
        obj = getattr(RealtimeTTS, attr, None)
        if inspect.isclass(obj) and attr.endswith("Engine"):
            engine_names.append(attr)
    if not engine_names:
        print("  (none found)")
    else:
        for name in engine_names:
            cls = getattr(RealtimeTTS, name)
            try:
                sig = str(inspect.signature(cls))
            except Exception:
                sig = "(...)"
            print(f"  - {name}{sig}")

    print("\nSystem voices (pyttsx3):")
    try:
        import pyttsx3  # type: ignore
    except Exception as exc:
        print(f"  pyttsx3 unavailable: {exc}")
        return

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        if not voices:
            print("  (no voices returned)")
            return
        for v in voices:
            vid = getattr(v, "id", "")
            name = getattr(v, "name", "")
            langs = getattr(v, "languages", [])
            print(f"  - id={vid} | name={name} | languages={langs}")
    except Exception as exc:
        print(f"  failed to enumerate voices: {exc}")


if __name__ == "__main__":
    main()
