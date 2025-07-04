def hijack_imports() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(-1, str(Path(__file__).parent))
