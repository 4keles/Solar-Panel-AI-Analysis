"""Download solar-panel-od model weights from Hugging Face Hub."""

import argparse
from pathlib import Path


REPO_ID = "4keles/solar-panel-od"
DEFAULT_VERSION = "v1.2.1"
DEFAULT_FORMAT = "onnx"


def download(version: str, fmt: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename = f"{version}/best.{fmt}"
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=dest,
    )
    return Path(local_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download solar-panel-od model from HF Hub")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Model version (e.g. v1.2.1)")
    parser.add_argument("--format", default=DEFAULT_FORMAT, choices=["onnx", "pt"], dest="fmt")
    parser.add_argument("--dest", default="models/downloaded", help="Local destination directory")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.version}/best.{args.fmt} from {REPO_ID} ...")
    path = download(args.version, args.fmt, dest)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
