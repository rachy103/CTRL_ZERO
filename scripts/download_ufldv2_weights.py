from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_ZOO = {
    "culane_res18": {
        "file_id": "1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W",
        "filename": "culane_res18.pth",
        "f1": "75.0",
    },
    "culane_res34": {
        "file_id": "1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj",
        "filename": "culane_res34.pth",
        "f1": "76.0",
    },
    "tusimple_res18": {
        "file_id": "1Clnj9-dLz81S3wXiYtlkc4HVusCb978t",
        "filename": "tusimple_res18.pth",
        "f1": "96.11",
    },
    "tusimple_res34": {
        "file_id": "1pkz8homK433z39uStGK3ZWkDXrnBAMmX",
        "filename": "tusimple_res34.pth",
        "f1": "96.24",
    },
    "curvelanes_res18": {
        "file_id": "1VfbUvorKKMG4tUePNbLYPp63axgd-8BX",
        "filename": "curvelanes_res18.pth",
        "f1": "80.42",
    },
    "curvelanes_res34": {
        "file_id": "1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN",
        "filename": "curvelanes_res34.pth",
        "f1": "81.34",
    },
}


def download_google_drive_file(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    url = "https://drive.google.com/uc"
    response = session.get(url, params={"export": "download", "id": file_id}, stream=True, timeout=30)
    response.raise_for_status()

    token = _confirm_token(response)
    if token:
        response = session.get(
            url,
            params={"export": "download", "confirm": token, "id": file_id},
            stream=True,
            timeout=30,
        )
        response.raise_for_status()
    elif _looks_like_html(response):
        confirm_url, params = _confirm_form(response.text, response.url)
        if confirm_url and params:
            response = session.get(confirm_url, params=params, stream=True, timeout=30)
            response.raise_for_status()

    total = int(response.headers.get("content-length", "0") or "0")
    downloaded = 0
    with destination.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100.0
                print(f"\rDownloading {destination.name}: {pct:5.1f}%", end="", flush=True)
    print()

    if destination.stat().st_size < 1024 * 1024:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is unexpectedly small: {destination}. "
            "Google Drive may have returned an HTML warning page instead of the checkpoint."
        )


def _confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _looks_like_html(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    return "text/html" in content_type


def _confirm_form(text: str, base_url: str) -> tuple[str | None, dict[str, str] | None]:
    form_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>', text, re.DOTALL)
    if not form_match:
        return None, None

    action = html.unescape(form_match.group(1))
    form_body = form_match.group(2)
    inputs = re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', form_body)
    params = {html.unescape(name): html.unescape(value) for name, value in inputs}
    return urljoin(base_url, action), params


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official UFLDv2 checkpoints.")
    parser.add_argument("--model", choices=sorted(MODEL_ZOO), default="culane_res34")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "models" / "ufldv2")
    args = parser.parse_args()

    model = MODEL_ZOO[args.model]
    destination = args.output_dir / model["filename"]
    if destination.exists():
        size = destination.stat().st_size
        if size >= 1024 * 1024:
            print(f"Already exists: {destination}")
            return
        print(f"Removing incomplete download: {destination} ({size} bytes)")
        destination.unlink()

    print(f"Downloading {args.model} (official reported F1={model['f1']}) to {destination}")
    download_google_drive_file(model["file_id"], destination)
    print(f"Done: {destination}")


if __name__ == "__main__":
    main()
