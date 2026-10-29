import csv
import io
import json
import random
import re
import subprocess
import time
import zipfile
from pathlib import Path

DATASET = "ubitquitin/geolocation-geoguessr-images-50k"
COUNTRIES = ["United States", "Japan", "Malaysia", "Ghana"]
IMAGES_PER_COUNTRY = 100

OUTPUT_DIR = Path("data")
CACHE_FILE = Path("dataset_files.json")


def run_kaggle(args, retries=5):
    """Run Kaggle CLI with retries for rate limits."""
    for attempt in range(retries):
        result = subprocess.run(
            ["kaggle", *args],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return result.stdout

        message = result.stderr + result.stdout

        if "429" in message or "Too Many Requests" in message:
            wait = min(60 * (attempt + 1), 300)
            print(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        raise RuntimeError(message)

    raise RuntimeError("Kaggle rate limit persisted after retries.")


def get_all_files():
    """Retrieve and cache the complete paginated file listing."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

        files = cache.get("files", [])
        if cache.get("complete"):
            print(f"Using cached listing: {len(files)} files.")
            return files

        print(f"Resuming listing from {len(files)} cached files.")
    else:
        cache = {"files": [], "page_token": None, "complete": False}
        files = []

    token = cache.get("page_token")

    while True:
        args = [
            "datasets", "files", DATASET,
            "--page-size", "200",
            "--csv",
        ]

        if token:
            args += ["--page-token", token]

        print(f"Retrieving page; {len(files)} files collected...")

        output = run_kaggle(args)

        match = re.search(r"Next Page Token\s*=\s*(\S+)", output)
        next_token = match.group(1) if match else None

        lines = output.splitlines()
        header = next(
            (
                i for i, line in enumerate(lines)
                if line.strip().lower().startswith("name,")
            ),
            None,
        )

        if header is None:
            raise RuntimeError(
                "Could not parse Kaggle CSV output:\n" + output[:1000]
            )

        reader = csv.DictReader(io.StringIO("\n".join(lines[header:])))
        page_files = [
            row["name"] for row in reader
            if row.get("name")
        ]

        files.extend(page_files)

        cache = {
            "files": files,
            "page_token": next_token,
            "complete": next_token is None,
        }

        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

        print(f"Collected {len(files)} filenames.")

        if not next_token:
            break

        token = next_token
        time.sleep(2)

    return files


def download_file(filename, destination):
    """Download an individual dataset file."""
    run_kaggle([
        "datasets", "download",
        "-d", DATASET,
        "-f", filename,
        "-p", str(destination),
    ])

    # Kaggle may package individual downloads as ZIP files.
    expected_name = Path(filename).name
    downloaded_zip = destination / f"{expected_name}.zip"

    if downloaded_zip.exists() and zipfile.is_zipfile(downloaded_zip):
        with zipfile.ZipFile(downloaded_zip) as archive:
            archive.extractall(destination)
        downloaded_zip.unlink()

    time.sleep(0.2)


def main():
    files = get_all_files()
    print(f"\nTotal filenames collected: {len(files)}")

    for country in COUNTRIES:
        print(f"\nProcessing {country}...")

        prefix = f"compressed_dataset/{country}/"

        matching = [
            name for name in files
            if name.startswith(prefix)
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not matching:
            print(f"No images found for {country}.")
            continue

        random.seed(42)
        selected = random.sample(
            matching,
            min(IMAGES_PER_COUNTRY, len(matching)),
        )

        destination = OUTPUT_DIR / country
        destination.mkdir(parents=True, exist_ok=True)

        for index, filename in enumerate(selected, 1):
            expected = destination / Path(filename).name

            if expected.exists() and expected.stat().st_size > 0:
                print(f"Already downloaded {index}/{len(selected)}")
                continue

            try:
                download_file(filename, destination)
                print(f"Downloaded {index}/{len(selected)}: {expected.name}")
            except Exception as error:
                print(f"Failed: {filename}\n  {error}")

    print("\nDownload process finished. Check the data/ directory.")


if __name__ == "__main__":
    main()