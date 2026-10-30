import json
import random
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "ubitquitin/geolocation-geoguessr-images-50k"
COUNTRIES = ["United States", "Japan", "Malaysia", "Ghana"]
IMAGES_PER_COUNTRY = 500

OUTPUT_DIR = Path("data")
CACHE_FILE = Path("dataset_files.json")

api = KaggleApi()


def run_kaggle(function, *args, retries=5, **kwargs):
    for attempt in range(retries):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            message = str(error)

            if "429" in message or "Too Many Requests" in message:
                wait = min(60 * (attempt + 1), 300)
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            raise

    raise RuntimeError("Kaggle rate limit persisted after retries.")


def get_all_files():
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
        print(f"Retrieving page; {len(files)} files collected...")

        result = run_kaggle(
            api.dataset_list_files,
            DATASET,
            page_token=token,
            page_size=200,
        )

        next_token = (
            getattr(result, "next_page_token", None)
            or getattr(result, "nextPageToken", None)
            or None
        )

        page_files = [
            entry.name for entry in result.files # type: ignore
            if entry.name # type: ignore
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
    run_kaggle(
        api.dataset_download_file,
        DATASET,
        filename,
        path=str(destination),
        quiet=True,
    )

    expected_name = Path(filename).name
    downloaded_zip = destination / f"{expected_name}.zip"

    if downloaded_zip.exists() and zipfile.is_zipfile(downloaded_zip):
        with zipfile.ZipFile(downloaded_zip) as archive:
            archive.extractall(destination)
        downloaded_zip.unlink()

    time.sleep(0.2)


def main():
    api.authenticate()

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