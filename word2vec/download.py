import os
import requests
import zipfile

DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"


def download_and_extract(url, data_dir, force=False):
    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Get filename from URL
    filename = url.split("/")[-1]
    filepath = os.path.join(data_dir, filename)

    # Download the zip file and save it to disk
    if not os.path.exists(filepath) or force:
        print(f"Downloading {url} to {filepath}")
        r = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print(f"File {filepath} already downloaded")

    # Extract the zip file
    if not os.path.exists(os.path.join(data_dir, "wikitext-2")) or force:
        print(f"Extracting {filepath} to {data_dir}")
        with zipfile.ZipFile(filepath, "r") as f:
            f.extractall(data_dir)
    else:
        print(f"File {filepath} already extracted")
