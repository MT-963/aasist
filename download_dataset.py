"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import urllib.request
import zipfile

if __name__ == "__main__":
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    output_path = "./LA.zip"

    print("Downloading LA.zip...")
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")

    print("Extracting LA.zip...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete.")
