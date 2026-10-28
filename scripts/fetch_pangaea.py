#!/usr/bin/env python3
"""Download the published PANGAEA files the analysis starts from, into data/pangaea/<dataset>/.

The measurements are published in the bundle "Observations from the BACSAM II aircraft
campaign with Polar 6" (Köhler et al., 2026, doi:10.1594/PANGAEA.995979). This script fetches
the netCDF files of three of its datasets, one file per research flight, about 1.5 GB:

    995975  T-Bird, 100 Hz
    995974  Polar 6 nose boom, 100 Hz
    995980  KT-19, 50 Hz

Every file is checked against the size and SHA-256 checksum of the file the paper's numbers were
computed from; a file that is already present and correct is not downloaded again. PANGAEA
serves some files from tape and then answers HTTP 503 for a while; the script waits and retries.
scripts/prepare_inputs.py (pixi run inputs) then cuts the files to the low-level legs.

Run:  pixi run fetch-pangaea
"""
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PANGAEA = REPO / "data" / "pangaea"
URL = "https://download.pangaea.de/dataset/{dataset}/files/{name}"
RETRIES = 30                  # attempts per file
WAIT_S = 60.0                 # wait between attempts (tape staging takes about a minute)

# (dataset, file name, size in bytes, SHA-256)
FILES = [
    ("995975", "BACSAM2_20240413_RF41_tbird_100Hz.nc", 142492852, "d3aa7d811be684a92ae1644b354eccfe54a866d78015beb083b92d532947db6f"),
    ("995975", "BACSAM2_20240415_RF05_tbird_100Hz.nc", 75292852, "fff81644bf3611d47bf9d7587bff9fc272a6e3d4b673af1a9ffcc7a0f5c03012"),
    ("995975", "BACSAM2_20240417_RF06_tbird_100Hz.nc", 110236852, "90eb79b1228a88e03cfae7683e05de4c1924c657e4fd60854418e878852dc5d8"),
    ("995975", "BACSAM2_20240423_RF07_tbird_100Hz.nc", 104860852, "1061ecad66f40f0826e8eba327a2e6f8a5f50c32ba3386b01f6b5341b05d3f9f"),
    ("995975", "BACSAM2_20240425_RF09_tbird_100Hz.nc", 112252852, "7194d7480bd9e450fdacc455bcd11e5f5420b2af23f41939fdcba883e2dad757"),
    ("995974", "BACSAM2_20240413_RF41_noseboom_100Hz.nc", 172821819, "1e5141913e4ccc163c2acdabd322e190e11b67ebb72000a9a7cfc3f2bc2ef267"),
    ("995974", "BACSAM2_20240415_RF05_noseboom_100Hz.nc", 199785819, "a10aa5599be0fc1a94255fe6828ebae346ff03ba80732c87d4a8f1153067aec5"),
    ("995974", "BACSAM2_20240417_RF06_noseboom_100Hz.nc", 200901819, "db47e12414c3891f316da9058340337ad70b30d840cce95b68325510befb282c"),
    ("995974", "BACSAM2_20240423_RF07_noseboom_100Hz.nc", 192261819, "00a77fd99a289152823180434b7d7261cc76e2c37356f84e91a30d3d4962a8fa"),
    ("995980", "BACSAM2_20240413_RF41_kt19.nc", 34574794, "6fdfdd7a8ef1882baee631320d120a757b3a5be15cb1d7e9c0bf535d9f2216e5"),
    ("995980", "BACSAM2_20240415_RF05_kt19.nc", 39956986, "6c636215feaaacf95d31c20450f8f7c392eb0e81721f247ed0dc8a5c1348d487"),
    ("995980", "BACSAM2_20240417_RF06_kt19.nc", 40185802, "dc734bf063104a489fe6cc8fb6046e286464d9f4c84358b2224494112e269f86"),
    ("995980", "BACSAM2_20240423_RF07_kt19.nc", 38462506, "7f087501e9b12a3a7571c376db0499b692ae5f0a6e3ef3ead014c7223aad601c"),
    ("995980", "BACSAM2_20240424_RF08_kt19.nc", 38884186, "58051be1a659cefb47e23ea80903694ace33b9eafd8c877bd8b404bfedca4339"),
    ("995980", "BACSAM2_20240425_RF09_kt19.nc", 35581450, "53b7587cf07fc91f51895bc9584b2e52a3d39621da88958151f2df59bc1c8e2c"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_correct(path: Path, size: int, checksum: str) -> bool:
    return path.exists() and path.stat().st_size == size and sha256(path) == checksum


def download(url: str, target: Path) -> None:
    """Stream url to target via a .part file; retry while PANGAEA stages the file from tape."""
    part = target.with_name(target.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as f:
                while block := response.read(1 << 20):
                    f.write(block)
            part.replace(target)
            return
        except urllib.error.HTTPError as err:
            if err.code < 500:                    # not found / forbidden: retrying will not help
                raise
            print(f"    attempt {attempt}/{RETRIES}: {err}; waiting {WAIT_S:.0f} s")
            time.sleep(WAIT_S)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            print(f"    attempt {attempt}/{RETRIES}: {err}; waiting {WAIT_S:.0f} s")
            time.sleep(WAIT_S)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}")


def main() -> int:
    bad = 0
    for dataset, name, size, checksum in FILES:
        target = PANGAEA / dataset / name
        if is_correct(target, size, checksum):
            print(f"ok          {dataset}/{name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {dataset}/{name} ({size / 1e6:.0f} MB)")
        download(URL.format(dataset=dataset, name=name), target)
        if is_correct(target, size, checksum):
            print(f"ok          {dataset}/{name}")
        else:
            print(f"DIFFERENT   {dataset}/{name}: size or checksum differs from the file the paper used")
            bad += 1
    print(f"{len(FILES) - bad} of {len(FILES)} files present and correct in {PANGAEA}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
