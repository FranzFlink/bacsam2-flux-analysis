#!/usr/bin/env python3
"""Download the published PANGAEA files the analysis starts from, into data/pangaea/<dataset>/.

The measurements are published in the bundle "Observations from the BACSAM II aircraft
campaign with Polar 6" (Köhler et al., 2026, doi:10.1594/PANGAEA.995979). This script fetches
the netCDF files of three of its datasets, one file per research flight, about 1.5 GB:

    995975  T-Bird, 100 Hz
    995974  Polar 6 nose boom, 100 Hz
    995980  KT-19, 50 Hz

It also fetches the 13 Polar 6 master tracks of the campaign (Jurányi et al., 2024,
doi:10.1594/PANGAEA.971763; one dataset per flight, as PANGAEA text export) into
data/observations/raw/polar6/mastertracks/, which Fig. 1 reads.

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

TRACK_URL = "https://doi.pangaea.de/10.1594/PANGAEA.{dataset}?format=textfile"
TRACKS = REPO / "data" / "observations" / "raw" / "polar6" / "mastertracks"
MASTERTRACKS = [
    ("971737", "P6-247_BACSAM2_2024_2404070101_mastertrack.tab", 26262, "c1724a7976c4ffccafd100a6ba8ca7e66e455788186f3c31299c6780a8c504aa"),
    ("971740", "P6-247_BACSAM2_2024_2404100201_mastertrack.tab", 27713, "1096952779f7044f1103ccec420e6946cde004f1a576a85a23750f1457b8462e"),
    ("971741", "P6-247_BACSAM2_2024_2404110301_mastertrack.tab", 24598, "ea2e59273d5cbad6f2af7cf7dcbec53ed2f5d16bf1438080d114292f3cff157f"),
    ("971742", "P6-247_BACSAM2_2024_2404130401_mastertrack.tab", 17562, "c316611dbdbdfa343b57e560ee9ee6b65f4580f9d9a804c92a1007cf4f65953b"),
    ("971739", "P6-247_BACSAM2_2024_2404130502_mastertrack.tab", 25756, "803e294f669335c0622254512d211d439eae51f2bed30ba398be5d120cb2a6e6"),
    ("971743", "P6-247_BACSAM2_2024_2404130603_mastertrack.tab", 16073, "eb0fc5eec30e13091601a96e09150b73ec7d100b27f4fbda60c991b4875923e0"),
    ("971747", "P6-247_BACSAM2_2024_2404150701_mastertrack.tab", 28368, "94bb33389408050676f81bfba0d1e8f4b6e83ecbbdd71e6c5a6656059682a2a5"),
    ("971735", "P6-247_BACSAM2_2024_2404170801_mastertrack.tab", 29153, "bd348e31a03a8a31fcce5e5462cf4aa90ab35877ebb413f6accda0330742c257"),
    ("971738", "P6-247_BACSAM2_2024_2404230901_mastertrack.tab", 27124, "44e4c70cc40b5d87f14d7a7a846c0e42a65d2b6f2435bfc8598bb7ac0a424e69"),
    ("971745", "P6-247_BACSAM2_2024_2404241001_mastertrack.tab", 27841, "e7400fbca62455846025adb27d6f24b2afd265ee9fbffb5217f74c7d77bff695"),
    ("971736", "P6-247_BACSAM2_2024_2404251101_mastertrack.tab", 25200, "10e271164640032e55ce756bd6411484df6a8b698ec575cad5255baa493ba0bc"),
    ("971744", "P6-247_BACSAM2_2024_2404261201_mastertrack.tab", 27993, "835d38225b7989e9f0cd3dd92f57bb8f48b433b389c06926bb37c7816fa0f0cd"),
    ("971746", "P6-247_BACSAM2_2024_2404291301_mastertrack.tab", 19370, "cf8785a0e5282f30521639eca6b42b9a60ef24f0bab35497f1160a07a410f38f"),
]


def entries():
    """(url, target, label, size, sha256) of every file: measurements first, then master tracks."""
    for dataset, name, size, checksum in FILES:
        yield URL.format(dataset=dataset, name=name), PANGAEA / dataset / name, f"{dataset}/{name}", size, checksum
    for dataset, name, size, checksum in MASTERTRACKS:
        yield TRACK_URL.format(dataset=dataset), TRACKS / name, f"{dataset}/{name}", size, checksum


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
    bad = n = 0
    for url, target, label, size, checksum in entries():
        n += 1
        if is_correct(target, size, checksum):
            print(f"ok          {label}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {label} ({size / 1e6:.1f} MB)")
        download(url, target)
        if is_correct(target, size, checksum):
            print(f"ok          {label}")
        else:
            print(f"DIFFERENT   {label}: size or checksum differs from the file the paper used")
            bad += 1
    print(f"{n - bad} of {n} files present and correct")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
