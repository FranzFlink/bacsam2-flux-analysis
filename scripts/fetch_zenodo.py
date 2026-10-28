"""Download the input files of the Zenodo data record into the places the scripts read them from.

The record holds every input that is neither in this repository nor on PANGAEA, one file per data
set: the ERA5 and CARRA subsets, the MODIS/Aqua ice surface temperature, the merged MODIS-AMSR2
sea-ice concentration, the NASA GIBS images behind Fig. 5, and the leg table (start and end of
every straight leg, cut once from the Polar 6 archive by preprocessing/flight_segments.py).
The measurements come from PANGAEA (pixi run fetch-pangaea).

Every file is checked against the size and SHA-256 checksum listed in data/manifest.tsv (archive
"zenodo:data"), i.e. the file the paper's numbers were computed from; a file that is already present
and correct is not downloaded again.

Run:  pixi run fetch-zenodo
"""
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "manifest.tsv"
RECORD = None                 # Zenodo record id of the data record; set once the record is published
URL = "https://zenodo.org/records/{record}/files/{name}?download=1"
RETRIES = 10
WAIT_S = 30.0

# file name in the Zenodo record -> path in this repository
FILES = {
    "era5_model_levels.nc": "data/reanalysis/era5/ERA5_model_levels/era5_ml.nc",
    "era5_single_levels_instant.nc": "data/reanalysis/era5/ERA5_single_levels/data_stream-oper_stepType-instant.nc",
    "era5_single_levels_avg.nc": "data/reanalysis/era5/ERA5_single_levels/data_stream-oper_stepType-avg.nc",
    "carra_single_levels.nc": "data/reanalysis/carra/carra_east_latlon0p05.nc",
    "carra_height_levels.nc": "data/reanalysis/carra/height_level/carra_latlon0p05.nc",
    "carra_surface_flux_forecast.nc": "data/reanalysis/carra/carra_east_forecast_sf_latlon0p05.nc",
    "modis_aqua_ist_daily.nc":
        "data/observations/processed/satellite_unified/MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc",
    "sic_modis_amsr2_merged.nc":
        "data/observations/processed/satellite_unified/sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc",
    "polar6_legs.csv": "data/observations/ar/analysis_ready_polar6_legs_metadata.csv",
    **{name: f"figures_manuscript/_gibs_cache/{name}" for name in [
        "gibs_20240413_ist.png", "gibs_20240413_truecolor.png",
        "gibs_20240415a_ist.png", "gibs_20240415a_truecolor.png",
        "gibs_20240417a_ist.png", "gibs_20240417a_truecolor.png",
        "gibs_20240417b_ist.png", "gibs_20240417b_truecolor.png",
        "gibs_20240423a_ist.png", "gibs_20240423a_truecolor.png",
        "gibs_20240423b_ist.png", "gibs_20240423b_truecolor.png",
        "gibs_modis_ist_legend.png"]},
}


def manifest() -> dict[str, tuple[int, str]]:
    """path -> (bytes, sha256) of the files the manifest assigns to the Zenodo record."""
    rows = [line.split("\t") for line in MANIFEST.read_text(encoding="utf-8").splitlines()[1:] if line]
    return {r[0]: (int(r[1]), r[2]) for r in rows if r[3] == "zenodo:data"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_correct(path: Path, size: int, checksum: str) -> bool:
    return path.exists() and path.stat().st_size == size and sha256(path) == checksum


def download(url: str, target: Path) -> None:
    """Stream url to target via a .part file; retry on rate limits and server errors."""
    part = target.with_name(target.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as f:
                while block := response.read(1 << 20):
                    f.write(block)
            part.replace(target)
            return
        except urllib.error.HTTPError as err:
            if err.code < 500 and err.code != 429:      # not found / forbidden: retrying will not help
                raise
            print(f"    attempt {attempt}/{RETRIES}: {err}; waiting {WAIT_S:.0f} s")
            time.sleep(WAIT_S)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            print(f"    attempt {attempt}/{RETRIES}: {err}; waiting {WAIT_S:.0f} s")
            time.sleep(WAIT_S)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}")


def main() -> int:
    expected = manifest()
    unlisted = sorted(set(expected) - set(FILES.values()))
    if unlisted:
        sys.exit(f"in data/manifest.tsv as zenodo:data but without a file name in the record: {unlisted}")
    bad = 0
    for name, rel in FILES.items():
        size, checksum = expected[rel]
        target = REPO / rel
        if is_correct(target, size, checksum):
            print(f"ok          {name}")
            continue
        if RECORD is None:
            print(f"MISSING     {name}: the Zenodo record id is not set yet in scripts/fetch_zenodo.py")
            bad += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ({size / 1e6:.1f} MB)")
        download(URL.format(record=RECORD, name=name), target)
        if is_correct(target, size, checksum):
            print(f"ok          {name}")
        else:
            print(f"DIFFERENT   {name}: size or checksum differs from the file the paper used")
            bad += 1
    print(f"{len(FILES) - bad} of {len(FILES)} files present and correct")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
