#!/usr/bin/env python3
"""Checksum manifest of the data files (data/manifest.tsv), and where each file is archived.

The paper's numbers were computed from exactly these bytes. The manifest records, for every
data file, its size, its SHA-256 checksum and the archive it is published in, so that a file
downloaded from PANGAEA or Zenodo can be checked against the file the numbers came from
(DATA_PUBLICATION.md describes the publication of the data).

Archive column:
  pangaea:published   already published on PANGAEA (source: its DOI)
  pangaea:new         derived from the measurements; to be published on PANGAEA by the authors
  zenodo:data         subsets of third-party gridded products; to be published on Zenodo
  regenerated         written by the pipeline (pixi run precompute); not archived
  unused              not read by any script; not archived

Run:
    pixi run data-manifest            check every listed file (size and checksum)
    pixi run data-manifest --write    recompute size and checksum; new files get the archive
                                      of the first matching rule below, existing rows keep theirs
"""
import fnmatch
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "manifest.tsv"
COLUMNS = ("path", "bytes", "sha256", "archive", "source")
LOCATIONS = ("data/**/*", "figures_manuscript/_gibs_cache/*")

# (pattern, archive, source) for files that are not yet in the manifest; first match wins
RULES = [
    ("data/manifest.tsv", None, None),
    ("data/observations/raw/polar6/mastertracks/*.tab", "pangaea:published",
     "doi:10.1594/PANGAEA.971763 (master tracks, Jurányi et al., 2024)"),
    ("data/observations/processed/t-bird_unified/*", "pangaea:new",
     "leg-window subset of doi:10.1594/PANGAEA.995975 (T-Bird 100 Hz); see DATA_PUBLICATION.md, decision D1"),
    ("data/observations/processed/polar6_unified/noseboom_unified.nc", "regenerated",
     "cut from doi:10.1594/PANGAEA.995974 (Polar 6 nose boom 100 Hz) by pixi run inputs"),
    ("data/observations/processed/polar6_unified/*", "pangaea:new",
     "KT-19 used from it; source doi:10.1594/PANGAEA.995980 (KT-19 50 Hz); see DATA_PUBLICATION.md, decision D2"),
    ("data/observations/ar/along_track_inputs.nc", "zenodo:data",
     "ERA5, MODIS and CARRA values at the segments (pixi run extract)"),
    ("data/observations/ar/*", "pangaea:new", "derived from the BACSAM-II measurements"),
    ("data/observations/processed/satellite_unified/*", "zenodo:data", "MODIS / University of Bremen, regridded subset"),
    ("data/reanalysis/*", "zenodo:data", "ERA5 / CARRA (Copernicus), regional subset"),
    ("figures_manuscript/_gibs_cache/*", "zenodo:data", "NASA GIBS / Worldview imagery (background of Fig. 5)"),
    ("data/model_output/*", "regenerated", "pixi run precompute"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    return {row["path"]: row for row in (dict(zip(COLUMNS, line.split("\t"))) for line in lines[1:] if line)}


def rule(path: str):
    for pattern, archive, source in RULES:
        if fnmatch.fnmatch(path, pattern):
            return archive, source
    return "unassigned", ""


def write() -> None:
    old = read()
    found = sorted({p.relative_to(REPO).as_posix() for loc in LOCATIONS for p in REPO.glob(loc) if p.is_file()}
                   - {p.relative_to(REPO).as_posix() for p in (REPO / "data" / "pangaea").glob("**/*")})  # downloads
    rows = []
    for path in found:
        archive, source = (old[path]["archive"], old[path]["source"]) if path in old else rule(path)
        if archive is None:
            continue
        p = REPO / path
        rows.append((path, str(p.stat().st_size), sha256(p), archive, source))
    MANIFEST.write_text("\t".join(COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST.relative_to(REPO)}: {len(rows)} files")
    for archive in sorted({r[3] for r in rows}):
        sel = [r for r in rows if r[3] == archive]
        print(f"  {archive:<18} {len(sel):3d} files  {sum(int(r[1]) for r in sel) / 1e6:7.1f} MB")


def check() -> int:
    rows = read()
    bad = 0
    for path, row in rows.items():
        p = REPO / path
        if not p.exists():
            status = "missing" if row["archive"] != "regenerated" else "missing (run pixi run precompute)"
        elif row["archive"] == "regenerated":
            status = "ok (regenerated; not compared)"
        elif str(p.stat().st_size) != row["bytes"] or sha256(p) != row["sha256"]:
            status = "DIFFERENT"
        else:
            status = "ok"
        bad += status in ("missing", "DIFFERENT")
        print(f"{status:<34} {path}")
    print(f"\n{len(rows) - bad} of {len(rows)} files as in data/manifest.tsv" + ("" if not bad else f"; {bad} not"))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        write()
    else:
        sys.exit(check())
