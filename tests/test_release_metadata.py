"""The release metadata agree: version, title, licence and author in pixi.toml, CITATION.cff and
.zenodo.json. Zenodo reads only .zenodo.json when it archives a GitHub release, GitHub's
"Cite this repository" only CITATION.cff, so the two must not drift apart."""
import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _cff(key):
    text = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    return re.search(rf'^{key}:\s*"?([^"\n]+)"?\s*$', text, flags=re.M).group(1)


def test_version_title_and_licence_agree():
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    pixi = tomllib.loads((REPO / "pixi.toml").read_text(encoding="utf-8"))["workspace"]
    assert pixi["version"] == _cff("version") == zenodo["version"]
    assert _cff("title") == zenodo["title"]
    assert pixi["license"] == _cff("license") == "MIT" and zenodo["license"] == "mit"
    assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_zenodo_creators_are_the_citation_authors():
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    text = (REPO / "CITATION.cff").read_text(encoding="utf-8").split("preferred-citation:")[0]
    authors = re.findall(r"family-names:\s*(.+)\n\s*given-names:\s*(.+)", text)
    assert [c["name"] for c in zenodo["creators"]] == [f"{f.strip()}, {g.strip()}" for f, g in authors]
