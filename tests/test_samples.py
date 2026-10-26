"""Check the real sample tracks against samples/manifest.tsv.

Skipped unless the audio has been fetched with scripts/fetch_samples.sh.
Set CLEAN_STEM_ONLINE=1 to also look up lyrics from online providers.
"""

import os
from pathlib import Path

import pytest

from clean_stem.cli import check_file

SAMPLES = Path(__file__).parent.parent / "samples"
ONLINE = os.environ.get("CLEAN_STEM_ONLINE") == "1"


def load_manifest():
    rows = []
    for line in (SAMPLES / "manifest.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sample_id, source, expect, _notes = line.split("\t")
        rows.append(pytest.param(SAMPLES / f"{sample_id}.{source.rsplit('.', 1)[1]}", expect, id=sample_id))
    return rows


@pytest.mark.parametrize("path,expect", load_manifest())
def test_sample_matches_manifest(path, expect):
    if not path.exists():
        pytest.skip("sample not fetched")
    result = check_file(path, online=ONLINE)
    if result["status"] == "no-lyrics":
        pytest.skip("no lyrics found" if ONLINE else "no embedded lyrics (set CLEAN_STEM_ONLINE=1)")
    assert result["status"] == expect
