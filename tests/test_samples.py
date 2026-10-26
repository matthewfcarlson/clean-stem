"""Check the real sample tracks against samples/manifest.tsv.

Skipped unless the audio has been fetched with scripts/fetch_samples.sh.
"""

from pathlib import Path

import pytest

from clean_stem.cli import check_file

SAMPLES = Path(__file__).parent.parent / "samples"


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
    result = check_file(path)
    if result["status"] == "no-lyrics":
        pytest.skip("no embedded lyrics; needs transcription")
    assert result["status"] == expect
