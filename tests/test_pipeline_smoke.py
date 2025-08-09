import json
import os
from pathlib import Path

from keyword_lab.pipeline import run_pipeline
from keyword_lab.schema import validate_items


def test_pipeline_smoke(tmp_path):
    data_dir = Path(__file__).parent / "data"
    items = run_pipeline(
        seed_topic="espresso",
        audience="home baristas",
        sources=str(data_dir),
        output=str(tmp_path / "keywords.json"),
        save_csv=str(tmp_path / "keywords.csv"),
        provider="none",
        verbose=False,
    )

    assert isinstance(items, list)
    assert len(items) > 0

    # validate schema
    validate_items(items)

    # lowercase checks and duplicates within cluster
    seen_per_cluster = {}
    for it in items:
        assert it["keyword"] == it["keyword"].lower()
        assert it["cluster"] == it["cluster"].lower()
        seen_per_cluster.setdefault(it["cluster"], set())
        assert it["keyword"] not in seen_per_cluster[it["cluster"]]
        seen_per_cluster[it["cluster"]].add(it["keyword"])

    # files written
    out_json = tmp_path / "keywords.json"
    out_csv = tmp_path / "keywords.csv"
    assert out_json.exists() and out_json.stat().st_size > 0
    assert out_csv.exists() and out_csv.stat().st_size > 0

    # JSON content is an array
    arr = json.loads(out_json.read_text())
    assert isinstance(arr, list)
    assert len(arr) == len(items)
