import json
import logging
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd


def write_json(items: List[Dict], output_path: str) -> None:
    if output_path == "-":
        # print formatted JSON array to stdout only
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logging.info(f"Saved JSON to {path}")


def write_csv(items: List[Dict], csv_path: str) -> None:
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(items)
    df.to_csv(path, index=False)
    logging.info(f"Saved CSV to {path}")
