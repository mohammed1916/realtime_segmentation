#!/usr/bin/env python3
"""Load a val_iou parquet/csv from outputs/run_<ts>/ and pretty-print it.

Usage: python tools/show_val_iou_parquet.py [path_to_parquet]
If no path provided, the script finds the most recent run_*/val_iou_*.parquet
"""
from pathlib import Path
import sys
import pandas as pd


def find_latest_parquet():
    out = Path('outputs')
    runs = sorted(out.glob('run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        files = list(r.glob('val_iou_*.parquet')) + list(r.glob('val_iou_*.csv'))
        if files:
            return files[0]
    return None


def main():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        p = find_latest_parquet()
        if p is None:
            print('No parquet/csv found under outputs/run_*')
            return 1
    print('Loading', p)
    if p.suffix == '.parquet':
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    print('\nPer-class IoU table:')
    print(df.to_string(index=False))
    print('\nSummary:')
    metaf = p.parent / f"{p.stem}.metadata.json"
    if metaf.exists():
        import json
        print(json.loads(metaf.read_text()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
