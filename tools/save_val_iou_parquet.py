#!/usr/bin/env python3
"""Parse outputs/val_iou_table.txt and save the per-class IoU table as Parquet

Creates outputs/run_<timestamp>/val_iou_<timestamp>.parquet and a small
metadata.json with summary metrics.
"""
from pathlib import Path
import re
import json
from datetime import datetime
import pandas as pd


def parse_table(lines):
    # Find the table start (header) and end
    header_idx = None
    for i, l in enumerate(lines):
        if re.search(r"\|\s*Class\s*\|\s*IoU\s*\|\s*Acc", l):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError('Could not find IoU table header in log')

    # rows follow; table ends when a line with '+---' appears after header's surrounding lines
    rows = []
    for l in lines[header_idx+2:]:
        if l.strip().startswith('+') and '---' in l:
            break
        if l.strip().startswith('|'):
            parts = [p.strip() for p in l.strip().split('|')]
            # parts: ['', 'Class', 'IoU', 'Acc', '']  or ['', ' name ', ' val ', ' val ', '']
            if len(parts) >= 4:
                cls = parts[1]
                iou = parts[2]
                acc = parts[3]
                # skip header line if present
                if cls.lower().startswith('class'):
                    continue
                rows.append((cls, iou, acc))
    df = pd.DataFrame(rows, columns=['class', 'iou', 'acc'])
    # convert numeric columns
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    df['iou'] = df['iou'].map(to_float)
    df['acc'] = df['acc'].map(to_float)
    return df


def parse_summary(lines):
    text = '\n'.join(lines)
    # aAcc: 37.6300  mIoU: 1.9800  mAcc: 5.2600
    m = re.search(r'aAcc:\s*([0-9.]+)\s*mIoU:\s*([0-9.]+)\s*mAcc:\s*([0-9.]+)', text)
    if m:
        return {'aAcc': float(m.group(1)), 'mIoU': float(m.group(2)), 'mAcc': float(m.group(3))}
    return {}


def main():
    logp = Path('outputs/val_iou_table.txt')
    if not logp.exists():
        print(f'Log file not found: {logp.resolve()}')
        return 1
    lines = logp.read_text(encoding='utf-8', errors='ignore').splitlines()

    df = parse_table(lines)
    meta = parse_summary(lines)

    # timestamp from file mtime
    ts = datetime.fromtimestamp(logp.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
    outdir = Path('outputs') / f'run_{ts}'
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f'val_iou_{ts}.parquet'

    # write parquet (require pandas with pyarrow or fastparquet available)
    try:
        df.to_parquet(outp, index=False)
    except Exception as e:
        # fallback: write csv and then write parquet via pyarrow if available
        csvp = outdir / f'val_iou_{ts}.csv'
        df.to_csv(csvp, index=False)
        try:
            import pyarrow as pa, pyarrow.parquet as pq
            table = pa.Table.from_pandas(df)
            pq.write_table(table, outp)
        except Exception:
            print('Failed to write parquet (missing engine). Saved CSV instead at', csvp)
            outp = csvp

    # write metadata
    meta_p = outdir / f'val_iou_{ts}.metadata.json'
    meta_d = {'generated_at': datetime.now().isoformat(), 'source_log': str(logp), 'metrics': meta}
    meta_p.write_text(json.dumps(meta_d, indent=2))

    print('Wrote:', outp)
    print('Metadata:', meta_p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
