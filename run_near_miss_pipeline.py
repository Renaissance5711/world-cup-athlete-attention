#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from ijsms.near_miss_workflow import run_near_miss_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description='Rebuild the StatsBomb shot-opportunity panel and near-miss results.')
    parser.add_argument('--output-dir', default='outputs/near_miss_rebuilt')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = run_near_miss_workflow(workspace=root, output_dir=root / args.output_dir)
    for key, path in paths.items():
        print(f'{key}: {path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
