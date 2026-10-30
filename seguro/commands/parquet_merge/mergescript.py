# SPDX-FileCopyrightText: 2026 Henrik Hörter
# SPDX-License-Identifier: Apache-2.0

from main import parquet_merger
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import argparse
from datetime import datetime
from itertools import groupby

from seguro.common import config

DAILY_DIR = Path("merged/daily")
WEEKLY_DIR = Path("merged/weekly")
MONTHLY_DIR = Path("merged/monthly")

WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 0. setup arguments
    # 1. download entire store in daily format
    # 2. use daily files to merge into weekly and monthly files
    ap = argparse.ArgumentParser(description="Reuse downloaded files when merging into multiple formats")
    ap.add_argument("--start", required=True, type=str, help="Start-Timestamp in ISO 8601 format")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--end", type=str)
    group.add_argument("-d", "--duration", type=str, help="Duration starting from start-timestamp")
    ap.add_argument(
        "-g",
        "--gateway",
        default="demo-data",
        type=str,
        help="Gateway to merge sample data from. String in format <location-id>/<device_id>/<point_id>",
    )
    ap.add_argument(
        "-p",
        "--path",
        default="merged/",
        type=str,
        help="Path or directory where to store merged file",
    )
    ap.add_argument(
        "-r",
        "--recurrence",
        default="daily",
        type=str,
        help="Smallest size of merged files: daily, weekly or monthly (day, week, month)",
    )
    merge_args = ap.parse_args()

    args = argparse.Namespace(start=merge_args.start, end=merge_args.end, duration=merge_args.duration, gateway=merge_args.gateway, path=merge_args.path, recurrence=merge_args.recurrence)

    parquet_merger(args)

    def extract_calendar_date(key: Path) -> datetime:
        return datetime.fromisoformat(key.stem)

    def extract_isocalendar(pair: tuple[datetime, Path]):
        iso_year, iso_week, _ = pair[0].isocalendar()
        return (iso_year, iso_week)

    def extract_year_month(pair: tuple[datetime, Path]):
        date_obj = pair[0]
        return (date_obj.year, date_obj.month)

    def concat_and_save(file_paths: list[Path], destination: Path):

        tables = [pq.read_table(file_path) for file_path in file_paths]
        merged_table = pa.concat_tables(tables)
        pq.write_table(merged_table, destination)
        print(f"Saved: {destination}")


    daily_files = sorted(list(DAILY_DIR.rglob("*.parquet")))
    pairs = [(extract_calendar_date(file_path), file_path) for file_path in daily_files]

    print("--- Creating Weekly Files ---")
    for (iso_year, iso_week), group in groupby(pairs, key=extract_isocalendar):
        paths = [path for _, path in group]
        output_file = WEEKLY_DIR / f"{iso_year}_CW_{iso_week:02d}.parquet"
        concat_and_save(paths, output_file)

    print("--- Creating Monthly Files ---")
    for (year, month), group in groupby(pairs, key=extract_year_month):
        paths = [path for _, path in group]
        output_file = MONTHLY_DIR / f"{year}_{month:02d}.parquet"
        concat_and_save(paths, output_file)

    return

if __name__ == "__main__":
    main()
