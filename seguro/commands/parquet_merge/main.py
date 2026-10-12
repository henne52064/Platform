# SPDX-FileCopyrightText: 2026 Henrik Hörter
# SPDX-License-Identifier: Apache-2.0
import cProfile
import pstats

import pyarrow.parquet as pq
import pyarrow as pa
import io

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import re
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone

from seguro.common import config
from seguro.common.store import Client

LOCAL_TMP = ".LOCAL/"
REMOTE_FOLDER = "data/measurements/"
MERGED_DIR = "merged/"
_thread_local = threading.local()


def resolve_timeframe(start_input: str, end_or_duration) -> tuple[datetime, datetime]:

    start_dt = datetime.fromisoformat(start_input)

    if start_dt.tzinfo is None:  # check for timezone
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    if isinstance(end_or_duration, timedelta):
        end_dt = start_dt + end_or_duration  # is duration
    else:
        end_dt = datetime.fromisoformat(end_or_duration)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

    return start_dt, end_dt


def fetch_candidates(s: Client, start_dt: datetime, end_dt: datetime, args) -> list[str]:
    """
    fetch candidate object keys from store within [start_dt, end_dt] bounds
    stops requesting additional pages as soon as an object key exceeds end_dt
    """

    if args.gateway.endswith("/"):
        remotepath = args.gateway
    else:
        remotepath = args.gateway + "/"

    start_after_key = f"{REMOTE_FOLDER + remotepath}{(start_dt - timedelta(seconds=1)).isoformat()}.parquet"
    end_key = f"{REMOTE_FOLDER + remotepath}{end_dt.isoformat()}.parquet"

    objects = s.client.list_objects(
        bucket_name=s.bucket,
        prefix=REMOTE_FOLDER + remotepath,
        start_after=start_after_key,
    )

    candidate_keys = []

    for object in objects:
        key = object.object_name
        # EXIT LIMIT: Stop as soon as keys exceed end_dt
        if key > end_key:
            break

        candidate_keys.append(key)

    if not candidate_keys:
        print("candidate keys empty")

    return candidate_keys


def parse_duration(value: str) -> timedelta:
    matches = re.findall(r"(\d+)(d|h|min|s)", value)

    if not matches:
        raise ValueError("Duration must look like '5h', '30min', '1h30min', or '2d4h15min'")

    total = timedelta()

    for amount_str, unit in matches:
        amount = int(amount_str)

        if unit == "d":
            total += timedelta(days=amount)
        elif unit == "h":
            total += timedelta(hours=amount)
        elif unit == "min":
            total += timedelta(minutes=amount)
        elif unit == "s":
            total += timedelta(seconds=amount)
    return total


def get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = Client()
    return _thread_local.client


def process_and_merge(
    candidates: list[str],
    client: Client,
    output_path: str,
    start_dt: datetime,
    end_dt: datetime,
    epsilon: timedelta,
    max_workers: int = 16,
):
    """
    Downloads parquet files concurrently into memory, validates metadata on the fly,
    and merges them into a single output file using PyArrow.
    """
    tables = [None] * len(candidates)
    sorted_candidates = sorted(candidates)
    filename_dates: list[datetime | None] = [None] * len(candidates)

    def _download_and_parse(idx: int, candidate: str):
        # Fetch directly into bytes buffer
        # work with http response object to avoid disk I/O
        response = client.client.get_object(client.bucket, candidate)

        data_bytes = response.read()
        buffer = io.BytesIO(data_bytes)

        table = pq.read_table(buffer)

        # Fast validation using PyArrow / pandas index metadata without full DataFrame overhead
        df_stub = table.select([]).to_pandas()
        if not df_stub.index.is_monotonic_increasing:
            logging.warning(f"Timestamps in '{candidate}' are not monotonically increasing.")

        return idx, table

    # Parallel download & in-memory parse
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_and_parse, i, candidate) for i, candidate in enumerate(sorted_candidates)]
        for future in as_completed(futures):
            idx, table = future.result()
            tables[idx] = table  # Keep original key order
            clean_name = candidates[idx].removesuffix(".parquet").split("/")[-1]
            filename_dates[idx] = datetime.fromisoformat(clean_name)

    # Check filename timestamps

    dates_series = pd.Series(filename_dates)
    time_diffs = dates_series.diff().dropna()

    median_diff = pd.Timedelta(time_diffs.median())
    threshold = median_diff + epsilon

    first_file = filename_dates[0]
    last_file = filename_dates[-1]

    start_diff = abs(first_file - start_dt)
    end_diff = abs(last_file - end_dt)

    if start_diff > threshold:  # check if first file is close to start datetime
        logging.warning(
            f"Time gap detected between start date '{start_dt}' and first file '{first_file}': "
            f"Difference is {start_diff} (Expected ~{median_diff}, Threshold: {threshold})"
        )
    if end_diff > threshold:  # check if last file is close to end datetime
        logging.warning(
            f"Time gap detected between last file '{last_file}' and end date '{end_dt}': "
            f"Difference is {end_diff} (Expected ~{median_diff}, Threshold: {threshold})"
        )

    for idx, diff in enumerate(time_diffs, start=1):
        if diff > threshold:
            prev_file = filename_dates[idx - 1]
            curr_file = filename_dates[idx]
            logging.warning(
                f"Time gap detected between '{prev_file}' and '{curr_file}': "
                f"Difference is {diff} (Expected ~{median_diff}, Threshold: {threshold})"
            )

    # Combine PyArrow tables zero-copy & write directly
    merged_table = pa.concat_tables(tables)
    pq.write_table(merged_table, output_path)

    print(f"Successfully merged {len(candidates)} files into {output_path}")


def parquet_merger(args):
    client = Client()

    if args.end is not None and args.duration is None:
        start_dt, end_dt = resolve_timeframe(args.start, args.end)

    if args.end is None and args.duration is not None:
        start_dt, end_dt = resolve_timeframe(args.start, parse_duration(args.duration))

    print(f"start: {start_dt.isoformat()}, end: {end_dt.isoformat()}")

    candidates = fetch_candidates(client, start_dt, end_dt, args)

    if len(candidates) == 0:
        print("No files found to merge")
        return

    process_and_merge(
        candidates=candidates,
        client=client,
        output_path=args.path,
        start_dt=start_dt,
        end_dt=end_dt,
        epsilon=parse_duration(args.epsilon),
    )

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="parquet_merge", description="Combine multiple data files in parquet format to one file"
    )

    parser.add_argument(
        "-l",
        "--log-level",
        default="debug" if config.DEBUG else "info",
        help="Logging level",
        choices=["debug", "info", "warn", "error", "critical"],
    )

    parser.add_argument("-s", "--start", required=True, type=str, help="Start-Timestamp in ISO 8601 format")

    group = parser.add_mutually_exclusive_group(required=True)  # allows to either have --end or --duration flag
    group.add_argument("-e", "--end", type=str, help="End-Timestamp in ISO 8601 format")
    group.add_argument("-d", "--duration", type=str, help="Duration starting from start-timestamp")

    parser.add_argument(
        "-g",
        "--gateway",
        default="demo-data",
        type=str,
        help="Gateway to merge sample data from. String in format <location-id>/<device_id>/<point_id>",
    )

    parser.add_argument(
        "-p", "--path", default=MERGED_DIR + "merged.parquet", type=str, help="Path where to store merged file"
    )

    parser.add_argument(
        "--epsilon",
        default="30s",
        type=str,
        help="Set maximum deviation of timegap between files before giving out a warning",
    )

    parser.set_defaults(func=parquet_merger)

    args = parser.parse_args()

    if args.end is None and args.duration is None:
        parser.error("Use either --end or --duration, not both.")

    profiler = cProfile.Profile()
    profiler.enable()
    args.func(args)

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)
    return


if __name__ == "__main__":
    main()
