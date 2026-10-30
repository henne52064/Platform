# SPDX-FileCopyrightText: 2026 Henrik Hörter
# SPDX-License-Identifier: Apache-2.0
import cProfile
import pstats
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import re
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

from seguro.common import config
from seguro.common.store import Client

LOCAL_TMP = ".LOCAL/"
REMOTE_FOLDER = "data/measurements/"
MERGED_DIR = "merged/"

SAMPLE_K = 4


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


def get_data(
    candidates: list[str],
    client: Client,
    output_path: str,
    start_dt: datetime,
    end_dt: datetime,
    max_workers: int = 16,
):
    """
    Downloads parquet files concurrently into memory, validates metadata on the fly,
    and merges them into a single output file using PyArrow
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

        # use PyArrow for faster validation, load index to pandas without full dataframe overhead
        df_stub = table.select([]).to_pandas()
        if not df_stub.index.is_monotonic_increasing:
            logging.warning(f"Timestamps in '{candidate}' are not monotonically increasing.")

        # CHECK TIMEGAPS BETWEEN SAMPLES
        index = pd.to_datetime(df_stub.index, unit="s")
        ts_ns = index.to_numpy(dtype="datetime64[ns]").astype("int64")

        if len(ts_ns) < 2:
            logging.warning(f"Not enough sample points in '{candidate}' to evaluate timesteps.")
            return idx, table

        diffs_ns = np.diff(ts_ns)

        median_ns = float(np.median(diffs_ns))

        mad_ns = max(1.4826 * float(np.median(np.abs(diffs_ns - median_ns))), 0.01 * median_ns)
        upper_ns = median_ns + SAMPLE_K * mad_ns
        lower_ns = median_ns - SAMPLE_K * mad_ns

        expected = pd.Timedelta(int(median_ns), unit="ns")

        for pos in np.flatnonzero(diffs_ns > upper_ns):
            logging.warning(
                f"Sample gap detected in '{candidate}' between '{index[pos]}' and '{index[pos + 1]}': "
                f"Difference is {pd.Timedelta(int(diffs_ns[pos]), unit='ns')} "
                f"(Expected ~{expected}, Threshold: {pd.Timedelta(int(upper_ns), unit='ns')})"
            )
        for pos in np.flatnonzero((diffs_ns <= 0) | (diffs_ns < lower_ns)):
            logging.warning(
                f"Too short sample interval in '{candidate}' between '{index[pos]}' and '{index[pos + 1]}': "
                f"Difference is {pd.Timedelta(int(diffs_ns[pos]), unit='ns')} "
                f"(Expected ~{expected}, Threshold: {pd.Timedelta(int(lower_ns), unit='ns')})"
            )

        return idx, table

    # Parallel download & sort file to correct position
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_and_parse, i, candidate) for i, candidate in enumerate(sorted_candidates)]
        for future in as_completed(futures):
            idx, table = future.result()
            tables[idx] = table  # Keep original key order
            clean_name = sorted_candidates[idx].removesuffix(".parquet").split("/")[-1]
            filename_dates[idx] = datetime.fromisoformat(clean_name)

    # Check filename timestamps
    dates_series = pd.Series(filename_dates)
    time_diffs = dates_series.diff().dropna()

    median_diff = time_diffs.median()
    mad = 1.4826 * (time_diffs - median_diff).abs().median()
    threshold = median_diff + max(SAMPLE_K * mad, 0.05 * median_diff)

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
    print(f"number of files {len(filename_dates)}")
    # zip data to combine filename with table
    data = zip(filename_dates, tables)

    return data


def merger(data: zip, output_path: str, recurrence: str):

    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

    def extract_calendar_date(pair):
        return pair[0].date()

    def extract_isocalendar(pair):
        iso_year, iso_week, _ = pair[0].isocalendar()
        return (iso_year, iso_week)

    def extract_year_month(pair):
        date_obj = pair[0]
        return (date_obj.year, date_obj.month)

    if recurrence == "single":
        tables_to_merge = [table for _, table in data]
        merged_table = pa.concat_tables(tables_to_merge)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        pq.write_table(merged_table, output_path + "merged.parquet")

    # daily
    # output structure daily/year/month/{date}.parquet
    if recurrence == "daily":

        for current_date, group in groupby(data, key=extract_calendar_date):
            daily_tables = [table for _, table in group]
            merged_daily_table = pa.concat_tables(daily_tables)
            path = output_path + "daily/" + str(current_date.year) + "/" + str(current_date.month)
            Path(path).mkdir(parents=True, exist_ok=True)
            pq.write_table(merged_daily_table, path + "/" + str(current_date) + ".parquet")

    # weekly
    # output structure weekly/{year}_CW_{week}.parquet
    if recurrence == "weekly":
        for (year, week), group in groupby(data, key=extract_isocalendar):
            weekly_tables = [table for _, table in group]
            merged_weekly_table = pa.concat_tables(weekly_tables)
            path = output_path + "weekly/" + str(year)
            Path(path).mkdir(parents=True, exist_ok=True)
            pq.write_table(merged_weekly_table, path + "/CW_" + str(week) + ".parquet")

    # monthly
    # output structure monthly/{year}_{month}.parquet
    if recurrence == "monthly":
        for (year, month), group in groupby(data, key=extract_year_month):
            monthly_tables = [table for _, table in group]
            merged_monthly_table = pa.concat_tables(monthly_tables)
            path = output_path + "monthly/" + str(year)
            Path(path).mkdir(parents=True, exist_ok=True)
            pq.write_table(merged_monthly_table, path + "/month_" + str(month) + ".parquet")


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

    data = get_data(
        candidates=candidates,
        client=client,
        output_path=args.path,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    merger(data=data, output_path=args.path, recurrence=args.recurrence)

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
        "-p",
        "--path",
        default=MERGED_DIR,  # + "merged.parquet",
        type=str,
        help="Path or directory where to store merged file",
    )

    parser.add_argument(
        "-r",
        "--recurrence",
        default="single",
        type=str,
        help="Size of merged files: daily, weekly or monthly (day, week, month)",
    )

    parser.set_defaults(func=parquet_merger)

    args = parser.parse_args()

    if not args.path.endswith("/"):
        args.path = args.path + "/"

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
