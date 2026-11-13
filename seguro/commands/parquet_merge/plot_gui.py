# SPDX-FileCopyrightText: 2026 Henrik Hörter
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, date
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from nicegui import ui

DATA_DIR = Path("merged")
RECURRENCES = ["daily", "weekly", "monthly"]


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------

def discover_tree(base: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Walk  merged/<gateway>/<device>/<measurement_point>/  and return a nested dict:
        { gateway: { device: { measurement_point: Path(merged/gw/dev/mp) } } }

    Adding a new device is as simple as placing its merged files in the right folder —
    the GUI picks it up automatically on the next page load / refresh.
    """
    tree: dict[str, dict[str, dict[str, Path]]] = {}

    for gateway_dir in sorted(base.iterdir()):
        if not gateway_dir.is_dir():
            continue
        for device_dir in sorted(gateway_dir.iterdir()):
            if not device_dir.is_dir():
                continue
            for point_dir in sorted(device_dir.iterdir()):
                if not point_dir.is_dir():
                    continue
                # Only add if at least one recurrence folder exists inside
                if any((point_dir / r).is_dir() for r in RECURRENCES):
                    (tree
                     .setdefault(gateway_dir.name, {})
                     .setdefault(device_dir.name, {})
                     )[point_dir.name] = point_dir

    return tree


def available_recurrences(point_path: Path) -> list[str]:
    """Return the recurrence folders that actually exist for a given measurement point."""
    return [r for r in RECURRENCES if (point_path / r).is_dir()]


def get_available_signals(point_path: Path, recurrence: str) -> list[str]:
    """Read the schema from the first parquet file found and return signal column names."""
    target_dir = point_path / recurrence
    files = sorted(target_dir.glob("*.parquet"))
    if not files:
        return []
    schema = pq.read_schema(files[0])
    return [name for name in schema.names
            if name.startswith("signal")]# and name.endswith(".real")]


def find_parquet_files(point_path: Path, recurrence: str,
                       start: date, end: date) -> list[Path]:
    """Return parquet files inside point_path/recurrence/ that fall within [start, end]."""
    target_dir = point_path / recurrence
    if not target_dir.exists():
        return []

    matched: list[Path] = []
    for p in sorted(target_dir.glob("*.parquet")):
        if recurrence == "daily":
            try:
                f_date = datetime.fromisoformat(p.stem).date()
                if start <= f_date <= end:
                    matched.append(p)
            except ValueError:
                pass
        else:
            # For weekly / monthly include everything and let the user zoom
            matched.append(p)

    return matched


def load_and_prepare_data(files: list[Path],
                          selected_signals: list[str],
                          max_points: int = 5000) -> pd.DataFrame:
    """Load only selected columns and downsample if necessary."""
    if not files or not selected_signals:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f, columns=selected_signals) for f in files]
    combined = pd.concat(dfs).sort_index()

    if len(combined) > max_points:
        rule = "1min" if len(combined) < 50_000 else "10min"
        combined = combined.resample(rule).mean().dropna(how="all")

    return combined


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

@ui.page("/")
def main_page():
    # Discover the full device tree once per page load
    tree = discover_tree(DATA_DIR)

    # ---- Helper to derive the currently selected point_path ----------------
    def current_point_path() -> Path | None:
        gw = gateway_select.value
        dev = device_select.value
        mp = point_select.value
        if gw and dev and mp:
            try:
                return tree[gw][dev][mp]
            except KeyError:
                pass
        return None

    # ---- Cascading update callbacks ----------------------------------------

    def on_gateway_change():
        gw = gateway_select.value
        devices = sorted(tree.get(gw, {}).keys())
        device_select.options = devices
        device_select.value = devices[0] if devices else None
        on_device_change()

    def on_device_change():
        gw = gateway_select.value
        dev = device_select.value
        points = sorted(tree.get(gw, {}).get(dev, {}).keys())
        point_select.options = points
        point_select.value = points[0] if points else None
        on_point_change()

    def on_point_change():
        pp = current_point_path()
        if pp is None:
            recurrence_select.options = RECURRENCES
            recurrence_select.value = RECURRENCES[0]
            signal_select.options = []
            signal_select.value = []
            return

        recs = available_recurrences(pp)
        recurrence_select.options = recs
        if recurrence_select.value not in recs:
            recurrence_select.value = recs[0] if recs else None

        on_recurrence_change()

    def on_recurrence_change():
        pp = current_point_path()
        if pp is None:
            return
        sigs = get_available_signals(pp, recurrence_select.value or "daily")
        signal_select.options = sigs
        signal_select.value = sigs[:2] if len(sigs) >= 2 else sigs

    # ---- Build initial option lists ----------------------------------------
    gateways = sorted(tree.keys())
    init_gw = gateways[0] if gateways else None
    init_devices = sorted(tree.get(init_gw, {}).keys()) if init_gw else []
    init_dev = init_devices[0] if init_devices else None
    init_points = sorted(tree.get(init_gw, {}).get(init_dev, {}).keys()) if init_dev else []
    init_pt = init_points[0] if init_points else None
    init_path = tree.get(init_gw, {}).get(init_dev, {}).get(init_pt) if init_pt else None
    init_recs = available_recurrences(init_path) if init_path else RECURRENCES
    init_rec = init_recs[0] if init_recs else "daily"
    init_sigs = get_available_signals(init_path, init_rec) if init_path else []
    init_sel_sigs = init_sigs[:2] if len(init_sigs) >= 2 else init_sigs

    # ---- Page layout -------------------------------------------------------
    ui.label("SEGuRo Parquet Measurement Visualizer").classes("text-2xl font-bold mb-4")

    with ui.card().classes("w-full p-4 mb-4"):

        # Row 1 — Device hierarchy selectors
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            ui.label("Device:").classes("text-sm font-semibold self-center")

            gateway_select = ui.select(
                options=gateways,
                value=init_gw,
                label="Gateway",
                on_change=on_gateway_change,
            ).classes("w-40")

            device_select = ui.select(
                options=init_devices,
                value=init_dev,
                label="Device",
                on_change=on_device_change,
            ).classes("w-32")

            point_select = ui.select(
                options=init_points,
                value=init_pt,
                label="Measurement Point",
                on_change=on_point_change,
            ).classes("w-44")

            recurrence_select = ui.select(
                options=init_recs,
                value=init_rec,
                label="Recurrence",
                on_change=on_recurrence_change,
            ).classes("w-32")

        ui.separator()

        # Row 2 — Time range + signal selector + plot button
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            start_input = ui.input(
                "Start Date (YYYY-MM-DD)", value="2025-08-01"
            ).classes("w-44")
            end_input = ui.input(
                "End Date (YYYY-MM-DD)", value="2025-08-31"
            ).classes("w-44")

            signal_select = ui.select(
                options=init_sigs,
                value=init_sel_sigs,
                multiple=True,
                label="Select Signals",
            ).classes("w-80")

            load_btn = ui.button("Plot", icon="show_chart").classes(
                "bg-blue-600 text-white"
            )

    # ---- ECharts container -------------------------------------------------
    chart = ui.echart({
        "title": {"text": "Select device and date range, then click Plot", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": "8%", "type": "scroll"},
        "grid": {"left": "3%", "right": "4%", "bottom": "15%", "containLabel": True},
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            }
        },
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "scale": True},
        "dataZoom": [
            {"type": "inside"},
            {"type": "slider", "bottom": "2%"},
        ],
        "series": [],
    }).classes("w-full h-[600px]")

    status_label = ui.label("").classes("text-gray-500 mt-2")

    # ---- Plot callback -----------------------------------------------------
    def update_plot():
        try:
            s_date = datetime.strptime(start_input.value.strip(), "%Y-%m-%d").date()
            e_date = datetime.strptime(end_input.value.strip(), "%Y-%m-%d").date()
        except ValueError:
            ui.notify("Invalid date format. Use YYYY-MM-DD", color="negative")
            return

        pp = current_point_path()
        if pp is None:
            ui.notify("No device / measurement point selected.", color="warning")
            return

        files = find_parquet_files(pp, recurrence_select.value, s_date, e_date)
        if not files:
            ui.notify(
                f"No parquet files found in {pp / recurrence_select.value} for the selected date range.",
                color="warning",
            )
            return

        df = load_and_prepare_data(files, signal_select.value)
        if df.empty:
            ui.notify("No data loaded — check signal selection.", color="warning")
            return

        series = [
            {
                "name": col,
                "type": "line",
                "showSymbol": False,
                "data": [
                    [ts.isoformat(), None if pd.isna(val) else float(val)]
                    for ts, val in zip(df.index, df[col])
                ],
            }
            for col in signal_select.value
        ]

        device_label = f"{gateway_select.value} / {device_select.value} / {point_select.value}"
        chart.options["title"]["text"] = (
            f"{device_label}  |  {s_date} → {e_date}  ({len(files)} files)"
        )
        chart.options["series"] = series
        chart.update()
        status_label.set_text(
            f"Loaded {len(df):,} points from {len(files)} file(s) · "
            f"{device_label} · {recurrence_select.value}"
        )

    load_btn.on("click", update_plot)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="SEGuRo Parquet Plotter", port=8085, reload=False)
