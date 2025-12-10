from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from epiweeks import Week

from covid_abm.calibration.utils.neighborhood import Neighborhood

MOBILITY_COLUMNS = [
    "retail_and_recreation_change_week",
    "grocery_and_pharmacy_change_week",
    "parks_change_week",
    "transit_stations_change_week",
    "workplaces_change_week",
    "residential_change_week",
]

OUTPUT_COLUMNS = [
    "date",
    "epiweek",
    "nta_id",
    "county",
    "neighborhood",
    "cases_week",
    "cases_4_week_avg",
    "hospitalizations_month",
    "deaths_month",
    *MOBILITY_COLUMNS,
]


def epiweek_int(dt: pd.Timestamp) -> int:
    return int(str(Week.fromdate(dt)))


def compute_deaths_28_days(df: pd.DataFrame, end_day: pd.Timestamp) -> float:
    window_start = end_day - pd.Timedelta(days=27)
    mask = (df["date"] >= window_start) & (df["date"] <= end_day)
    return float(df.loc[mask, "deaths"].sum())


def aggregate_county(
    neighborhood: Neighborhood,
    daily_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    fips = neighborhood.nta_id
    daily_path = daily_dir / f"{fips}_daily_data.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing daily data for {fips}: {daily_path}")

    daily = pd.read_csv(daily_path, parse_dates=["date"]).sort_values("date")
    mask = (daily["date"] >= start_date) & (daily["date"] <= end_date)
    daily = daily.loc[mask].copy()
    if daily.empty:
        raise ValueError(f"No rows for {fips} between {start_date.date()} and {end_date.date()}.")

    daily["cases"] = daily["cases"].astype(float)
    daily["deaths"] = daily["deaths"].astype(float)
    daily["epiweek"] = daily["date"].apply(epiweek_int)

    weekly = (
        daily.groupby("epiweek", as_index=False)
        .agg({"cases": "sum"})
        .rename(columns={"cases": "cases_week"})
    )
    weekly = weekly.sort_values("epiweek").reset_index(drop=True)
    weekly["cases_4_week_avg"] = (
        weekly["cases_week"].rolling(window=4, min_periods=4).mean().fillna(0.0)
    )

    week_end_dates = weekly["epiweek"].apply(
        lambda ew: Week.fromstring(str(int(ew))).enddate()
    )
    week_end_dates = pd.to_datetime(week_end_dates)
    weekly["date"] = week_end_dates.dt.strftime("%Y/%m/%d")
    weekly["deaths_month"] = [
        compute_deaths_28_days(daily, pd.Timestamp(end_day)) for end_day in week_end_dates
    ]
    weekly["hospitalizations_month"] = 0.0
    for column in MOBILITY_COLUMNS:
        weekly[column] = 0.0

    weekly.insert(2, "nta_id", fips)
    weekly.insert(3, "county", neighborhood.name)
    weekly.insert(4, "neighborhood", neighborhood.name)

    weekly = weekly[OUTPUT_COLUMNS]
    weekly["epiweek"] = weekly["epiweek"].astype(int)
    return weekly


def build_dataset(
    neighborhoods: Iterable[Neighborhood],
    daily_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    frames = [
        aggregate_county(neighborhood, daily_dir, start_ts, end_ts)
        for neighborhood in neighborhoods
    ]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Aggregate daily county data into weekly slices.")
    parser.add_argument(
        "--daily_dir",
        default="data/daily_county_data",
        help="Directory containing {fips}_daily_data.csv files.",
    )
    parser.add_argument(
        "--output",
        default="covid_abm/data/county_data.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--start_date",
        default="2020-06-01",
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        default="2021-07-31",
        help="Inclusive end date (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    daily_dir = Path(args.daily_dir)
    output_path = Path(args.output)
    neighborhoods = sorted(list(Neighborhood), key=lambda n: n.nta_id)
    dataset = build_dataset(neighborhoods, daily_dir, args.start_date, args.end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Wrote {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()


