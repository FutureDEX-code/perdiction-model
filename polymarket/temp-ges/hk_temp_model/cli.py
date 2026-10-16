"""Print settlement-station forecasts for Hong Kong, Paris, and New York."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime

from hk_temp_model.climatology import load_climatology
from hk_temp_model.decide import format_no_board, no_board
from hk_temp_model.live import execute, exit_hopeless
from hk_temp_model.paper import settle_scored, status_line as paper_status
from hk_temp_model.market import fetch_yes_book
from hk_temp_model import intl
from hk_temp_model.learn import load_state, save_state, verify_settled
from hk_temp_model.model import predict
from hk_temp_model.observations import fetch_observation
from hk_temp_model.settlement import bracket_for, default_brackets


def _prediction_dict(result) -> dict:
    payload = asdict(result)
    payload["mode"] = result.mode.label
    payload["printing"] = bracket_for(result.hko_max_c).label
    return payload


def format_hk_report(result, state_line: str, forecast_carried: bool = False) -> str:
    lines = [
        f"结算站  {result.settlement_name} ({result.settlement_station})",
        f"观测    {result.observed_at}",
        f"已印最高  {result.hko_max_c:.1f}°C  →  若现在截停落 {bracket_for(result.hko_max_c).label}",
    ]
    if result.hka_max_c is None:
        lines.append("赤鱲角  无实况，不参与结算")
    else:
        lines.append(
            "赤鱲角  "
            f"{result.hka_max_c:.1f}°C，比总部 {result.airport_minus_hko_c:+.1f}°C。"
            f"9月历史平均偏高 {result.historical_airport_gap_c:.2f}°C，这条不进模型"
        )
    if result.forecast_max_c is None:
        forecast = "九日预报已不含今日"
    elif forecast_carried:
        forecast = f"{result.forecast_max_c:.0f}°C（九日预报已不含今日，沿用早间）"
    else:
        forecast = f"{result.forecast_max_c:.0f}°C"
    lines.extend(
        [
            f"气候同日均值  {result.climate_mean_c:.2f}°C    预报最高  {forecast}",
            f"加热 {result.heating:.2f}   风 {result.wind:.2f}   "
            f"晴空剩余升幅 {result.remaining_clear_rise_c:.2f}°C   路径 {result.path_max_c:.2f}°C",
            f"公允模型预测  {result.mu_c:.2f}°C    标准差 {result.sigma_c:.2f}°C    最高概率  {result.mode.label}",
            state_line,
            "",
            f"{'档位':<18}{'公允':>8}{'距下沿':>10}",
        ]
    )
    for row in result.fair:
        if row.probability < 0.005 and row.dist_c != 0.0:
            continue
        dist = "已过" if row.dist_c is None else ("已印" if row.dist_c == 0.0 else f"{row.dist_c:.1f}°C")
        lines.append(f"{row.label:<18}{row.probability * 100:7.1f}%{dist:>10}")
    return "\n".join(lines)


def _book_for(city_key: str, day) -> tuple[dict, dict]:
    try:
        return fetch_yes_book(city_key, day)
    except Exception as exc:
        print(f"盘口卖价未取到  {city_key}  {exc}", flush=True)
        return {}, {}


def _ask_line(asks: dict) -> str:
    shown = [f"{label} {price:.3f}" for label, price in asks.items() if price is not None and price >= 0.01]
    if not asks:
        return "盘口卖价  未取到"
    if not shown:
        return "盘口卖价  全部低于 0.01"
    return "盘口卖价  " + "   ".join(shown)


def _print_nos(rows: list[dict]) -> None:
    print(format_no_board(rows), flush=True)


def _trading_floor(running: float, current: float, hour: float) -> tuple[float, bool]:
    """The first hour can still be showing yesterday's maximum. Don't trade on it."""

    if hour < 1 and running - current >= 1:
        return current, False
    return running, True


def _emit_hk(args, climatology, state) -> None:
    obs = fetch_observation()
    held = state.hold_forecast(obs.observed_at.date(), obs.forecast_max_c)
    carried = obs.forecast_max_c is None and held is not None
    if carried:
        obs = replace(obs, forecast_max_c=held)
    result = predict(
        obs,
        climatology,
        default_brackets(args.low, args.high),
        state.adjustment,
    )
    asks, bids = _book_for("hk", obs.observed_at.date())
    climate = climatology.for_date(obs.observed_at.date())
    current = result.hko_max_c if obs.hko_temp_c is None else obs.hko_temp_c
    running, trade = _trading_floor(result.hko_max_c, current, obs.hour_hkt)
    nos = no_board(
        running=running,
        current=current,
        hour=obs.hour_hkt,
        peak=14.75,
        forecast=result.forecast_max_c,
        fair=result.fair,
        asks=asks,
        bids=bids,
        city="hk",
        mu=result.mu_c,
    )
    state.record(
        result,
        obs,
        forecast_carried=carried,
        climate_std_c=climate.std_c,
        asks=asks,
        bids=bids,
        nos=nos,
    )
    try:
        notes = verify_settled(state)
    except Exception as exc:
        notes = [f"结算核对失败  {exc}"]
    save_state(state)
    for note in notes:
        print(note, flush=True)
    if trade:
        for note in exit_hopeless("hk", obs.observed_at.date(), nos):
            print(note, flush=True)
        for note in execute("hk", obs.observed_at.date(), result.observed_at, nos):
            print(note, flush=True)
    for note in settle_scored("hk", state.scored):
        print(note, flush=True)
    if args.json:
        payload = _prediction_dict(result)
        payload["city"] = "hongkong"
        payload["bias_c"] = state.bias_c
        payload["sigma_mult"] = state.sigma_mult
        payload["verified_days"] = len(state.scored or {})
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    else:
        print("======== 香港 HKO ========", flush=True)
        print(format_hk_report(result, state.status_line(), carried), flush=True)
        print(_ask_line(asks), flush=True)
        _print_nos(nos)
        print(paper_status(), flush=True)
        print(flush=True)


def _emit_intl(args, city: intl.CitySpec, state: intl.IntlOnlineState) -> None:
    obs = intl.fetch_observation(city)
    result = intl.predict(
        obs,
        intl.whole_brackets(city.low, city.high, city.unit),
        state.adjustment,
    )
    asks, bids = _book_for(city.key, obs.observed_at.date())
    running, trade = _trading_floor(result.running_max, result.current, obs.hour_local)
    nos = no_board(
        running=running,
        current=result.current,
        hour=obs.hour_local,
        peak=city.temp_peak,
        forecast=result.forecast_max,
        fair=result.fair,
        asks=asks,
        bids=bids,
        overshoot=3.0 if city.unit == "F" else 1.5,
        city=city.key,
        mu=result.mu,
    )
    state.record(result, obs, asks=asks, bids=bids, nos=nos)
    try:
        notes = intl.verify_settled(city, state)
    except Exception as exc:
        notes = [f"结算核对失败  {city.key}  {exc}"]
    intl.save_state(city, state)
    for note in notes:
        print(note, flush=True)
    if trade:
        for note in exit_hopeless(city.key, obs.observed_at.date(), nos):
            print(note, flush=True)
        for note in execute(city.key, obs.observed_at.date(), result.observed_at, nos):
            print(note, flush=True)
    for note in settle_scored(city.key, state.scored):
        print(note, flush=True)
    title = "巴黎 LFPB" if city.key == "paris" else "纽约 KLGA"
    if args.json:
        payload = asdict(result)
        payload["mode"] = result.mode.label
        payload["bias"] = state.bias
        payload["sigma_mult"] = state.sigma_mult
        payload["verified_days"] = len(state.scored or {})
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"======== {title} ========", flush=True)
        print(intl.format_report(result, state.status_line()), flush=True)
        print(_ask_line(asks), flush=True)
        _print_nos(nos)
        print(paper_status(), flush=True)
        print(flush=True)


def _emit(args, climatology, hk_state, intl_states) -> None:
    cities = set(args.cities)
    jobs = []
    if "hk" in cities:
        jobs.append(("hk", lambda: _emit_hk(args, climatology, hk_state)))
    for city in intl.INTL_CITIES:
        if city.key in cities:
            jobs.append((city.key, lambda city=city: _emit_intl(args, city, intl_states[city.key])))
    for name, job in jobs:
        try:
            job()
        except Exception as exc:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp}  读取失败  {name}  {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict Polymarket daily highs for Hong Kong, Paris, and New York."
    )
    parser.add_argument("--json", action="store_true", help="print the raw forecast as JSON")
    parser.add_argument("--low", type=int, default=24, help="HK lowest interior-tail label, default 24")
    parser.add_argument("--high", type=int, default=34, help="HK highest tail label, default 34")
    parser.add_argument(
        "--cities",
        default="hk,paris,nyc",
        help="comma list: hk,paris,nyc (default all)",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        help="refresh every N minutes and keep running; 0 runs once",
    )
    args = parser.parse_args(argv)
    if args.watch < 0:
        parser.error("--watch must be >= 0")
    args.cities = [part.strip().lower() for part in args.cities.split(",") if part.strip()]
    unknown = set(args.cities) - {"hk", "paris", "nyc"}
    if unknown:
        parser.error(f"unknown cities: {', '.join(sorted(unknown))}")

    climatology = load_climatology()
    hk_state = load_state()
    intl_states = {city.key: intl.load_state(city) for city in intl.INTL_CITIES}
    while True:
        try:
            _emit(args, climatology, hk_state, intl_states)
        except Exception as exc:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp}  读取失败  {exc}", flush=True)
        if args.watch == 0:
            return 0
        time.sleep(args.watch * 60)
