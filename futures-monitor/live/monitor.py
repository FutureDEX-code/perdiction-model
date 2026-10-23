#!/usr/bin/env python3
"""国内豆类 2701 + 外盘美豆复合监控。

用法:
  python3 monitor.py              终端实时刷新
  python3 monitor.py --web        打开本地网页看板
  python3 monitor.py --once       拉一次行情后退出
  python3 monitor.py --web --port 8765 --notify
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.train import (
    direction,
    domestic_tape,
    load_models,
    news_before,
    news_ema_as_of,
    _cached_rows,
    news_scores,
    series_path,
    views_from_features,
)

TZ = ZoneInfo("Asia/Shanghai")
SINA = "https://hq.sinajs.cn/list="
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# 涨跌提醒阈值（相对昨收）
ALERT_BEAN_CENTS = 8.0
ALERT_MEAL_USD = 3.0
ALERT_CN_PCT = 1.5


@dataclass
class Quote:
    code: str
    name: str
    last: Optional[float] = None
    prev: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    oi: Optional[float] = None
    time: str = ""
    date: str = ""
    ok: bool = False
    error: str = ""

    @property
    def chg(self) -> Optional[float]:
        if self.last is None or self.prev in (None, 0):
            return None
        return self.last - self.prev

    @property
    def pct(self) -> Optional[float]:
        if self.last is None or self.prev in (None, 0):
            return None
        return (self.last - self.prev) / self.prev * 100.0


@dataclass
class Bias:
    code: str
    name: str
    lean: str
    score: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class Snapshot:
    updated: str
    session: str
    domestic: list[Quote]
    overseas: list[Quote]
    biases: list[Bias]
    headline: str
    alerts: list[str]


def now_cn() -> datetime:
    return datetime.now(TZ)


def session_name(dt: Optional[datetime] = None) -> str:
    t = dt or now_cn()
    hm = t.hour * 100 + t.minute
    wd = t.weekday()
    if wd >= 5:
        return "周末休市"
    if 900 <= hm <= 1130 or 1330 <= hm <= 1500:
        return "日盘"
    if 2100 <= hm <= 2300:
        return "夜盘"
    if 1500 < hm < 2100:
        return "日盘收盘 / 夜盘未开"
    return "盘前 / 外盘时段"


def _get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.read()


def _f(parts: list[str], i: int) -> Optional[float]:
    if i >= len(parts):
        return None
    s = parts[i].strip()
    if not s or s in ("--", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_sina(codes: list[str]) -> dict[str, str]:
    raw = _get(SINA + ",".join(codes)).decode("gbk", errors="replace")
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if "hq_str_" not in line or '="' not in line:
            continue
        key = line.split("hq_str_", 1)[1].split("=", 1)[0]
        val = line.split('="', 1)[1].rsplit('";', 1)[0]
        out[key] = val
    return out


def parse_domestic(code: str, payload: str) -> Quote:
    q = Quote(code=code, name=code)
    if not payload:
        q.error = "无数据"
        return q
    p = payload.split(",")
    q.name = p[0] or code
    q.time = p[1] if len(p) > 1 else ""
    q.open = _f(p, 2)
    q.high = _f(p, 3)
    q.low = _f(p, 4)
    q.last = _f(p, 5) or _f(p, 8) or _f(p, 6)
    # 9=昨结 10=昨收；涨跌幅按昨收，和东财一致
    q.prev = _f(p, 10) or _f(p, 9)
    q.volume = _f(p, 13)
    q.oi = _f(p, 14)
    q.date = p[17] if len(p) > 17 else ""
    q.ok = q.last is not None
    return q


def parse_hf(code: str, payload: str) -> Quote:
    q = Quote(code=code, name=code)
    if not payload:
        q.error = "无数据"
        return q
    p = payload.split(",")
    q.last = _f(p, 0)
    q.high = _f(p, 4)
    q.low = _f(p, 5)
    q.time = p[6] if len(p) > 6 else ""
    q.prev = _f(p, 7)
    q.open = _f(p, 8)
    q.date = p[12] if len(p) > 12 else ""
    q.name = p[13] if len(p) > 13 else code
    q.ok = q.last is not None
    return q


def fetch_snapshot() -> Snapshot:
    cn_map = [
        ("nf_A2701", parse_domestic),
        ("nf_B2701", parse_domestic),
        ("nf_M2701", parse_domestic),
        ("nf_RM2701", parse_domestic),
        ("nf_Y2701", parse_domestic),
    ]
    hf_map = [("hf_S", parse_hf), ("hf_SM", parse_hf), ("hf_BO", parse_hf)]
    codes = [c for c, _ in cn_map + hf_map]
    try:
        raw = fetch_sina(codes)
        err = ""
    except Exception as exc:
        raw, err = {}, str(exc)

    domestic: list[Quote] = []
    for code, parser in cn_map:
        q = parser(code, raw.get(code, ""))
        if err and not q.ok:
            q.error = err
        domestic.append(q)

    overseas: list[Quote] = []
    for code, parser in hf_map:
        q = parser(code, raw.get(code, ""))
        if err and not q.ok:
            q.error = err
        overseas.append(q)

    biases, headline, alerts = build_bias(domestic, overseas)
    night, night_head = model_night(domestic, overseas)
    if night is not None:
        biases, headline = night, night_head
    return Snapshot(
        updated=now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        session=session_name(),
        domestic=domestic,
        overseas=overseas,
        biases=biases,
        headline=headline,
        alerts=alerts,
    )


def q_by_name(quotes: list[Quote], keyword: str) -> Optional[Quote]:
    for q in quotes:
        if keyword in (q.name or "") or keyword in q.code:
            return q
    return None


def build_bias(domestic: list[Quote], overseas: list[Quote]) -> tuple[list[Bias], str, list[str]]:
    a = q_by_name(domestic, "豆一") or (domestic[0] if domestic else Quote("", "豆一"))
    m = q_by_name(domestic, "豆粕") or (domestic[1] if len(domestic) > 1 else Quote("", "豆粕"))
    rm = q_by_name(domestic, "菜粕") or (domestic[2] if len(domestic) > 2 else Quote("", "菜粕"))
    beans = q_by_name(overseas, "大豆") or (overseas[0] if overseas else Quote("", "美豆"))
    meal = q_by_name(overseas, "豆粉") or q_by_name(overseas, "豆粕") or (
        overseas[1] if len(overseas) > 1 else Quote("", "美豆粕")
    )
    oil = q_by_name(overseas, "豆油") or (overseas[2] if len(overseas) > 2 else Quote("", "美豆油"))

    bean_cents = beans.chg if beans.chg is not None else 0.0
    meal_usd = meal.chg if meal.chg is not None else 0.0
    oil_chg = oil.chg if oil.chg is not None else 0.0
    m_pct = m.pct or 0.0
    rm_pct = rm.pct or 0.0

    score_m = 0
    reasons_m: list[str] = []

    if bean_cents >= 8:
        score_m += 2
        reasons_m.append(f"美豆 +{bean_cents:.1f} 美分，夜盘有新高度")
    elif bean_cents >= 3:
        score_m += 1
        reasons_m.append(f"美豆小涨 {bean_cents:.1f} 美分")
    elif bean_cents <= -8:
        score_m -= 2
        reasons_m.append(f"美豆 {bean_cents:.1f} 美分，外盘转弱")
    elif bean_cents <= -3:
        score_m -= 1
        reasons_m.append(f"美豆小跌 {bean_cents:.1f} 美分")
    else:
        reasons_m.append(f"美豆基本持平 ({bean_cents:+.1f} 美分)，没有新驱动")

    if meal_usd >= 3:
        score_m += 1
        reasons_m.append(f"美豆粕 +{meal_usd:.1f} 美元，粕略强")
    elif meal_usd <= -3:
        score_m -= 1
        reasons_m.append(f"美豆粕 {meal_usd:.1f} 美元")

    if oil_chg <= -0.8:
        reasons_m.append("美豆油偏弱，油粕跷跷板，粕难单独狂拉")

    sess = session_name()
    if sess in ("日盘收盘 / 夜盘未开", "夜盘") and m_pct >= 1.5 and bean_cents < 4:
        score_m -= 1
        reasons_m.append(f"豆粕日盘已涨 {m_pct:.2f}%，外盘未给新高，夜盘容易回吐")

    score_rm = score_m
    reasons_rm = list(reasons_m)
    if rm_pct >= 2.0 and bean_cents < 5:
        score_rm -= 1
        reasons_rm.append(f"菜粕日盘涨 {rm_pct:.2f}% 更大，盘子薄，夜盘回吐概率更高")
    if rm_pct >= 1.0:
        reasons_rm.append("菜粕跟外盘弹性大于豆粕")

    score_a = 1 if score_m >= 2 else (-1 if score_m <= -2 else 0)
    reasons_a = ["豆一跟东北现货/新作，夜盘对外盘弹性最小"]
    if (a.pct or 0) >= 1:
        reasons_a.append(f"日盘已涨 {a.pct:.2f}%，夜盘更可能走平")

    def lean(score: int) -> str:
        if score >= 2:
            return "偏涨"
        if score == 1:
            return "震荡偏强"
        if score <= -2:
            return "偏跌"
        if score == -1:
            return "震荡偏弱"
        return "震荡"

    biases = [
        Bias("A2701", "豆一2701", lean(score_a), score_a, reasons_a),
        Bias("M2701", "豆粕2701", lean(score_m), score_m, reasons_m),
        Bias("RM2701", "菜粕2701", lean(score_rm), score_rm, reasons_rm),
    ]

    ranking = sorted(biases, key=lambda b: b.score)
    weak = ranking[0]
    strong = ranking[-1]
    if all(b.lean.startswith("震荡") for b in biases):
        headline = "今晚基准：高位震荡，略偏回吐，不像日盘那样再拉一波。"
    elif weak.score <= -1 and strong.score <= 0:
        headline = f"今晚基准：偏弱。{weak.name}最容易先跌，{strong.name}相对抗跌。"
    elif strong.score >= 1 and weak.score >= 0:
        headline = f"今晚基准：偏强。外盘仍在给高度，注意空单。"
    else:
        headline = f"今晚基准：分化。{strong.name}{strong.lean}，{weak.name}{weak.lean}。"

    alerts: list[str] = []
    if abs(bean_cents) >= ALERT_BEAN_CENTS:
        alerts.append(f"美豆波动 {bean_cents:+.1f} 美分")
    if abs(meal_usd) >= ALERT_MEAL_USD:
        alerts.append(f"美豆粕波动 {meal_usd:+.1f} 美元")
    if abs(m_pct) >= ALERT_CN_PCT:
        alerts.append(f"豆粕2701 日涨跌 {m_pct:+.2f}%")
    if abs(rm_pct) >= ALERT_CN_PCT:
        alerts.append(f"菜粕2701 日涨跌 {rm_pct:+.2f}%")
    return biases, headline, alerts


def _domestic_tape() -> list[float]:
    series: dict[str, dict[str, float]] = {}
    for key in ("A", "M", "RM", "B"):
        path = series_path(key)
        if not path.exists():
            continue
        series[key] = {day: close for day, close in _cached_rows(path) if close}
    return domestic_tape(series)


def model_night(domestic: list[Quote], overseas: list[Quote]) -> tuple[list[Bias] | None, str]:
    """Night call. Overseas input is this session versus its own previous settle."""

    names = (
        ("豆一昨涨跌", domestic, "豆一"),
        ("豆二昨涨跌", domestic, "豆二"),
        ("豆油昨涨跌", domestic, "豆油"),
        ("豆粕昨涨跌", domestic, "豆粕"),
        ("菜粕昨涨跌", domestic, "菜粕"),
        ("美豆隔夜", overseas, "大豆"),
        ("美豆粕隔夜", overseas, "豆粉"),
        ("美豆油隔夜", overseas, "豆油"),
    )
    named: dict[str, float] = {}
    for label, pool, keyword in names:
        quote = q_by_name(pool, keyword)
        if label in ("豆二昨涨跌", "豆油昨涨跌") and (quote is None or quote.pct is None):
            continue
        if quote is None or quote.pct is None:
            return None, "今晚：报价不齐，没有方向"
        named[label] = quote.pct
    today = now_cn().date()
    tape = _domestic_tape()
    for label, value in zip(("豆一这段", "豆粕这段", "菜粕这段"), tape):
        named[label] = value
    b_path = series_path("B")
    if b_path.exists():
        b_series = {"B": {day: close for day, close in _cached_rows(b_path) if close}}
        named["豆二这段"] = domestic_tape(b_series, keys=("B",))[0]
    y_path = series_path("Y")
    if y_path.exists():
        y_series = {"Y": {day: close for day, close in _cached_rows(y_path) if close}}
        named["豆油这段"] = domestic_tape(y_series, keys=("Y",))[0]
    closes = {}
    for key in ("A", "M", "RM", "S", "SM", "BO"):
        path = series_path(key)
        if path.exists():
            closes[key] = {day: close for day, close in _cached_rows(path) if close}
    if {"A", "M", "RM"} <= set(closes):
        named["新闻三日"] = news_ema_as_of(closes, news_scores(), today.isoformat())
    named["菜粕价差波段"] = tape[2] - tape[1]
    try:
        models = load_models()
    except (OSError, json.JSONDecodeError, KeyError):
        return None, "今晚：模型文件读不到，没有方向"
    views = views_from_features(named, models)
    biases = [
        Bias(
            view["code"],
            view["name"],
            view["lean"],
            0,
            [
                f"上涨概率 {view['prob_up'] * 100:.1f}%",
                "外盘用这一节相对前结算",
                "国内这段 "
                + "，".join(
                    f"{name} {value:+.1f}%"
                    for name, value in zip(("豆一", "豆粕", "菜粕"), tape)
                )
                + "，回头不到3%仍算同一段",
            ],
        )
        for view in views
    ]
    spoken = [f"{view['name']}{view['lean']}" for view in views]
    return biases, "今晚：" + "，".join(spoken)


def snapshot_dict(s: Snapshot) -> dict:
    def q(x: Quote) -> dict:
        d = asdict(x)
        d["chg"] = x.chg
        d["pct"] = x.pct
        return d

    return {
        "updated": s.updated,
        "session": s.session,
        "headline": s.headline,
        "alerts": s.alerts,
        "domestic": [q(x) for x in s.domestic],
        "overseas": [q(x) for x in s.overseas],
        "biases": [asdict(b) for b in s.biases],
    }


def color(pct: Optional[float], text: str) -> str:
    if not sys.stdout.isatty() or pct is None:
        return text
    if pct > 0:
        return f"\033[31m{text}\033[0m"
    if pct < 0:
        return f"\033[32m{text}\033[0m"
    return text


def fmt_px(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def fmt_chg(q: Quote, digits: int = 1) -> str:
    if q.chg is None or q.pct is None:
        return "--"
    return f"{q.chg:+.{digits}f} ({q.pct:+.2f}%)"


def render_terminal(s: Snapshot) -> str:
    lines = [
        f"期货监控  {s.updated}  [{s.session}]",
        s.headline,
        "",
        f"{'合约':<10}{'最新':>10}{'涨跌':>18}{'开盘':>10}{'最高':>10}{'最低':>10}",
        "-" * 72,
    ]
    for q in s.domestic + overseas_gap(s.overseas):
        row = (
            f"{q.name:<10}{fmt_px(q.last, 2):>10}"
            f"{fmt_chg(q, 2):>18}{fmt_px(q.open, 2):>10}"
            f"{fmt_px(q.high, 2):>10}{fmt_px(q.low, 2):>10}"
        )
        lines.append(color(q.pct, row))
    lines += ["", "夜盘判断"]
    for b in s.biases:
        lines.append(f"  {b.name}: {b.lean}  (score {b.score:+d})")
        for r in b.reasons[:3]:
            lines.append(f"    - {r}")
    if s.alerts:
        lines.append("")
        lines.append("提醒: " + "；".join(s.alerts))
    outlook = _outlook_lines()
    if outlook:
        lines.append("")
        lines.extend(outlook)
    return "\n".join(lines)


def _outlook_lines() -> list[str]:
    path = Path(__file__).resolve().parents[1] / "data" / "outlook.json"
    if not path.exists():
        return []
    try:
        view = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    tape = view.get("tape_pct") or []
    lines = [f"下一交易日（{view.get('as_of', '')} 收盘后训练）"]
    if len(tape) == 3:
        lines.append(
            "国内这段 "
            + "，".join(f"{name} {value:+.1f}%" for name, value in zip(("豆一", "豆粕", "菜粕"), tape))
            + "，回头不到3%仍算同一段"
        )
    for item in view.get("views") or []:
        prob = item.get("prob_up")
        lean = direction(prob) if isinstance(prob, (int, float)) else item.get("lean", "")
        lines.append(f"  {item['name']}: {lean}  上涨概率 {prob * 100:.1f}%")
    return lines


def overseas_gap(items: list[Quote]) -> list[Quote]:
    return items


def notify_mac(title: str, body: str) -> None:
    script = f'display notification {json.dumps(body, ensure_ascii=False)} with title {json.dumps(title, ensure_ascii=False)}'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=3)
    except Exception:
        pass


class AlertState:
    def __init__(self) -> None:
        self.last_key = ""

    def maybe_notify(self, s: Snapshot, enabled: bool) -> None:
        if not enabled or not s.alerts:
            return
        key = "|".join(s.alerts + [b.lean for b in s.biases])
        if key == self.last_key:
            return
        self.last_key = key
        notify_mac("期货监控", s.headline + " " + "；".join(s.alerts))


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>豆类期货监控</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#111; --card:#1b1b1b; --fg:#eee; --muted:#888; --line:#2a2a2a; --up:#d94c4c; --down:#3d9a64; --acc:#c8a35a; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:20px 24px 8px; }
  h1 { font-size:18px; font-weight:600; margin:0 0 6px; }
  .meta { color:var(--muted); font-size:13px; }
  .headline { margin:12px 24px; padding:14px 16px; background:var(--card); border-left:3px solid var(--acc); font-size:15px; line-height:1.5; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; padding:0 24px 24px; }
  @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); padding:14px 16px; }
  .card h2 { margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:500; }
  .px { font-size:28px; font-variant-numeric:tabular-nums; }
  .chg { margin-top:4px; font-size:14px; font-variant-numeric:tabular-nums; }
  .up { color:var(--up); } .down { color:var(--down); } .flat { color:var(--muted); }
  .row { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-top:8px; }
  .bias { padding:0 24px 28px; }
  .bias-item { background:var(--card); padding:12px 16px; margin-bottom:10px; }
  .bias-item b { margin-right:8px; }
  ul { margin:8px 0 0 18px; color:var(--muted); font-size:13px; }
  .alerts { margin:0 24px 20px; color:var(--acc); font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>豆一 / 豆二 / 豆粕 / 菜粕 / 豆油 2701 监控</h1>
  <div class="meta" id="meta">加载中…</div>
</header>
<div class="headline" id="headline"></div>
<div class="alerts" id="alerts"></div>
<div class="grid" id="quotes"></div>
<section class="bias" id="bias"></section>
<script>
function cls(pct) {
  if (pct == null) return "flat";
  if (pct > 0) return "up";
  if (pct < 0) return "down";
  return "flat";
}
function n(v, d=1) { return v == null ? "--" : Number(v).toFixed(d); }
function chg(q, d=1) {
  if (q.chg == null || q.pct == null) return "--";
  const sign = q.chg > 0 ? "+" : "";
  return sign + Number(q.chg).toFixed(d) + "  " + sign + Number(q.pct).toFixed(2) + "%";
}
function card(q, digits) {
  return `<div class="card">
    <h2>${q.name} <span style="float:right">${q.time || ""}</span></h2>
    <div class="px ${cls(q.pct)}">${n(q.last, digits)}</div>
    <div class="chg ${cls(q.pct)}">${chg(q, digits)}</div>
    <div class="row"><span>开 ${n(q.open, digits)}</span><span>高 ${n(q.high, digits)}</span><span>低 ${n(q.low, digits)}</span></div>
  </div>`;
}
async function tick() {
  const r = await fetch("/api/snapshot");
  const s = await r.json();
  document.getElementById("meta").textContent = s.updated + "  ·  " + s.session + "  ·  每 3 秒刷新";
  document.getElementById("headline").textContent = s.headline;
  document.getElementById("alerts").textContent = (s.alerts || []).join("  ·  ");
  const qh = [...s.domestic.map(q => card(q, 1)), ...s.overseas.map(q => card(q, 2))].join("");
  document.getElementById("quotes").innerHTML = qh;
  document.getElementById("bias").innerHTML = s.biases.map(b =>
    `<div class="bias-item"><b>${b.name}</b><span class="${b.lean.includes("涨")||b.lean.includes("强")?"up":(b.lean.includes("跌")||b.lean.includes("弱")?"down":"flat")}">${b.lean}</span>
     <ul>${(b.reasons||[]).map(x => "<li>"+x+"</li>").join("")}</ul></div>`
  ).join("");
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    store = {"snap": None}

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/api/snapshot":
            snap = self.store["snap"] or fetch_snapshot()
            body = json.dumps(snapshot_dict(snap), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def poll_loop(interval: float, notify: bool, sink: Optional[dict]) -> None:
    alerts = AlertState()
    while True:
        try:
            snap = fetch_snapshot()
        except Exception as exc:
            snap = Snapshot(
                updated=now_cn().strftime("%Y-%m-%d %H:%M:%S"),
                session=session_name(),
                domestic=[],
                overseas=[],
                biases=[],
                headline=f"拉行情失败: {exc}",
                alerts=[],
            )
        if sink is not None:
            sink["snap"] = snap
        else:
            os.system("clear" if os.name != "nt" else "cls")
            print(render_terminal(snap))
            print("\nCtrl+C 退出")
        alerts.maybe_notify(snap, notify)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="豆一 / 豆二 / 豆粕 / 菜粕 2701 + 美豆复合监控")
    parser.add_argument("--web", action="store_true", help="启动本地网页看板")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=3.0, help="刷新秒数")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--notify", action="store_true", help="macOS 通知")
    args = parser.parse_args()

    if args.once:
        print(render_terminal(fetch_snapshot()))
        return 0

    if args.web:
        Handler.store = {"snap": None}
        t = threading.Thread(
            target=poll_loop, args=(args.interval, args.notify, Handler.store), daemon=True
        )
        t.start()
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"看板已启动: {url}")
        if sys.platform == "darwin":
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
        return 0

    try:
        poll_loop(args.interval, args.notify, None)
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
