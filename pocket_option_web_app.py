from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from typing import List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template_string


@dataclass
class Config:
    api_key: str = os.getenv("TWELVEDATA_API_KEY", "")
    assets: Tuple[str, ...] = (
        "EUR/USD",
        "GBP/USD",
        "USD/JPY",
        "BTC/USD",
        "ETH/USD",
    )
    chart_tf: str = "1min"
    higher_tf: str = "5min"
    outputsize: int = 220
    expiry_minutes: int = 2
    ema_fast: int = 8
    ema_slow: int = 21
    rsi_period: int = 14
    adx_period: int = 14
    min_rsi_call: float = 53.0
    max_rsi_call: float = 70.0
    min_rsi_put: float = 30.0
    max_rsi_put: float = 47.0
    min_adx: float = 16.0
    min_final_score: float = 62.0


@dataclass
class Candle:
    timestamp: str
    asset: str
    open: float
    high: float
    low: float
    close: float


@dataclass
class SignalResult:
    asset: str
    timestamp: str
    signal: str
    entry: str
    expiry_minutes: int
    confidence: int
    higher_tf_bias: str
    rsi_value: float
    adx_value: float
    final_score: float
    reason: str


cfg = Config()
app = Flask(__name__)


def utc_now_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def http_get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def fetch_candles(symbol: str, interval: str, outputsize: int, api_key: str) -> List[Candle]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "timezone": "UTC",
        "apikey": api_key,
    }
    url = f"https://api.twelvedata.com/time_series?{urlencode(params)}"
    data = http_get_json(url)
    if data.get("status") == "error":
        raise RuntimeError(data.get("message", "unknown API error"))
    values = data.get("values", [])
    if not values:
        raise RuntimeError("no data returned")

    candles: List[Candle] = []
    for row in reversed(values):
        candles.append(
            Candle(
                timestamp=row["datetime"],
                asset=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        )
    return candles


def ema(values: List[float], period: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return result
    multiplier = 2 / (period + 1)
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * multiplier + prev
        result[i] = prev
    return result


def rsi(values: List[float], period: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains[i] = max(delta, 0.0)
        losses[i] = max(-delta, 0.0)
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    result[period] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    for i in range(period + 1, len(values)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        result[i] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    return result


def true_range(curr: Candle, prev_close: float) -> float:
    return max(curr.high - curr.low, abs(curr.high - prev_close), abs(curr.low - prev_close))


def adx(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    if n <= period * 2:
        return out
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up_move = candles[i].high - candles[i - 1].high
        down_move = candles[i - 1].low - candles[i].low
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = true_range(candles[i], candles[i - 1].close)
    tr_s = [0.0] * n
    plus_s = [0.0] * n
    minus_s = [0.0] * n
    tr_s[period] = sum(tr[1:period + 1])
    plus_s[period] = sum(plus_dm[1:period + 1])
    minus_s[period] = sum(minus_dm[1:period + 1])
    plus_di = [0.0] * n
    minus_di = [0.0] * n
    dx = [0.0] * n
    for i in range(period + 1, n):
        tr_s[i] = tr_s[i - 1] - (tr_s[i - 1] / period) + tr[i]
        plus_s[i] = plus_s[i - 1] - (plus_s[i - 1] / period) + plus_dm[i]
        minus_s[i] = minus_s[i - 1] - (minus_s[i - 1] / period) + minus_dm[i]
        if tr_s[i] != 0:
            plus_di[i] = 100 * (plus_s[i] / tr_s[i])
            minus_di[i] = 100 * (minus_s[i] / tr_s[i])
        denom = plus_di[i] + minus_di[i]
        dx[i] = 0.0 if denom == 0 else 100 * abs(plus_di[i] - minus_di[i]) / denom
    first = period * 2
    out[first] = sum(dx[period + 1:first + 1]) / period
    for i in range(first + 1, n):
        prev = out[i - 1] if out[i - 1] is not None else 0.0
        out[i] = ((prev * (period - 1)) + dx[i]) / period
    return out


def higher_tf_bias(candles: List[Candle], cfg: Config) -> str:
    closes = [c.close for c in candles]
    ef = ema(closes, cfg.ema_fast)
    es = ema(closes, cfg.ema_slow)
    rv = rsi(closes, cfg.rsi_period)
    xv = adx(candles, cfg.adx_period)
    i = len(candles) - 1
    if None in (ef[i], es[i], rv[i], xv[i]):
        return "UNKNOWN"
    if ef[i] > es[i] and rv[i] >= 50 and xv[i] >= cfg.min_adx:
        return "BULLISH"
    if ef[i] < es[i] and rv[i] <= 50 and xv[i] >= cfg.min_adx:
        return "BEARISH"
    return "NEUTRAL"


def compute_final_score(confidence: int, htf_bias_value: str, signal: str) -> float:
    score = float(confidence)
    if signal == "CALL" and htf_bias_value == "BULLISH":
        score += 6
    elif signal == "PUT" and htf_bias_value == "BEARISH":
        score += 6
    elif signal in ("CALL", "PUT") and htf_bias_value not in ("BULLISH", "BEARISH"):
        score -= 6
    return max(0.0, min(100.0, round(score, 2)))


def analyze_asset(asset: str, low_tf: List[Candle], high_tf: List[Candle], cfg: Config) -> Optional[SignalResult]:
    if len(low_tf) < 120 or len(high_tf) < 60:
        return None
    closes = [c.close for c in low_tf]
    ef = ema(closes, cfg.ema_fast)
    es = ema(closes, cfg.ema_slow)
    rv = rsi(closes, cfg.rsi_period)
    xv = adx(low_tf, cfg.adx_period)
    i = len(low_tf) - 1
    last = low_tf[i]
    if None in (ef[i], es[i], rv[i], xv[i]):
        return None

    fast = float(ef[i])
    slow = float(es[i])
    rsi_val = float(rv[i])
    adx_val = float(xv[i])
    htf = higher_tf_bias(high_tf, cfg)
    signal = "NO TRADE"
    reason = "conditions not met"
    confidence = 28

    if fast > slow and cfg.min_rsi_call <= rsi_val <= cfg.max_rsi_call and last.close > last.open and adx_val >= cfg.min_adx:
        signal = "CALL"
        reason = "trend up + RSI ok + bullish candle"
        confidence = 74
    elif fast < slow and cfg.min_rsi_put <= rsi_val <= cfg.max_rsi_put and last.close < last.open and adx_val >= cfg.min_adx:
        signal = "PUT"
        reason = "trend down + RSI ok + bearish candle"
        confidence = 74

    if signal == "CALL" and htf == "BEARISH":
        signal = "NO TRADE"
        reason = "5m conflict with 1m CALL"
        confidence = 40
    if signal == "PUT" and htf == "BULLISH":
        signal = "NO TRADE"
        reason = "5m conflict with 1m PUT"
        confidence = 40

    return SignalResult(
        asset=asset,
        timestamp=last.timestamp,
        signal=signal,
        entry="next candle open" if signal in ("CALL", "PUT") else "no entry",
        expiry_minutes=cfg.expiry_minutes,
        confidence=confidence,
        higher_tf_bias=htf,
        rsi_value=round(rsi_val, 2),
        adx_value=round(adx_val, 2),
        final_score=compute_final_score(confidence, htf, signal),
        reason=reason,
    )


def choose_best_signal(results: List[SignalResult], cfg: Config) -> Optional[SignalResult]:
    ranked = sorted(results, key=lambda x: (x.signal in ("CALL", "PUT"), x.final_score), reverse=True)
    for item in ranked:
        if item.signal in ("CALL", "PUT") and item.final_score >= cfg.min_final_score:
            return item
    return ranked[0] if ranked else None


def build_snapshot() -> dict:
    if not cfg.api_key:
        return {
            "error": "Missing TWELVEDATA_API_KEY",
            "generated_at": utc_now_str(),
            "best": None,
            "results": [],
        }

    results: List[SignalResult] = []
    errors = []
    for asset in cfg.assets:
        try:
            low_tf = fetch_candles(asset, cfg.chart_tf, cfg.outputsize, cfg.api_key)
            high_tf = fetch_candles(asset, cfg.higher_tf, 100, cfg.api_key)
            sig = analyze_asset(asset, low_tf, high_tf, cfg)
            if sig:
                results.append(sig)
        except Exception as e:
            errors.append({"asset": asset, "error": str(e)})

    best = choose_best_signal(results, cfg)
    return {
        "generated_at": utc_now_str(),
        "best": asdict(best) if best else None,
        "results": [asdict(r) for r in results],
        "errors": errors,
    }


@app.get("/api/scan")
def api_scan():
    snapshot = build_snapshot()
    status = 400 if snapshot.get("error") else 200
    return jsonify(snapshot), status


HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Pocket Option Signals</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#f5f7fb; color:#1f2937; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:16px; margin-bottom:20px; }
    .card { background:#fff; border-radius:16px; padding:18px; box-shadow:0 8px 24px rgba(0,0,0,.08); }
    h1,h2 { margin:0 0 12px 0; }
    table { width:100%; border-collapse:collapse; background:#fff; border-radius:16px; overflow:hidden; }
    th,td { padding:10px; border-bottom:1px solid #e5e7eb; text-align:left; font-size:14px; }
    th { background:#eef2ff; }
    .muted { color:#6b7280; }
    button { padding:10px 14px; border:0; border-radius:10px; cursor:pointer; }
    .section { margin-bottom:20px; overflow:auto; }
  </style>
</head>
<body>
  <h1>Pocket Option Signal Web App Simple</h1>
  <div class='section grid'>
    <div class='card'>
      <h2>Best signal now</h2>
      <div id='best' class='muted'>Loading...</div>
    </div>
    <div class='card'>
      <h2>Status</h2>
      <div><button onclick='loadData()'>Refresh</button></div>
      <p class='muted' id='stamp'></p>
      <p class='muted' id='errorBox'></p>
    </div>
  </div>
  <div class='section card'>
    <h2>All results</h2>
    <table>
      <thead>
        <tr>
          <th>Asset</th><th>Time</th><th>Signal</th><th>Entry</th><th>Expiry</th><th>Confidence</th><th>HTF</th><th>RSI</th><th>ADX</th><th>Final</th><th>Reason</th>
        </tr>
      </thead>
      <tbody id='rows'></tbody>
    </table>
  </div>
<script>
async function loadData() {
  const res = await fetch('/api/scan');
  const data = await res.json();
  document.getElementById('stamp').innerText = data.generated_at || '';
  document.getElementById('errorBox').innerText = data.error || '';
  const best = data.best;
  document.getElementById('best').innerHTML = best
    ? `<b>${best.asset}</b><br>Signal: ${best.signal}<br>Entry: ${best.entry}<br>Expiry: ${best.expiry_minutes} min<br>Confidence: ${best.confidence}%<br>Final score: ${best.final_score}<br>Reason: ${best.reason}`
    : 'No signal';

  const rows = document.getElementById('rows');
  rows.innerHTML = '';
  for (const r of data.results || []) {
    rows.innerHTML += `<tr><td>${r.asset}</td><td>${r.timestamp}</td><td>${r.signal}</td><td>${r.entry}</td><td>${r.expiry_minutes}</td><td>${r.confidence}%</td><td>${r.higher_tf_bias}</td><td>${r.rsi_value}</td><td>${r.adx_value}</td><td>${r.final_score}</td><td>${r.reason}</td></tr>`;
  }
}
loadData();
setInterval(loadData, 60000);
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        use_reloader=False,
    )
