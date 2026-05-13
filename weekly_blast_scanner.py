"""
Weekly Pre-Blast Setup Scanner
================================
Scans all NSE stocks for:
  1. EMA10 > EMA20 > MA40 (weekly stacking)
  2. RSI(14) > RSI 20-SMA (momentum shift)
  3. RSI in 50–68 zone (coiling, not overbought)
  4. Weekly volume < 20W avg volume (dryup)
  5. Weekly candle range < 80% of 10W avg range (contraction)
  6. +DI > -DI in DMI (directional bias bullish)

Run: python weekly_blast_scanner.py
Schedule: Every Saturday via GitHub Actions or cron
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# How many weeks of data to fetch
WEEKLY_BARS = 130   # ~2.5 years

# RSI coiling band
RSI_LOW  = 50
RSI_HIGH = 68

# Volume dryup threshold (current vol as fraction of 20W avg)
VOL_DRYUP_RATIO = 0.95

# Range contraction threshold (current range as fraction of 10W avg range)
RANGE_CONTRACT_RATIO = 0.80

# Batch size for yfinance (avoid rate limits)
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 4  # seconds

# ── NSE SYMBOL LIST ─────────────────────────────────────────────────────────
def get_nse_symbols():
    """
    Fetch full NSE equity symbol list from NSE website.
    Falls back to a hardcoded list if download fails.
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        symbols = df["SYMBOL"].dropna().tolist()
        print(f"[+] Fetched {len(symbols)} NSE symbols")
        return symbols
    except Exception as e:
        print(f"[!] NSE symbol fetch failed: {e}")
        print("[!] Using fallback list")
        return FALLBACK_SYMBOLS

# Fallback — key midcap/smallcap names for testing
FALLBACK_SYMBOLS = [
    "MCX", "ANGELONE", "DATAPATTNS", "SHAILY", "AVALON", "AZAD",
    "PGEL", "ABB", "SIEMENS", "GROWW", "OSWALGREEN", "KPIGREEN",
    "RPSGVENT", "GALLANTT", "BAJAJCON", "ATHER",
]

# ── INDICATORS ───────────────────────────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def dmi(high, low, close, period=14):
    """Returns +DI, -DI, ADX"""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    up_move   = high - high.shift()
    down_move = low.shift() - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr       = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()  # proxy
    tr_smooth = tr.ewm(span=period, adjust=False).mean()

    plus_di  = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / tr_smooth

    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()

    return plus_di, minus_di, adx

# ── SCREENING LOGIC ──────────────────────────────────────────────────────────
def screen_symbol(symbol):
    """
    Downloads weekly data and checks all 6 pre-blast traits.
    Returns a dict with result if setup found, else None.
    """
    ticker = symbol + ".NS"
    try:
        df = yf.download(
            ticker,
            period="3y",
            interval="1wk",
            progress=False,
            auto_adjust=True
        )
        if df is None or len(df) < 60:
            return None

        df.dropna(subset=["Close"], inplace=True)
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # ── Trait 1+2: EMA stacking
        e10  = ema(close, 10)
        e20  = ema(close, 20)
        ma40 = sma(close, 40)

        stack_ok = (
            e10.iloc[-1] > e20.iloc[-1] and
            e20.iloc[-1] > ma40.iloc[-1] and
            close.iloc[-1] > ma40.iloc[-1]
        )
        if not stack_ok:
            return None

        # ── Trait 3: RSI > RSI-SMA
        r = rsi(close, 14)
        r_sma = sma(r, 20)
        rsi_cross_ok = r.iloc[-1] > r_sma.iloc[-1]
        if not rsi_cross_ok:
            return None

        # ── Trait 4: RSI in 50–68 zone
        rsi_val = round(float(r.iloc[-1]), 1)
        rsi_zone_ok = RSI_LOW <= rsi_val <= RSI_HIGH
        if not rsi_zone_ok:
            return None

        # ── Trait 5a: Volume dryup (current < 20W avg)
        vol_avg_20 = sma(volume, 20).iloc[-1]
        vol_current = volume.iloc[-1]
        vol_ok = vol_current < (vol_avg_20 * VOL_DRYUP_RATIO)
        if not vol_ok:
            return None

        # ── Trait 5b: Range contraction
        wk_range     = (high - low)
        range_avg_10 = sma(wk_range, 10).iloc[-1]
        range_curr   = wk_range.iloc[-1]
        range_ok = range_curr < (range_avg_10 * RANGE_CONTRACT_RATIO)
        if not range_ok:
            return None

        # ── Trait 6: +DI > -DI
        plus_di, minus_di, adx_line = dmi(high, low, close, 14)
        dmi_ok = float(plus_di.iloc[-1]) > float(minus_di.iloc[-1])
        if not dmi_ok:
            return None

        # ── All passed — build result
        entry_ref  = round(float(close.iloc[-1]), 2)
        sl_ref     = round(float(e20.iloc[-1]) * 0.98, 2)   # 2% below EMA20 weekly
        risk_pct   = round((entry_ref - sl_ref) / entry_ref * 100, 1)
        t1_ref     = round(entry_ref * 1.25, 2)              # ~25% T1
        rr_ratio   = round(0.25 / (risk_pct / 100), 1)

        return {
            "symbol"    : symbol,
            "close"     : entry_ref,
            "ema10"     : round(float(e10.iloc[-1]), 2),
            "ema20"     : round(float(e20.iloc[-1]), 2),
            "ma40"      : round(float(ma40.iloc[-1]), 2),
            "rsi"       : rsi_val,
            "sl"        : sl_ref,
            "risk_pct"  : risk_pct,
            "t1"        : t1_ref,
            "rr"        : rr_ratio,
            "+di"       : round(float(plus_di.iloc[-1]), 1),
            "-di"       : round(float(minus_di.iloc[-1]), 1),
        }

    except Exception as e:
        # Silently skip bad tickers
        return None

# ── TELEGRAM ALERT ───────────────────────────────────────────────────────────
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram not configured — printing to console only")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id"    : TELEGRAM_CHAT_ID,
        "text"       : message,
        "parse_mode" : "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print("[+] Telegram sent")
    except Exception as e:
        print(f"[!] Telegram error: {e}")

def format_alert(results):
    date_str = datetime.now().strftime("%d %b %Y")
    lines = [
        f"📊 <b>Weekly Pre-Blast Scanner — {date_str}</b>",
        f"Found <b>{len(results)}</b> setup(s)\n",
    ]
    for r in results:
        lines.append(
            f"🟢 <b>{r['symbol']}</b>  ₹{r['close']}\n"
            f"   EMA: {r['ema10']} / {r['ema20']} / {r['ma40']}\n"
            f"   RSI: {r['rsi']}  |  +DI: {r['+di']}  -DI: {r['-di']}\n"
            f"   SL: ₹{r['sl']} ({r['risk_pct']}%)  →  T1: ₹{r['t1']}  R:R ~1:{r['rr']}\n"
        )
    if not results:
        lines.append("No setups today. Market not ready.")
    return "\n".join(lines)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Weekly Pre-Blast Scanner")
    print(f"  {datetime.now().strftime('%A, %d %b %Y  %H:%M')}")
    print("=" * 55)

    symbols = get_nse_symbols()
    total   = len(symbols)
    results = []

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        print(f"\n[Batch {i//BATCH_SIZE + 1}]  {i+1}–{min(i+BATCH_SIZE, total)} / {total}")
        for sym in batch:
            r = screen_symbol(sym)
            if r:
                results.append(r)
                print(f"  ✅  {sym}  RSI={r['rsi']}  R:R=1:{r['rr']}")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Sort by R:R descending
    results.sort(key=lambda x: x["rr"], reverse=True)

    print(f"\n{'='*55}")
    print(f"  SETUPS FOUND: {len(results)}")
    print(f"{'='*55}")

    alert_msg = format_alert(results)
    send_telegram(alert_msg)

    # Also save to CSV
    if results:
        out_df = pd.DataFrame(results)
        fname  = f"blast_setups_{datetime.now().strftime('%Y%m%d')}.csv"
        out_df.to_csv(fname, index=False)
        print(f"[+] Saved → {fname}")

if __name__ == "__main__":
    main()
