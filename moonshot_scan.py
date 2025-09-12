#!/usr/bin/env python3
# ruff: noqa
import os
import time
import math
import json
import random
import requests
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timezone
from typing import Dict, Any, List

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# ------------------------
# Konfig (ENV überschreibt)
# ------------------------
VS_CURRENCY = os.getenv("VS_CURRENCY", "usd")
PAGES = int(os.getenv("PAGES", "8"))
PER_PAGE = int(os.getenv("PER_PAGE", "250"))
MCAP_MAX = int(os.getenv("MCAP_MAX", "250000000"))
MCAP_MIN = int(os.getenv("MCAP_MIN", "2000000"))
VOL_MIN = int(os.getenv("VOL_MIN", "50000"))
TOP_N = int(os.getenv("TOP_N", "25"))

# Sanity-Bounds
PRICE_MIN = float(os.getenv("PRICE_MIN", "0.0000001"))   # 1e-7 USD
PRICE_MAX = float(os.getenv("PRICE_MAX", "100000"))      # 100k USD

# Relaxed-Fallback (wenn strikte Filter leer)
RELAXED_MCAP_MIN = int(os.getenv("RELAXED_MCAP_MIN", "1000000"))
RELAXED_VOL_MIN = int(os.getenv("RELAXED_VOL_MIN", "25000"))

# Gewichte (Summe 1.0)
WEIGHTS = {
    "inv_mcap": float(os.getenv("W_INV_MCAP", "0.35")),
    "vol_to_mcap": float(os.getenv("W_VOL_TO_MCAP", "0.25")),
    "mom_7d": float(os.getenv("W_MOM_7D", "0.20")),
    "mom_30d": float(os.getenv("W_MOM_30D", "0.20")),
}

API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
HEADERS = {
    "User-Agent": "MoonshotCopilot/0.3 (+github.com/tomschall)",
    **({"x-cg-demo-api-key": API_KEY} if API_KEY else {}),
}

# Backoff-Settings
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "6"))
BASE_WAIT = float(os.getenv("BASE_WAIT", "2.0"))
JITTER_MIN = float(os.getenv("JITTER_MIN", "0.3"))
JITTER_MAX = float(os.getenv("JITTER_MAX", "1.7"))


def log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def zscore(series: pd.Series) -> pd.Series:
    s = series.fillna(series.median())
    std = s.std(ddof=0)
    if std == 0 or math.isnan(std):
        return pd.Series([0] * len(s), index=s.index)
    z = (s - s.mean()) / std
    return z.clip(-3, 3)


def winsorize(series: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def fetch_markets(page: int, vs_currency: str) -> List[Dict[str, Any]]:
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_asc",
        "per_page": PER_PAGE,
        "page": page,
        "price_change_percentage": "24h,7d,30d",
        "sparkline": "false",
    }
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params=params,
            headers=HEADERS,
            timeout=30,
        )
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                wait = float(ra)
            else:
                wait = (2 ** (attempt - 1)) * BASE_WAIT + random.uniform(JITTER_MIN, JITTER_MAX)
            log(f"429 rate-limit page={page} attempt={attempt}/{MAX_ATTEMPTS} → sleep {wait:.1f}s")
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last_err = e
            if 500 <= r.status_code < 600:
                wait = (2 ** (attempt - 1)) + random.uniform(JITTER_MIN, JITTER_MAX)
                log(f"{r.status_code} server err page={page} attempt={attempt} → sleep {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed to fetch page {page} after {MAX_ATTEMPTS} attempts") from last_err


def stablecoin_mask(df: pd.DataFrame) -> pd.Series:
    maybe_stable = (
        df["name"].str.contains("usd", case=False, na=False)
        | df["symbol"].str.contains("usd", case=False, na=False)
    )
    low_vol = df["price_change_percentage_24h_in_currency"].abs().fillna(0) < 0.2
    return (maybe_stable & low_vol)


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filtert offensichtliche Datenfehler (price 0, NaN, extreme Preise)."""
    num_cols = [
        "current_price",
        "market_cap",
        "total_volume",
        "price_change_percentage_7d_in_currency",
        "price_change_percentage_30d_in_currency",
        "price_change_percentage_24h_in_currency",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df[
        df["current_price"].notna()
        & (df["current_price"] >= PRICE_MIN)
        & (df["current_price"] <= PRICE_MAX)
        & df["market_cap"].notna()
        & df["total_volume"].notna()
    ].copy()
    after = len(df)
    log(f"Sanitize: removed {before - after} rows due to price/mcap/volume sanity")

    return df


def engineer_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-Engineering, Winsorize und Scoring."""
    df["inv_mcap_feat"] = 1.0 / df["market_cap"].astype(float)
    df["vol_to_mcap_feat"] = (df["total_volume"] / df["market_cap"]).astype(float)
    df["mom_7d_feat"] = df["price_change_percentage_7d_in_currency"].astype(float)
    df["mom_30d_feat"] = df["price_change_percentage_30d_in_currency"].astype(float)

    # Robust gegen Ausreißer
    df["inv_mcap_feat"] = winsorize(df["inv_mcap_feat"])
    df["vol_to_mcap_feat"] = winsorize(df["vol_to_mcap_feat"])
    df["mom_7d_feat"] = winsorize(df["mom_7d_feat"])
    df["mom_30d_feat"] = winsorize(df["mom_30d_feat"])

    # Z-Scores
    df["z_inv_mcap"] = zscore(df["inv_mcap_feat"])
    df["z_vol_to_mcap"] = zscore(df["vol_to_mcap_feat"])
    df["z_mom_7d"] = zscore(df["mom_7d_feat"])
    df["z_mom_30d"] = zscore(df["mom_30d_feat"])

    # Weighted Score
    df["moonshot_score"] = (
        WEIGHTS["inv_mcap"] * df["z_inv_mcap"]
        + WEIGHTS["vol_to_mcap"] * df["z_vol_to_mcap"]
        + WEIGHTS["mom_7d"] * df["z_mom_7d"]
        + WEIGHTS["mom_30d"] * df["z_mom_30d"]
    )
    return df


def format_top(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [
        "market_cap_rank",
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
        "total_volume",
        "price_change_percentage_7d_in_currency",
        "price_change_percentage_30d_in_currency",
        "moonshot_score",
    ]
    out = df.sort_values("moonshot_score", ascending=False)[cols].head(top_n).copy()
    out.rename(
        columns={
            "market_cap_rank": "rank",
            "current_price": "price_usd",
            "market_cap": "mcap_usd",
            "total_volume": "vol_24h_usd",
            "price_change_percentage_7d_in_currency": "ch_7d_pct",
            "price_change_percentage_30d_in_currency": "ch_30d_pct",
        },
        inplace=True,
    )
    for c in ["price_usd", "mcap_usd", "vol_24h_usd"]:
        out[c] = out[c].astype(float).round(2)
    for c in ["ch_7d_pct", "ch_30d_pct", "moonshot_score"]:
        out[c] = out[c].astype(float).round(3)
    return out


def render_report_md(out: pd.DataFrame, stats: Dict[str, Any], weights: Dict[str, float], filters: Dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"# Moonshot Report\n\nStand: **{ts}**\n\n"
    desc = (
        "Kriterien: Market Cap ≥ {mmin:,} & ≤ {mmax:,} USD, 24h-Volumen ≥ {vmin:,}, "
        "Stablecoins ausgeschlossen. Preise bereinigt (0/unsinnig gefiltert), "
        "Features winsorized (1–99%).\n"
        "Score = gewichtete Z-Scores: kleiner MCAP (besser), Volumen/MCAP, Momentum 7d & 30d.\n\n"
        "**Hinweis:** Nur Research/Signale, keine Finanzberatung.\n\n"
    ).format(mmin=filters["MCAP_MIN"], mmax=filters["MCAP_MAX"], vmin=filters["VOL_MIN"])

    if out.empty:
        body = "_(Keine Kandidaten nach aktuellen Filtern gefunden.)_\n"
    else:
        table = out.copy()
        for c in ["price_usd", "mcap_usd", "vol_24h_usd"]:
            table[c] = table[c].map(lambda v: f"{v:,.2f}")
        for c in ["ch_7d_pct", "ch_30d_pct", "moonshot_score"]:
            table[c] = table[c].map(lambda v: f"{v:.3f}")
        body = tabulate(table, headers="keys", tablefmt="github", showindex=False)

    stats_md = (
        "\n\n### Scan-Stats\n"
        f"- Pages requested: **{stats['pages']}** à {stats['per_page']} Items\n"
        f"- Fetched rows: **{stats['fetched']}**\n"
        f"- Sanitized removed: **{stats['sanitized_removed']}**\n"
        f"- After MCAP/VOL: **{stats['after_base_filter']}**\n"
        f"- Stable-like removed: **{stats['stable_removed']}**\n"
        f"- Remaining for scoring: **{stats['remaining']}**\n"
        + (f"- Mode: **{stats['mode']}**\n" if "mode" in stats else "")
    )

    footer = (
        "\n\n*Weights:* "
        f"{json.dumps(weights)}  \n*Quelle:* CoinGecko `/coins/markets`  \n"
        "*Nächste Ausbaustufe:* Dev-Aktivität (GitHub), Social-Momentum, Narrativ-Erkennung.\n"
    )
    return header + desc + body + stats_md + footer


def strict_pipeline(raw: pd.DataFrame) -> (pd.DataFrame, Dict[str, int]):
    before = len(raw)
    df = sanitize_df(raw)
    sanitized_removed = before - len(df)

    df = df[
        (df["market_cap"] >= MCAP_MIN)
        & (df["market_cap"] <= MCAP_MAX)
        & (df["total_volume"] >= VOL_MIN)
    ].copy()
    after_base = len(df)

    st_mask = stablecoin_mask(df)
    stable_cnt = int(st_mask.sum())
    df = df[~st_mask].copy()
    remaining = len(df)

    return df, {
        "sanitized_removed": sanitized_removed,
        "after_base_filter": after_base,
        "stable_removed": stable_cnt,
        "remaining": remaining,
        "mode": "strict",
    }


def relaxed_pipeline(raw: pd.DataFrame) -> (pd.DataFrame, Dict[str, int]):
    before = len(raw)
    df = sanitize_df(raw)
    sanitized_removed = before - len(df)

    df = df[
        (df["market_cap"] >= RELAXED_MCAP_MIN)
        & (df["market_cap"] <= MCAP_MAX)
        & (df["total_volume"] >= RELAXED_VOL_MIN)
    ].copy()
    after_base = len(df)

    st_mask = stablecoin_mask(df)
    stable_cnt = int(st_mask.sum())
    df = df[~st_mask].copy()
    remaining = len(df)

    return df, {
        "sanitized_removed": sanitized_removed,
        "after_base_filter": after_base,
        "stable_removed": stable_cnt,
        "remaining": remaining,
        "mode": "relaxed",
    }


def main() -> None:
    t0 = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []

    for p in range(1, PAGES + 1):
        data = fetch_markets(p, VS_CURRENCY)
        all_rows.extend(data)
        time.sleep(BASE_WAIT + random.uniform(JITTER_MIN, JITTER_MAX))

    raw = pd.DataFrame(all_rows)
    fetched = len(raw)
    if fetched == 0:
        md = "# Moonshot Report\n\n*(API lieferte keine Daten – später erneut versuchen.)*\n"
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(md)
        with open("report_top.json", "w", encoding="utf-8") as f:
            f.write("[]\n")
        log("No data from API. Wrote empty report & JSON.")
        return

    # Strict
    strict_df, strict_stats = strict_pipeline(raw)

    if strict_df.empty:
        # Relaxed-Scoring Fallback (mit echtem Score & Ranking)
        relaxed_df, relaxed_stats = relaxed_pipeline(raw)
        stats = {
            "pages": PAGES,
            "per_page": PER_PAGE,
            "fetched": fetched,
            **relaxed_stats,
        }

        if relaxed_df.empty:
            # Gar nichts übrig → zeige kleinste MCAP als Watchlist
            watch = (
                raw[raw["market_cap"].notna() & (raw["market_cap"] <= MCAP_MAX)]
                .sort_values("market_cap", ascending=True)
                .head(max(TOP_N, 25))
                .copy()
            )
            for c in ["current_price", "market_cap", "total_volume",
                      "price_change_percentage_7d_in_currency",
                      "price_change_percentage_30d_in_currency"]:
                if c in watch.columns:
                    watch[c] = pd.to_numeric(watch[c], errors="coerce")
            watch = watch.rename(
                columns={
                    "market_cap_rank": "rank",
                    "current_price": "price_usd",
                    "market_cap": "mcap_usd",
                    "total_volume": "vol_24h_usd",
                    "price_change_percentage_7d_in_currency": "ch_7d_pct",
                    "price_change_percentage_30d_in_currency": "ch_30d_pct",
                }
            )
            for c in ["price_usd", "mcap_usd", "vol_24h_usd"]:
                if c in watch.columns:
                    watch[c] = watch[c].astype(float).round(2)
            for c in ["ch_7d_pct", "ch_30d_pct"]:
                if c in watch.columns:
                    watch[c] = watch[c].astype(float).round(3)

            md = render_report_md(
                out=pd.DataFrame(columns=[]),
                stats=stats,
                weights=WEIGHTS,
                filters={"MCAP_MIN": MCAP_MIN, "MCAP_MAX": MCAP_MAX, "VOL_MIN": VOL_MIN},
            )
            md += "\n\n### Watchlist (Relaxed Filters)\n"
            md += tabulate(watch, headers="keys", tablefmt="github", showindex=False)

            with open("report.md", "w", encoding="utf-8") as f:
                f.write(md)
            with open("report_top.json", "w", encoding="utf-8") as f:
                f.write(watch.to_json(orient="records", indent=2))
            log("No candidates even after relaxed filters. Wrote Watchlist.")
            return

        # Scoring im Relaxed-Modus
        scored = engineer_and_score(relaxed_df)
        out = format_top(scored, TOP_N)

        stats = {
            "pages": PAGES,
            "per_page": PER_PAGE,
            "fetched": fetched,
            **relaxed_stats,
        }
        md = render_report_md(
            out=out,
            stats=stats,
            weights=WEIGHTS,
            filters={"MCAP_MIN": RELAXED_MCAP_MIN, "MCAP_MAX": MCAP_MAX, "VOL_MIN": RELAXED_VOL_MIN},
        )
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(md)
        out.to_json("report_top.json", orient="records", indent=2)
        log(f"Relaxed candidates: {len(out)} → wrote report.")
        return

    # Scoring im Strict-Modus
    scored = engineer_and_score(strict_df)
    out = format_top(scored, TOP_N)

    stats = {
        "pages": PAGES,
        "per_page": PER_PAGE,
        "fetched": fetched,
        **strict_stats,
    }
    md = render_report_md(
        out=out,
        stats=stats,
        weights=WEIGHTS,
        filters={"MCAP_MIN": MCAP_MIN, "MCAP_MAX": MCAP_MAX, "VOL_MIN": VOL_MIN},
    )

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)
    out.to_json("report_top.json", orient="records", indent=2)

    elapsed = time.perf_counter() - t0
    log(f"Strict candidates: {len(out)}")
    log(f"Wrote report.md and report_top.json in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
