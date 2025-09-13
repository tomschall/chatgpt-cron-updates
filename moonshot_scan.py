#!/usr/bin/env python3
import os
import json
import math
import time
import argparse
import random
from typing import Dict, Any, List, Tuple

import requests
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timezone

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# =========================
# Anzeige / Output-Konfig
# =========================
DISPLAY_COLS = [
    "rank", "symbol", "name", "price_usd", "mcap_usd", "vol_24h_usd",
    "ch_24h_pct", "ch_7d_pct", "ch_30d_pct", "moonshot_score", "cg_url",
]
IMG_THUMB_SIZE = 32  # px

# Derivate-/Stable-/LP-/Bridge-Heuristik
EXCLUDE_NAME_SUBSTRINGS = [
    "aave", "amm", "bridged", "staked", "liquid staking", "lp token", "pool",
    "wrapped", "bonded", "rebasing", "rebase", "bond", "vault",
    " usdc", " usdt", "busd", "tusd", "gusd", "usd ", "usd)",
]
EXCLUDE_ID_PREFIXES = [
    "aave-", "aamm", "aammuni", "adai", "aaave", "aageur", "ausdc", "ausdt",
]

# =========================
# Defaults (werden überschrieben)
# =========================
DEFAULTS: Dict[str, Any] = {
    "VS_CURRENCY": "usd",
    "PAGES": 8,
    "PER_PAGE": 250,
    "SCAN_ORDER": "volume_desc",   # alternativ: "market_cap_asc"
    "PAGE_OFFSET": 1,
    "PAGE_STEP": 1,
    "MCAP_MAX": 250_000_000,
    "MCAP_MIN": 2_000_000,
    "VOL_MIN": 50_000,
    "TOP_N": 25,
    "RELAXED_MCAP_MIN": 1_000_000,
    "RELAXED_VOL_MIN": 25_000,
    "PRICE_MIN": 1e-7,
    "PRICE_MAX": 20_000,           # deckelt Ausreißer
    "TARGET_MIN_CANDS": 10,        # Zielanzahl Kandidaten (min)
    "MAX_RELAX_STEPS": 4,          # wie oft lockern
    "RELAX_VOL_FACTOR": 0.6,       # vol_min *= 0.6 pro Schritt
    "RELAX_MCAP_FACTOR": 0.7,      # mcap_min *= 0.7 pro Schritt
    "MAX_ATTEMPTS": 6,
    "BASE_WAIT": 2.0,
    "JITTER_MIN": 0.3,
    "JITTER_MAX": 1.7,
    "WEIGHTS": {
        "inv_mcap": 0.35,
        "vol_to_mcap": 0.25,
        "mom_7d": 0.20,
        "mom_30d": 0.20,
    },
}

API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
HEADERS = {
    "User-Agent": "MoonshotCopilot/0.5 (+github.com/tomschall)",
    **({"x-cg-demo-api-key": API_KEY} if API_KEY else {}),
}

# =========================
# Helpers
# =========================


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}")


def load_config_file(path: str = "config.json") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def apply_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    def env(key, cast=str, default=None):
        val = os.getenv(key)
        return cast(val) if val is not None else default

    overrides = {
        "VS_CURRENCY": env("VS_CURRENCY", str),
        "PAGES": env("PAGES", int),
        "PER_PAGE": env("PER_PAGE", int),
        "SCAN_ORDER": env("SCAN_ORDER", str),
        "PAGE_OFFSET": env("PAGE_OFFSET", int),
        "PAGE_STEP": env("PAGE_STEP", int),
        "MCAP_MAX": env("MCAP_MAX", int),
        "MCAP_MIN": env("MCAP_MIN", int),
        "VOL_MIN": env("VOL_MIN", int),
        "TOP_N": env("TOP_N", int),
        "RELAXED_MCAP_MIN": env("RELAXED_MCAP_MIN", int),
        "RELAXED_VOL_MIN": env("RELAXED_VOL_MIN", int),
        "PRICE_MIN": env("PRICE_MIN", float),
        "PRICE_MAX": env("PRICE_MAX", float),
        "TARGET_MIN_CANDS": env("TARGET_MIN_CANDS", int),
        "MAX_RELAX_STEPS": env("MAX_RELAX_STEPS", int),
        "RELAX_VOL_FACTOR": env("RELAX_VOL_FACTOR", float),
        "RELAX_MCAP_FACTOR": env("RELAX_MCAP_FACTOR", float),
        "MAX_ATTEMPTS": env("MAX_ATTEMPTS", int),
        "BASE_WAIT": env("BASE_WAIT", float),
        "JITTER_MIN": env("JITTER_MIN", float),
        "JITTER_MAX": env("JITTER_MAX", float),
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Moonshot scanner (adaptive)")
    p.add_argument("--pages", type=int)
    p.add_argument("--per-page", type=int)
    p.add_argument("--order", dest="SCAN_ORDER", choices=["market_cap_asc", "volume_desc"])
    p.add_argument("--page-offset", type=int)
    p.add_argument("--page-step", type=int)
    p.add_argument("--mcap-min", type=int)
    p.add_argument("--mcap-max", type=int)
    p.add_argument("--vol-min", type=int)
    p.add_argument("--top-n", type=int)
    p.add_argument("--target", dest="TARGET_MIN_CANDS", type=int)
    p.add_argument("--price-min", type=float)
    p.add_argument("--price-max", type=float)
    p.add_argument("--relaxed-mcap-min", type=int)
    p.add_argument("--relaxed-vol-min", type=int)
    p.add_argument("--max-relax-steps", type=int)
    p.add_argument("--relax-vol-factor", type=float)
    p.add_argument("--relax-mcap-factor", type=float)
    args = {k: v for k, v in vars(p.parse_args()).items() if v is not None}
    cfg.update(args)
    return cfg


def load_config() -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    cfg.update(load_config_file())
    cfg = apply_env(cfg)
    cfg = parse_args(cfg)
    # safety
    cfg["PAGES"] = max(1, cfg["PAGES"])
    cfg["PER_PAGE"] = max(10, min(250, cfg["PER_PAGE"]))
    cfg["TOP_N"] = max(1, min(100, cfg["TOP_N"]))
    return cfg


# =========================
# HTTP
# =========================
def fetch_markets(page: int, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = {
        "vs_currency": cfg["VS_CURRENCY"],
        "order": cfg["SCAN_ORDER"],
        "per_page": cfg["PER_PAGE"],
        "page": page,
        "price_change_percentage": "24h,7d,30d",
        "sparkline": "false",
    }
    last_err = None
    for attempt in range(1, cfg["MAX_ATTEMPTS"] + 1):
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
                wait = (2 ** (attempt - 1)) * cfg["BASE_WAIT"] + random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"])
            log(f"429 rate-limit page={page} attempt={attempt}/{cfg['MAX_ATTEMPTS']} → sleep {wait:.1f}s")
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last_err = e
            if 500 <= r.status_code < 600:
                wait = (2 ** (attempt - 1)) + random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"])
                log(f"{r.status_code} server err page={page} attempt={attempt} → sleep {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed to fetch page {page} after {cfg['MAX_ATTEMPTS']} attempts") from last_err


# =========================
# Data Prep
# =========================
def sanitize_df(df: pd.DataFrame, price_min: float, price_max: float) -> Tuple[pd.DataFrame, int, int]:
    num_cols = [
        "current_price", "market_cap", "total_volume",
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
        & (df["current_price"] >= price_min)
        & (df["current_price"] <= price_max)
        & df["market_cap"].notna()
        & (df["market_cap"] > 0)
        & df["total_volume"].notna()
    ].copy()
    sanity_removed = before - len(df)

    before_der = len(df)
    df = df[~df.apply(looks_derivative, axis=1)].copy()
    deriv_removed = before_der - len(df)

    return df, sanity_removed, deriv_removed


def looks_derivative(row: pd.Series) -> bool:
    n = str(row.get("name", "")).lower()
    i = str(row.get("id", "")).lower()
    s = str(row.get("symbol", "")).lower()

    if any(i.startswith(p) for p in EXCLUDE_ID_PREFIXES):
        return True
    if any(sub in n for sub in EXCLUDE_NAME_SUBSTRINGS):
        return True

    # zusätzliche Stable/Peg-Erkennung
    ch24 = float(row.get("price_change_percentage_24h_in_currency") or 0)
    if (("usd" in n) or ("usd" in s)) and abs(ch24) < 0.5:
        return True
    return False


def stablecoin_mask(df: pd.DataFrame) -> pd.Series:
    maybe = df["name"].str.contains("usd", case=False, na=False) | df["symbol"].str.contains("usd", case=False, na=False)
    lowv = df["price_change_percentage_24h_in_currency"].abs().fillna(0) < 0.2
    return (maybe & lowv)


def winsorize(series: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def zscore(series: pd.Series) -> pd.Series:
    s = series.fillna(series.median())
    std = s.std(ddof=0)
    if std == 0 or math.isnan(std):
        return pd.Series([0] * len(s), index=s.index)
    z = (s - s.mean()) / std
    return z.clip(-3, 3)


def engineer_and_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    df["inv_mcap_feat"] = 1.0 / df["market_cap"].astype(float)
    df["vol_to_mcap_feat"] = (df["total_volume"] / df["market_cap"]).astype(float)
    df["mom_7d_feat"] = df["price_change_percentage_7d_in_currency"].astype(float)
    df["mom_30d_feat"] = df["price_change_percentage_30d_in_currency"].astype(float)

    df["inv_mcap_feat"] = winsorize(df["inv_mcap_feat"])
    df["vol_to_mcap_feat"] = winsorize(df["vol_to_mcap_feat"])
    df["mom_7d_feat"] = winsorize(df["mom_7d_feat"])
    df["mom_30d_feat"] = winsorize(df["mom_30d_feat"])

    df["z_inv_mcap"] = zscore(df["inv_mcap_feat"])
    df["z_vol_to_mcap"] = zscore(df["vol_to_mcap_feat"])
    df["z_mom_7d"] = zscore(df["mom_7d_feat"])
    df["z_mom_30d"] = zscore(df["mom_30d_feat"])

    df["moonshot_score"] = (
        weights["inv_mcap"] * df["z_inv_mcap"]
        + weights["vol_to_mcap"] * df["z_vol_to_mcap"]
        + weights["mom_7d"] * df["z_mom_7d"]
        + weights["mom_30d"] * df["z_mom_30d"]
    )
    return df


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def ensure(col: str, fallback_cols: list[str] = None):
        if col in df.columns:
            return
        for fb in (fallback_cols or []):
            if fb in df.columns:
                df[col] = df[fb]
                return
        df[col] = pd.NA

    # Zahlenfelder (akzeptiere raw oder schon umbenannt)
    ensure("current_price", ["price_usd"])
    ensure("market_cap", ["mcap_usd"])
    ensure("total_volume", ["vol_24h_usd"])
    ensure("price_change_percentage_24h_in_currency", ["ch_24h_pct"])
    ensure("price_change_percentage_7d_in_currency", ["ch_7d_pct"])
    ensure("price_change_percentage_30d_in_currency", ["ch_30d_pct"])

    # Identität/Meta
    ensure("id")
    ensure("symbol")
    ensure("name")
    ensure("image")

    # Rank robust ableiten
    if "rank" not in df.columns:
        if "market_cap_rank" in df.columns:
            df["rank"] = df["market_cap_rank"]
        else:
            df["rank"] = pd.NA

    # Zahlen konvertieren & runden
    df["price_usd"] = pd.to_numeric(df["current_price"], errors="coerce").round(6)
    df["mcap_usd"] = pd.to_numeric(df["market_cap"], errors="coerce").round(0)
    df["vol_24h_usd"] = pd.to_numeric(df["total_volume"], errors="coerce").round(0)
    df["ch_24h_pct"] = pd.to_numeric(df["price_change_percentage_24h_in_currency"], errors="coerce").round(3)
    df["ch_7d_pct"] = pd.to_numeric(df["price_change_percentage_7d_in_currency"], errors="coerce").round(3)
    df["ch_30d_pct"] = pd.to_numeric(df["price_change_percentage_30d_in_currency"], errors="coerce").round(3)

    # Links & Thumbnails
    df["cg_url"] = "https://www.coingecko.com/en/coins/" + df["id"].astype(str)

    def img_md(url: str) -> str:
        if not isinstance(url, str) or not url:
            return ""
        return f'<img alt="logo" src="{url}" width="{IMG_THUMB_SIZE}" height="{IMG_THUMB_SIZE}" />'

    df["image_md"] = df["image"].apply(img_md)
    return df


def format_top(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Wählt Top N nach Score und baut Anzeige-Spalten.
    Wichtig: KEINE Umbenennungen vor add_display_columns() mehr!
    """
    base_cols = [
        "market_cap_rank", "id", "symbol", "name", "image",
        "current_price", "market_cap", "total_volume",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
        "price_change_percentage_30d_in_currency",
        "moonshot_score",
    ]
    # Nimm nur Spalten, die existieren (robust gegen API-Variationen)
    cols = [c for c in base_cols if c in df.columns] + ["moonshot_score"]
    cols = list(dict.fromkeys(cols))  # de-dupe, Reihenfolge wahren

    out = df.sort_values("moonshot_score", ascending=False)[cols].head(top_n).copy()
    out = add_display_columns(out)

    # leichte Sort-Stabilität (Score desc, Rank asc falls da)
    if "rank" in out.columns:  # 'rank' kommt ggf. aus add_display_columns nicht; dann nutze market_cap_rank
        out = out.sort_values(["moonshot_score", "rank"], ascending=[False, True])
    elif "market_cap_rank" in out.columns:
        out = out.sort_values(["moonshot_score", "market_cap_rank"], ascending=[False, True])

    return out


# =========================
# Reporting
# =========================
def render_report_md_table(table_df: pd.DataFrame) -> str:
    # nur Spalten nehmen, die da sind
    want = ["image_md"] + DISPLAY_COLS
    have = [c for c in want if c in table_df.columns]
    table = table_df[have].rename(columns={"image_md": "logo"}).copy()

    # Formatierungen
    if "price_usd" in table.columns:
        table["price_usd"] = table["price_usd"].map(lambda v: f"{v:,.6f}")
    for c in ("mcap_usd", "vol_24h_usd"):
        if c in table.columns:
            table[c] = table[c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    for c in ("ch_24h_pct", "ch_7d_pct", "ch_30d_pct", "moonshot_score"):
        if c in table.columns:
            table[c] = table[c].map(lambda v: f"{float(v):.3f}" if pd.notna(v) else "")
    return tabulate(table, headers="keys", tablefmt="github", showindex=False)


def render_gallery_html(df: pd.DataFrame) -> str:
    cards = []
    for _, r in df.iterrows():
        price = f"${r['price_usd']:,}" if pd.notna(r["price_usd"]) else "-"
        cards.append(
            f'<p align="left">'
            f'{r.get("image_md", "")}<br/>'
            f'<a href="{r["cg_url"]}">{r["name"]}</a><br/>'
            f'<code>{str(r["symbol"]).upper()}</code><br/>'
            f'{price}<br/>'
            f'</p>'
        )
    return "<p>" + "\n".join(cards) + "</p>\n"


def render_header_desc(filters: Dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"# Moonshot Report\n\nStand: **{ts}**\n\n"
    desc = (
        "Kriterien: Market Cap ≥ {mmin:,} & ≤ {mmax:,} USD, 24h-Volumen ≥ {vmin:,}. "
        "Stable/Derivate/Bridges/LPs ausgeschlossen. Preise sanitisiert, "
        "Features winsorized (1–99%). Score = gewichtete Z-Scores: "
        "kleiner MCAP (besser), Volumen/MCAP, Momentum 7d & 30d.\n\n"
        "**Hinweis:** Nur Research/Signale, keine Finanzberatung.\n\n"
    ).format(mmin=filters["MCAP_MIN"], mmax=filters["MCAP_MAX"], vmin=filters["VOL_MIN"])
    return header + desc


def render_stats_md(stats: Dict[str, Any], weights: Dict[str, float]) -> str:
    base = (
        "\n\n### Scan-Stats\n"
        f"- Pages requested: **{stats['pages']}** à {stats['per_page']} Items\n"
        f"- Fetched rows: **{stats['fetched']}**\n"
        f"- Sanitized removed: **{stats['sanitized_removed']}**\n"
        f"- Derivative removed: **{stats['derivative_removed']}**\n"
        f"- After MCAP/VOL: **{stats['after_base_filter']}**\n"
        f"- Stable-like removed: **{stats['stable_removed']}**\n"
        f"- Remaining for scoring: **{stats['remaining']}**\n"
        f"- Mode: **{stats['mode']}**\n"
        f"- Order: **{stats['order']}** (offset {stats['page_offset']}, step {stats['page_step']})\n"
        f"- Filters used: mcap_min={stats['used_mcap_min']:,}, "
        f"mcap_max={stats['used_mcap_max']:,}, vol_min={stats['used_vol_min']:,}\n"
    )
    weights_md = f"\n\n*Weights:* {json.dumps(weights)}  \n*Quelle:* CoinGecko `/coins/markets`  \n"
    nxt = "*Nächste Ausbaustufe:* Dev-Aktivität (GitHub), Social-Momentum, Narrativ-Erkennung.\n"
    return base + weights_md + nxt


# =========================
# Pipelines
# =========================
def strict_pipeline(raw: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df, san_removed, der_removed = sanitize_df(raw, cfg["PRICE_MIN"], cfg["PRICE_MAX"])
    df = df[
        (df["market_cap"] >= cfg["MCAP_MIN"])
        & (df["market_cap"] <= cfg["MCAP_MAX"])
        & (df["total_volume"] >= cfg["VOL_MIN"])
    ].copy()
    after_base = len(df)

    st_mask = stablecoin_mask(df)
    stable_cnt = int(st_mask.sum())
    df = df[~st_mask].copy()
    remaining = len(df)

    return df, {
        "sanitized_removed": san_removed,
        "derivative_removed": der_removed,
        "after_base_filter": after_base,
        "stable_removed": stable_cnt,
        "remaining": remaining,
        "mode": "strict",
    }


def relaxed_pipeline(raw: pd.DataFrame, cfg: Dict[str, Any], mcap_min: int, vol_min: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df, san_removed, der_removed = sanitize_df(raw, cfg["PRICE_MIN"], cfg["PRICE_MAX"])
    df = df[
        (df["market_cap"] >= mcap_min)
        & (df["market_cap"] <= cfg["MCAP_MAX"])
        & (df["total_volume"] >= vol_min)
    ].copy()
    after_base = len(df)

    st_mask = stablecoin_mask(df)
    stable_cnt = int(st_mask.sum())
    df = df[~st_mask].copy()
    remaining = len(df)

    return df, {
        "sanitized_removed": san_removed,
        "derivative_removed": der_removed,
        "after_base_filter": after_base,
        "stable_removed": stable_cnt,
        "remaining": remaining,
        "mode": "relaxed",
    }


# =========================
# Main
# =========================
def main() -> None:
    cfg = load_config()

    # Daten abrufen (mit Offset/Step Rotation)
    all_rows: List[Dict[str, Any]] = []
    page = cfg["PAGE_OFFSET"]
    for _ in range(cfg["PAGES"]):
        data = fetch_markets(page, cfg)
        all_rows.extend(data)
        page += cfg["PAGE_STEP"]
        time.sleep(cfg["BASE_WAIT"] + random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"]))

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

    weights = cfg["WEIGHTS"]

    # 1) strict
    strict_df, strict_stats = strict_pipeline(raw, cfg)
    used_mcap_min = cfg["MCAP_MIN"]
    used_vol_min = cfg["VOL_MIN"]
    stats_common = {
        "pages": cfg["PAGES"],
        "per_page": cfg["PER_PAGE"],
        "fetched": fetched,
        "order": cfg["SCAN_ORDER"],
        "page_offset": cfg["PAGE_OFFSET"],
        "page_step": cfg["PAGE_STEP"],
    }

    if len(strict_df) >= cfg["TARGET_MIN_CANDS"]:
        scored = engineer_and_score(strict_df, weights)
        out = format_top(scored, cfg["TOP_N"])
        header = render_header_desc({"MCAP_MIN": used_mcap_min, "MCAP_MAX": cfg["MCAP_MAX"], "VOL_MIN": used_vol_min})
        gallery = render_gallery_html(out)
        table_md = render_report_md_table(out)

        stats = {**stats_common, **strict_stats,
                 "used_mcap_min": used_mcap_min, "used_mcap_max": cfg["MCAP_MAX"], "used_vol_min": used_vol_min}
        linebreak = "\n\n---\n\n"
        md = header + "\n\n### Top Kandidaten (Strict)\n\n" + gallery + linebreak + table_md + render_stats_md(stats, weights)

        with open("report.md", "w", encoding="utf-8") as f:
            f.write(md)
        out.to_json("report_top.json", orient="records", indent=2)
        log(f"Strict candidates: {len(out)} → wrote report.")
        return

    # 2) relaxed + auto-relax
    step = 0
    mcap_min = cfg["RELAXED_MCAP_MIN"]
    vol_min = cfg["RELAXED_VOL_MIN"]
    relaxed_df, relaxed_stats = relaxed_pipeline(raw, cfg, mcap_min, vol_min)

    while len(relaxed_df) < cfg["TARGET_MIN_CANDS"] and step < cfg["MAX_RELAX_STEPS"]:
        step += 1
        mcap_min = int(mcap_min * cfg["RELAX_MCAP_FACTOR"])
        vol_min = int(vol_min * cfg["RELAX_VOL_FACTOR"])
        log(f"Auto-relax step {step}: mcap_min={mcap_min:,}, vol_min={vol_min:,}")
        relaxed_df, relaxed_stats = relaxed_pipeline(raw, cfg, mcap_min, vol_min)

    if not relaxed_df.empty:
        scored = engineer_and_score(relaxed_df, weights)
        out = format_top(scored, cfg["TOP_N"])
        header = render_header_desc({"MCAP_MIN": mcap_min, "MCAP_MAX": cfg["MCAP_MAX"], "VOL_MIN": vol_min})
        gallery = render_gallery_html(out)
        table_md = render_report_md_table(out)

        stats = {**stats_common, **relaxed_stats,
                 "used_mcap_min": mcap_min, "used_mcap_max": cfg["MCAP_MAX"], "used_vol_min": vol_min}
        md = header + "\n\n### Top Kandidaten (Relaxed)\n\n" + gallery + table_md + render_stats_md(stats, weights)

        with open("report.md", "w", encoding="utf-8") as f:
            f.write(md)
        out.to_json("report_top.json", orient="records", indent=2)
        log(f"Relaxed candidates: {len(out)} → wrote report.")
        return

    # 3) letzter Fallback: Watchlist (immer mit Bild + Link + reichen Feldern)
    watch = (
        raw[raw["market_cap"].notna() & (raw["market_cap"] > 0) & (raw["market_cap"] <= cfg["MCAP_MAX"])]
        .sort_values("market_cap", ascending=True)
        .head(max(cfg["TOP_N"], 25))
        .copy()
    )
    watch, san_r, der_r = sanitize_df(watch, cfg["PRICE_MIN"], cfg["PRICE_MAX"])
    watch = add_display_columns(watch)

    header = render_header_desc({"MCAP_MIN": cfg["MCAP_MIN"], "MCAP_MAX": cfg["MCAP_MAX"], "VOL_MIN": cfg["VOL_MIN"]})
    gallery = render_gallery_html(watch)
    table_md = render_report_md_table(watch)

    stats = {
        **stats_common,
        "sanitized_removed": san_r,
        "derivative_removed": der_r,
        "after_base_filter": len(watch),
        "stable_removed": 0,
        "remaining": len(watch),
        "mode": "watchlist",
        "used_mcap_min": cfg["MCAP_MIN"],
        "used_mcap_max": cfg["MCAP_MAX"],
        "used_vol_min": cfg["VOL_MIN"],
    }
    md = header + "\n\n### Watchlist (Relaxed Filters)\n\n" + gallery + table_md + render_stats_md(stats, DEFAULTS["WEIGHTS"])

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)
    watch.to_json("report_top.json", orient="records", indent=2)
    log("No candidates even after relaxed filters. Wrote Watchlist.")


if __name__ == "__main__":
    main()
