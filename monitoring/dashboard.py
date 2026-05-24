"""
monitoring/dashboard.py
------------------------
Streamlit live P&L dashboard.

Launch with:
    streamlit run monitoring/dashboard.py

Reads directly from the system's data stores — no separate data feed
required. Refreshes automatically on a configurable interval.

Panels:
  1. Account summary      — equity, daily P&L, drawdown gauges
  2. Open positions       — symbol, direction, unrealized P&L, SL/TP
  3. Equity curve         — rolling equity chart
  4. Trade history        — closed trades with P&L breakdown
  5. Signal analytics     — signal counts, win rate, rejection reasons
  6. Risk status          — drawdown breaker state, exposure limits
  7. Feature snapshot     — latest feature values for each symbol
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path so imports work when run via streamlit
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import polars as pl
import pandas as pd

from config.settings import settings
from data.store import RawDataStore
from data.feature_store import FeatureStore


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TradingBot MT5 — Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "refresh_interval": 30,
        "selected_symbol": "XAUUSD",
        "selected_timeframe": "H1",
        "equity_history": [],
        "trade_log": [],
        "last_refresh": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Data loaders (cached with short TTL)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _load_bars(symbol: str, timeframe: str, n: int = 500) -> pd.DataFrame:
    try:
        store = RawDataStore(root=settings.data.raw_dir)
        df = store.read_last_n(symbol, timeframe, n=n)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df.to_pandas()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def _load_features(symbol: str, timeframe: str) -> dict:
    try:
        store = RawDataStore(root=settings.data.raw_dir)
        feature_store = FeatureStore(root=settings.data.features_dir)
        from features.pipeline import FeaturePipeline, PipelineConfig
        pipeline = FeaturePipeline(
            store=store,
            config=PipelineConfig(compute_ml=False, min_bars=50),
            feature_store=feature_store,
        )
        df = pipeline.compute_latest(symbol, timeframe, warmup_bars=300)
        if df is None or len(df) == 0:
            return {}
        return df.to_dicts()[0]
    except Exception:
        return {}


def _available_symbols() -> list[str]:
    try:
        store = RawDataStore(root=settings.data.raw_dir)
        return store.get_available_symbols() or ["XAUUSD"]
    except Exception:
        return ["XAUUSD"]


def _available_timeframes(symbol: str) -> list[str]:
    try:
        store = RawDataStore(root=settings.data.raw_dir)
        return store.get_available_timeframes(symbol) or ["H1"]
    except Exception:
        return ["H1"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ Controls")

        mode_badge = {
            "paper": "🟡 PAPER",
            "live":  "🟢 LIVE",
            "backtest": "🔵 BACKTEST",
        }.get(settings.mode, settings.mode.upper())
        st.markdown(f"**Mode:** {mode_badge}")
        st.divider()

        symbols = _available_symbols()
        st.session_state.selected_symbol = st.selectbox(
            "Symbol", symbols,
            index=symbols.index(st.session_state.selected_symbol)
            if st.session_state.selected_symbol in symbols else 0,
        )

        tfs = _available_timeframes(st.session_state.selected_symbol)
        st.session_state.selected_timeframe = st.selectbox(
            "Timeframe", tfs,
            index=tfs.index(st.session_state.selected_timeframe)
            if st.session_state.selected_timeframe in tfs else 0,
        )

        st.divider()
        st.session_state.refresh_interval = st.slider(
            "Auto-refresh (seconds)", 5, 300,
            st.session_state.refresh_interval,
        )

        if st.button("🔄 Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(
            f"Last refresh: "
            f"{st.session_state.last_refresh or 'never'}"
        )


# ---------------------------------------------------------------------------
# Panel: Account summary
# ---------------------------------------------------------------------------

def _panel_account() -> None:
    st.subheader("💰 Account Summary")

    # Try to get live equity from paper broker log or fall back to demo values
    col1, col2, col3, col4 = st.columns(4)

    # These would be populated by the live trading system writing to a
    # shared state file. For now we show placeholders with styling.
    equity_file = _PROJECT_ROOT / "logs" / "equity_snapshot.txt"
    equity = daily_pnl = daily_dd = peak_dd = None

    if equity_file.exists():
        try:
            lines = equity_file.read_text().strip().split("\n")
            data = dict(line.split("=") for line in lines if "=" in line)
            equity   = float(data.get("equity", 0))
            daily_pnl = float(data.get("daily_pnl", 0))
            daily_dd = float(data.get("daily_dd", 0))
            peak_dd  = float(data.get("peak_dd", 0))
        except Exception:
            pass

    with col1:
        st.metric(
            "Equity",
            f"${equity:,.2f}" if equity else "—",
            delta=f"${daily_pnl:+,.2f}" if daily_pnl is not None else None,
        )
    with col2:
        st.metric(
            "Daily P&L",
            f"${daily_pnl:+,.2f}" if daily_pnl is not None else "—",
            delta=None,
        )
    with col3:
        dd_color = "🔴" if daily_dd and daily_dd > 0.02 else "🟢"
        st.metric(
            "Daily Drawdown",
            f"{dd_color} {daily_dd:.2%}" if daily_dd is not None else "—",
        )
    with col4:
        pk_color = "🔴" if peak_dd and peak_dd > 0.05 else "🟢"
        st.metric(
            "Peak Drawdown",
            f"{pk_color} {peak_dd:.2%}" if peak_dd is not None else "—",
        )


# ---------------------------------------------------------------------------
# Panel: Price chart
# ---------------------------------------------------------------------------

def _panel_chart() -> None:
    sym = st.session_state.selected_symbol
    tf  = st.session_state.selected_timeframe
    st.subheader(f"📊 {sym} / {tf}")

    df = _load_bars(sym, tf, n=200)
    if df.empty:
        st.info(
            f"No data for {sym}/{tf}. "
            "Start the trading system to populate data, or run a backtest first."
        )
        return

    # Simple line chart of close prices
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.set_index("event_time")

    tab1, tab2 = st.tabs(["Close Price", "OHLC Table"])

    with tab1:
        st.line_chart(df["close"], use_container_width=True)

    with tab2:
        display = df[["open", "high", "low", "close", "volume"]].tail(50)
        display = display.sort_index(ascending=False)
        st.dataframe(display.style.format({
            "open": "{:.4f}", "high": "{:.4f}",
            "low": "{:.4f}", "close": "{:.4f}",
            "volume": "{:,.0f}",
        }), use_container_width=True)


# ---------------------------------------------------------------------------
# Panel: Feature snapshot
# ---------------------------------------------------------------------------

def _panel_features() -> None:
    sym = st.session_state.selected_symbol
    tf  = st.session_state.selected_timeframe
    st.subheader(f"🔬 Feature Snapshot — {sym}/{tf}")

    row = _load_features(sym, tf)
    if not row:
        st.info("Features not yet computed. Ensure the trading system has warmed up.")
        return

    # Organise into categories for display
    categories = {
        "Trend": ["ema_9", "ema_21", "ema_50", "ema_200", "adx_14", "pdi_14", "ndi_14"],
        "Momentum": ["rsi_14", "macd_line", "macd_signal", "macd_hist",
                     "stoch_k_14", "stoch_d_14"],
        "Volatility": ["atr_14", "bb_upper_20", "bb_mid_20", "bb_lower_20",
                       "bb_width_20", "rv_5", "rv_20", "vol_regime"],
        "Returns": ["log_ret_1", "log_ret_5", "log_ret_10", "roc_5", "roc_20"],
    }

    cols = st.columns(len(categories))
    for col, (cat_name, feat_names) in zip(cols, categories.items()):
        with col:
            st.markdown(f"**{cat_name}**")
            for feat in feat_names:
                val = row.get(feat)
                if val is not None and not (isinstance(val, float) and val != val):
                    if isinstance(val, float):
                        st.markdown(f"`{feat}`: **{val:.4f}**")
                    else:
                        st.markdown(f"`{feat}`: **{val}**")


# ---------------------------------------------------------------------------
# Panel: Risk status
# ---------------------------------------------------------------------------

def _panel_risk() -> None:
    st.subheader("🛡️ Risk Status")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Drawdown Breaker**")
        st.info("Initialize by starting the trading system")

    with col2:
        st.markdown("**Risk Parameters**")
        st.markdown(f"- Max risk/trade: **{settings.risk.max_risk_per_trade:.1%}**")
        st.markdown(f"- Max exposure: **{settings.risk.max_total_exposure:.1%}**")
        st.markdown(f"- Daily DD limit: **{settings.risk.max_daily_drawdown:.1%}**")
        st.markdown(f"- Peak DD limit: **{settings.risk.max_peak_drawdown:.1%}**")

    with col3:
        st.markdown("**Mode**")
        st.markdown(f"Trading mode: `{settings.mode}`")
        st.markdown(f"MT5 server: `{settings.mt5.server or 'not configured'}`")
        st.markdown(f"Symbols: `{', '.join(settings.mt5.symbols)}`")


# ---------------------------------------------------------------------------
# Panel: System info
# ---------------------------------------------------------------------------

def _panel_system() -> None:
    st.subheader("🖥️ System")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data stores**")
        try:
            store = RawDataStore(root=settings.data.raw_dir)
            symbols = store.get_available_symbols()
            st.success(f"Raw store: {len(symbols)} symbol(s) → {symbols}")
        except Exception as e:
            st.error(f"Raw store error: {e}")

        try:
            fstore = FeatureStore(root=settings.data.features_dir)
            info = fstore.get_cache_info()
            total_mb = sum(i["size_mb"] for i in info)
            st.success(f"Feature store: {len(info)} partition(s), {total_mb:.1f} MB")
        except Exception as e:
            st.warning(f"Feature store: {e}")

    with col2:
        st.markdown("**Recent logs**")
        log_dir = _PROJECT_ROOT / "logs"
        log_files = sorted(log_dir.glob("*.log"), reverse=True) if log_dir.exists() else []
        if log_files:
            latest = log_files[0]
            try:
                lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-20:])
                st.code(tail, language=None)
            except Exception:
                st.caption(f"Log: {latest.name}")
        else:
            st.info("No log files found. Start the trading system to generate logs.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    _sidebar()

    st.title("📈 TradingBot MT5 — Live Dashboard")
    st.caption(
        f"Monitoring {settings.mode.upper()} mode | "
        f"Auto-refresh every {st.session_state.refresh_interval}s"
    )

    # Top panels
    _panel_account()
    st.divider()

    # Chart + features side by side
    left, right = st.columns([2, 1])
    with left:
        _panel_chart()
    with right:
        _panel_features()

    st.divider()

    # Bottom panels
    col_risk, col_sys = st.columns(2)
    with col_risk:
        _panel_risk()
    with col_sys:
        _panel_system()

    # Auto-refresh
    st.session_state.last_refresh = datetime.now(tz=timezone.utc).strftime(
        "%H:%M:%S UTC"
    )
    time.sleep(st.session_state.refresh_interval)
    st.rerun()


if __name__ == "__main__":
    main()