"""TradingView MCP Server — crypto & stock screener tools via TradingView APIs.

Provides 13 tools for technical analysis: scanning, filtering, and
multi-timeframe analysis using tradingview-ta and tradingview-screener.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.indicators import compute_metrics
from tradingview_mcp.core.utils.validators import (
    EXCHANGE_SCREENER,
    sanitize_exchange,
    sanitize_timeframe,
)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from tradingview_ta import get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

try:
    from tradingview_screener import Query
    from tradingview_screener.column import Column
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TIMEFRAME_TO_TV_RESOLUTION: Dict[str, str] = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1D": "1D",
    "1W": "1W",
    "1M": "1M",
}


def _tv_resolution(tf: Optional[str]) -> Optional[str]:
    """Map a timeframe string to the TradingView resolution suffix."""
    if not tf:
        return None
    return TIMEFRAME_TO_TV_RESOLUTION.get(tf)


def _percent_change(open_price: Optional[float], close: Optional[float]) -> Optional[float]:
    if open_price in (None, 0) or close is None:
        return None
    return (close - open_price) / open_price * 100


def _check_connectivity() -> Optional[str]:
    """Return an error string if we cannot reach TradingView, else None."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        import socket
        try:
            socket.create_connection(("scanner.tradingview.com", 443), timeout=5)
        except Exception:
            return (
                "Cannot reach scanner.tradingview.com — outbound HTTPS is blocked "
                "by a network proxy. This MCP server requires direct internet access "
                "to TradingView's API. Run it locally or on a server without egress "
                "restrictions (e.g. your own machine, Render, Railway, Fly.io)."
            )
    return None


def _require_ta() -> Optional[str]:
    """Return an error string if tradingview_ta is unavailable."""
    if not TRADINGVIEW_TA_AVAILABLE:
        return "tradingview_ta is not installed. Run: uv sync"
    return None


def _make_indicators(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Normalise raw indicator dict to a consistent shape."""
    return {
        "open": raw.get("open"),
        "close": raw.get("close"),
        "SMA20": raw.get("SMA20"),
        "BB_upper": raw.get("BB.upper", raw.get("BB_upper")),
        "BB_lower": raw.get("BB.lower", raw.get("BB_lower")),
        "EMA50": raw.get("EMA50"),
        "RSI": raw.get("RSI"),
        "volume": raw.get("volume"),
    }


# ---------------------------------------------------------------------------
# Compact-output helpers (reduce token count by ~70%)
# ---------------------------------------------------------------------------

def _compact_scan_row(row: dict) -> dict:
    """Reduce a scanner row to regime-critical signals only."""
    ind = row.get("indicators") or {}
    close = ind.get("close")
    bb_upper = ind.get("BB_upper")
    bb_lower = ind.get("BB_lower")
    sma20 = ind.get("SMA20")
    rsi = ind.get("RSI")
    volume = ind.get("volume")

    bb_pos = None
    if close is not None and bb_upper is not None and bb_lower is not None:
        if close > bb_upper:
            bb_pos = "above"
        elif close < bb_lower:
            bb_pos = "below"
        else:
            bb_pos = "within"

    bbw = None
    if sma20 and sma20 != 0 and bb_upper is not None and bb_lower is not None:
        bbw = round((bb_upper - bb_lower) / sma20, 4)

    out: dict = {"symbol": row.get("symbol")}
    chg = row.get("changePercent")
    if chg is not None:
        out["chg%"] = round(chg, 2)
    if rsi is not None:
        out["rsi"] = round(rsi, 1)
    if bb_pos is not None:
        out["bb_pos"] = bb_pos
    if bbw is not None:
        out["bbw"] = bbw
    if volume is not None:
        out["vol"] = int(volume)
    for key in ("volume_ratio", "volume_strength", "breakout_type", "trading_recommendation"):
        if key in row:
            out[key] = row[key]
    return out


def _compact_coin_analysis(result: dict) -> dict:
    """Flatten coin_analysis result to regime-critical signals only."""
    if "error" in result:
        return result
    pd_ = result.get("price_data", {})
    ba = result.get("bollinger_analysis", {})
    ti = result.get("technical_indicators", {})
    ms = result.get("market_sentiment", {})
    out: dict = {"symbol": result.get("symbol"), "tf": result.get("timeframe")}
    if pd_.get("change_percent") is not None:
        out["chg%"] = round(pd_["change_percent"], 2)
    if ti.get("rsi") is not None:
        out["rsi"] = round(ti["rsi"], 1)
    if ti.get("rsi_signal"):
        out["rsi_sig"] = ti["rsi_signal"]
    if ba.get("position"):
        out["bb_pos"] = ba["position"]
    if ba.get("bbw") is not None:
        out["bbw"] = round(ba["bbw"], 4)
    if ba.get("signal"):
        out["bb_sig"] = ba["signal"]
    if ti.get("adx") is not None:
        out["adx"] = round(ti["adx"], 1)
    if ti.get("trend_strength"):
        out["trend"] = ti["trend_strength"]
    if ms.get("momentum"):
        out["momentum"] = ms["momentum"]
    return out


def _compact_vol_confirmation(result: dict) -> dict:
    """Flatten volume_confirmation_analysis to key signals only."""
    if "error" in result:
        return result
    pd_ = result.get("price_data", {})
    va = result.get("volume_analysis", {})
    ti = result.get("technical_indicators", {})
    out: dict = {"symbol": result.get("symbol")}
    if pd_.get("change_percent") is not None:
        out["chg%"] = pd_["change_percent"]
    if va.get("volume_ratio") is not None:
        out["vol_ratio"] = va["volume_ratio"]
    if va.get("volume_strength"):
        out["vol_str"] = va["volume_strength"]
    if ti.get("RSI") is not None:
        out["rsi"] = ti["RSI"]
    if ti.get("BB_position"):
        out["bb_pos"] = ti["BB_position"]
    signals = result.get("signals", [])
    if signals:
        out["signals"] = signals
    return out


# ---------------------------------------------------------------------------
# Data-fetching internals
# ---------------------------------------------------------------------------

def _fetch_ta_batch(exchange: str, symbols: List[str], timeframe: str) -> dict:
    """Call get_multiple_analysis with proper error wrapping."""
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    try:
        return get_multiple_analysis(screener=screener, interval=timeframe, symbols=symbols)
    except Exception as exc:
        error_msg = str(exc)
        if "ProxyError" in error_msg or "403 Forbidden" in error_msg or "Tunnel" in error_msg:
            raise ConnectionError(
                "Cannot reach scanner.tradingview.com — blocked by network proxy. "
                "Run this MCP server locally or on a host with unrestricted internet."
            ) from exc
        raise


def _fetch_bollinger_analysis(
    exchange: str,
    timeframe: str = "4h",
    limit: int = 50,
    bbw_filter: Optional[float] = None,
) -> List[dict]:
    """Fetch analysis using tradingview_ta with bollinger band logic."""
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    symbols = symbols[: limit * 2]
    analysis = _fetch_ta_batch(exchange, symbols, timeframe)

    rows: List[dict] = []
    for key, value in analysis.items():
        if value is None:
            continue
        try:
            indicators = value.indicators
            metrics = compute_metrics(indicators)

            if not metrics or metrics.get("bbw") is None:
                continue
            if bbw_filter is not None and (metrics["bbw"] >= bbw_filter or metrics["bbw"] <= 0):
                continue
            if not (indicators.get("EMA50") and indicators.get("RSI")):
                continue

            rows.append({
                "symbol": key,
                "changePercent": metrics["change"],
                "indicators": _make_indicators(indicators),
            })
        except (TypeError, ZeroDivisionError, KeyError):
            continue

    rows.sort(key=lambda x: x["changePercent"], reverse=True)
    return rows[:limit]


def _fetch_trending_analysis(
    exchange: str,
    timeframe: str = "5m",
    filter_type: str = "",
    rating_filter: Optional[int] = None,
    limit: int = 50,
) -> List[dict]:
    """Fetch trending coins, processing in batches with early exit."""
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    batch_size = 200
    # Cap total symbols to avoid excessive API calls on large exchanges
    max_symbols = min(len(symbols), limit * 10, 1000)
    all_coins: List[dict] = []

    for i in range(0, max_symbols, batch_size):
        batch = symbols[i : i + batch_size]
        try:
            analysis = _fetch_ta_batch(exchange, batch, timeframe)
        except ConnectionError:
            raise
        except Exception:
            continue

        for key, value in analysis.items():
            if value is None:
                continue
            try:
                indicators = value.indicators
                metrics = compute_metrics(indicators)
                if not metrics or metrics.get("bbw") is None:
                    continue

                if filter_type == "rating" and rating_filter is not None:
                    if metrics["rating"] != rating_filter:
                        continue

                all_coins.append({
                    "symbol": key,
                    "changePercent": metrics["change"],
                    "indicators": _make_indicators(indicators),
                })
            except (TypeError, ZeroDivisionError, KeyError):
                continue

        # Early exit when we have enough candidates to sort from
        if filter_type != "rating" and len(all_coins) >= limit * 3:
            break

    all_coins.sort(key=lambda x: x["changePercent"], reverse=True)
    return all_coins[:limit]


def _calculate_candle_pattern_score(
    indicators: dict,
    pattern_length: int,
    min_increase: float,
) -> dict:
    """Calculate candle pattern score based on available indicators."""
    open_price = indicators.get("open", 0)
    close_price = indicators.get("close", 0)
    high_price = indicators.get("high", 0)
    low_price = indicators.get("low", 0)
    volume = indicators.get("volume", 0)
    rsi = indicators.get("RSI", 50)

    if not all([open_price, close_price, high_price, low_price]):
        return {"detected": False, "score": 0}

    candle_body = abs(close_price - open_price)
    candle_range = high_price - low_price
    body_ratio = candle_body / candle_range if candle_range > 0 else 0
    price_change = ((close_price - open_price) / open_price) * 100

    score = 0
    details: List[str] = []

    if body_ratio > 0.7:
        score += 2
        details.append("Strong candle body")
    elif body_ratio > 0.5:
        score += 1
        details.append("Moderate candle body")

    if abs(price_change) >= min_increase:
        score += 2
        details.append(f"Strong momentum ({price_change:.1f}%)")
    elif abs(price_change) >= min_increase / 2:
        score += 1
        details.append(f"Moderate momentum ({price_change:.1f}%)")

    if volume > 5000:
        score += 1
        details.append("Good volume")

    if (price_change > 0 and 50 < rsi < 80) or (price_change < 0 and 20 < rsi < 50):
        score += 1
        details.append("RSI momentum aligned")

    ema50 = indicators.get("EMA50", close_price)
    if (price_change > 0 and close_price > ema50) or (price_change < 0 and close_price < ema50):
        score += 1
        details.append("Trend alignment")

    return {
        "detected": score >= 3,
        "score": score,
        "details": details,
        "price": round(close_price, 6),
        "total_change": round(price_change, 3),
        "body_ratio": round(body_ratio, 3),
        "volume": volume,
    }


def _fetch_multi_timeframe_patterns(
    exchange: str,
    symbols: List[str],
    base_tf: str,
    length: int,
    min_increase: float,
) -> List[dict]:
    """Fetch multi-timeframe pattern data using tradingview-screener."""
    if not TRADINGVIEW_SCREENER_AVAILABLE:
        return []

    tv_interval = TIMEFRAME_TO_TV_RESOLUTION.get(base_tf, "15")
    cols = [
        f"open|{tv_interval}",
        f"close|{tv_interval}",
        f"high|{tv_interval}",
        f"low|{tv_interval}",
        f"volume|{tv_interval}",
        "RSI",
    ]

    q = Query().set_markets("crypto").select(*cols)
    q = q.where(Column("exchange") == exchange.upper())
    q = q.limit(len(symbols))

    try:
        _total, df = q.get_scanner_data()
    except Exception:
        return []

    if df is None or df.empty:
        return []

    results: List[dict] = []
    for _, row in df.iterrows():
        symbol = row.get("ticker", "")
        open_val = row.get(f"open|{tv_interval}")
        close_val = row.get(f"close|{tv_interval}")
        high_val = row.get(f"high|{tv_interval}")
        low_val = row.get(f"low|{tv_interval}")
        volume_val = row.get(f"volume|{tv_interval}", 0)
        rsi_val = row.get("RSI", 50)

        if not all([open_val, close_val, high_val, low_val]):
            continue

        pattern_score = _calculate_candle_pattern_score(
            {"open": open_val, "close": close_val, "high": high_val,
             "low": low_val, "volume": volume_val, "RSI": rsi_val},
            length, min_increase,
        )
        if pattern_score["detected"]:
            results.append({
                "symbol": symbol,
                "pattern_score": pattern_score["score"],
                "price": pattern_score["price"],
                "change": pattern_score["total_change"],
                "body_ratio": pattern_score["body_ratio"],
                "volume": volume_val,
                "rsi": round(rsi_val, 2),
                "details": pattern_score["details"],
            })

    return sorted(results, key=lambda x: x["pattern_score"], reverse=True)


# ---------------------------------------------------------------------------
# MCP server + tool definitions
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="TradingView Screener",
    instructions=(
        "Crypto & stock screener utilities backed by TradingView. "
        "Tools: top_gainers, top_losers, bollinger_scan, rating_filter, "
        "coin_analysis, consecutive_candles_scan, advanced_candle_pattern, "
        "volume_breakout_scanner, volume_confirmation_analysis, "
        "smart_volume_scanner, rsi_scanner, trend_scanner, multi_timeframe_summary."
    ),
)


@mcp.tool()
def top_gainers(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    limit: int = 25,
    compact: bool = True,
) -> list[dict]:
    """Return top gainers for an exchange and timeframe using bollinger band analysis.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))

    try:
        rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    except (ConnectionError, RuntimeError) as e:
        return [{"error": str(e)}]

    return [_compact_scan_row(r) for r in rows] if compact else rows


@mcp.tool()
def top_losers(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    limit: int = 25,
    compact: bool = True,
) -> list[dict]:
    """Return top losers for an exchange and timeframe using bollinger band analysis.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))

    try:
        # Fetch a larger pool so we can pick the worst performers
        rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit * 5)
    except (ConnectionError, RuntimeError) as e:
        return [{"error": str(e)}]

    rows.sort(key=lambda x: x["changePercent"])
    losers = rows[:limit]
    return [_compact_scan_row(r) for r in losers] if compact else losers


@mcp.tool()
def bollinger_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    bbw_threshold: float = 0.04,
    limit: int = 50,
    compact: bool = True,
) -> list[dict]:
    """Scan for coins with low Bollinger Band Width (squeeze detection).

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 100))

    try:
        rows = _fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    except (ConnectionError, RuntimeError) as e:
        return [{"error": str(e)}]

    return [_compact_scan_row(r) for r in rows] if compact else rows


@mcp.tool()
def rating_filter(
    exchange: str = "KUCOIN",
    timeframe: str = "5m",
    rating: int = 2,
    limit: int = 25,
    compact: bool = True,
) -> list[dict]:
    """Filter coins by Bollinger Band rating.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "5m")
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))

    try:
        rows = _fetch_trending_analysis(
            exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit,
        )
    except (ConnectionError, RuntimeError) as e:
        return [{"error": str(e)}]

    return [_compact_scan_row(r) for r in rows] if compact else rows


@mcp.tool()
def coin_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    compact: bool = False,
) -> dict:
    """Get detailed analysis for a specific coin on specified exchange and timeframe.

    Args:
        symbol: Coin symbol (e.g., "ACEUSDT", "BTCUSDT")
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
        compact: If True, flatten to regime-critical signals only — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return {"error": err}

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")

    if ":" not in symbol:
        full_symbol = f"{exchange.upper()}:{symbol.upper()}"
    else:
        full_symbol = symbol.upper()

    try:
        analysis = _fetch_ta_batch(exchange, [full_symbol], timeframe)
    except ConnectionError as e:
        return {"error": str(e), "symbol": symbol, "exchange": exchange, "timeframe": timeframe}
    except Exception as e:
        return {"error": f"Analysis failed: {e}", "symbol": symbol, "exchange": exchange, "timeframe": timeframe}

    if full_symbol not in analysis or analysis[full_symbol] is None:
        return {"error": f"No data found for {symbol} on {exchange}", "symbol": symbol}

    data = analysis[full_symbol]
    indicators = data.indicators
    metrics = compute_metrics(indicators)
    if not metrics:
        return {"error": f"Could not compute metrics for {symbol}", "symbol": symbol}

    macd = indicators.get("MACD.macd", 0)
    macd_signal = indicators.get("MACD.signal", 0)
    adx = indicators.get("ADX", 0)
    stoch_k = indicators.get("Stoch.K", 0)
    stoch_d = indicators.get("Stoch.D", 0)
    volume = indicators.get("volume", 0)
    high = indicators.get("high", 0)
    low = indicators.get("low", 0)
    open_price = indicators.get("open", 0)
    close_price = indicators.get("close", 0)

    result = {
        "symbol": full_symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "price_data": {
            "current_price": metrics["price"],
            "open": round(open_price, 6) if open_price else None,
            "high": round(high, 6) if high else None,
            "low": round(low, 6) if low else None,
            "close": round(close_price, 6) if close_price else None,
            "change_percent": metrics["change"],
            "volume": volume,
        },
        "bollinger_analysis": {
            "rating": metrics["rating"],
            "signal": metrics["signal"],
            "bbw": metrics["bbw"],
            "bb_upper": round(indicators.get("BB.upper", 0), 6),
            "bb_middle": round(indicators.get("SMA20", 0), 6),
            "bb_lower": round(indicators.get("BB.lower", 0), 6),
            "position": (
                "Above Upper" if close_price > indicators.get("BB.upper", 0)
                else "Below Lower" if close_price < indicators.get("BB.lower", 0)
                else "Within Bands"
            ),
        },
        "technical_indicators": {
            "rsi": round(indicators.get("RSI", 0), 2),
            "rsi_signal": (
                "Overbought" if indicators.get("RSI", 0) > 70
                else "Oversold" if indicators.get("RSI", 0) < 30
                else "Neutral"
            ),
            "sma20": round(indicators.get("SMA20", 0), 6),
            "ema50": round(indicators.get("EMA50", 0), 6),
            "ema200": round(indicators.get("EMA200", 0), 6),
            "macd": round(macd, 6),
            "macd_signal": round(macd_signal, 6),
            "macd_divergence": round(macd - macd_signal, 6),
            "adx": round(adx, 2),
            "trend_strength": "Strong" if adx > 25 else "Weak",
            "stoch_k": round(stoch_k, 2),
            "stoch_d": round(stoch_d, 2),
        },
        "market_sentiment": {
            "overall_rating": metrics["rating"],
            "buy_sell_signal": metrics["signal"],
            "volatility": "High" if metrics["bbw"] > 0.05 else "Medium" if metrics["bbw"] > 0.02 else "Low",
            "momentum": "Bullish" if metrics["change"] > 0 else "Bearish",
        },
    }
    return _compact_coin_analysis(result) if compact else result


@mcp.tool()
def consecutive_candles_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    pattern_type: str = "bullish",
    candle_count: int = 3,
    min_growth: float = 2.0,
    limit: int = 20,
) -> dict:
    """Scan for coins with consecutive growing/shrinking candles pattern.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h)
        pattern_type: "bullish" (growing candles) or "bearish" (shrinking candles)
        candle_count: Number of consecutive candles to check (2-5)
        min_growth: Minimum growth percentage for each candle
        limit: Maximum number of results to return
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return {"error": err}

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    candle_count = max(2, min(5, candle_count))
    min_growth = max(0.5, min(20.0, min_growth))
    limit = max(1, min(50, limit))

    symbols = load_symbols(exchange)
    if not symbols:
        return {"error": f"No symbols found for exchange: {exchange}"}

    symbols = symbols[: min(limit * 3, 200)]

    try:
        analysis = _fetch_ta_batch(exchange, symbols, timeframe)
    except ConnectionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Pattern analysis failed: {e}"}

    pattern_coins: List[dict] = []

    for symbol, data in analysis.items():
        if data is None:
            continue
        try:
            indicators = data.indicators
            open_price = indicators.get("open")
            close_price = indicators.get("close")
            high_price = indicators.get("high")
            low_price = indicators.get("low")
            volume = indicators.get("volume", 0)

            if not all([open_price, close_price, high_price, low_price]):
                continue

            current_change = ((close_price - open_price) / open_price) * 100
            candle_body = abs(close_price - open_price)
            candle_range = high_price - low_price
            body_to_range_ratio = candle_body / candle_range if candle_range > 0 else 0

            rsi = indicators.get("RSI", 50)
            sma20 = indicators.get("SMA20", close_price)
            ema50 = indicators.get("EMA50", close_price)
            price_above_sma = close_price > sma20
            price_above_ema = close_price > ema50

            if pattern_type == "bullish":
                conditions = [
                    current_change > min_growth,
                    body_to_range_ratio > 0.6,
                    price_above_sma,
                    45 < rsi < 80,
                    volume > 1000,
                ]
            else:
                conditions = [
                    current_change < -min_growth,
                    body_to_range_ratio > 0.6,
                    not price_above_sma,
                    20 < rsi < 55,
                    volume > 1000,
                ]

            pattern_strength = sum(conditions)
            if pattern_strength < 3:
                continue

            metrics = compute_metrics(indicators)
            pattern_coins.append({
                "symbol": symbol,
                "price": round(close_price, 6),
                "current_change": round(current_change, 3),
                "candle_body_ratio": round(body_to_range_ratio, 3),
                "pattern_strength": pattern_strength,
                "volume": volume,
                "bollinger_rating": metrics.get("rating", 0) if metrics else 0,
                "rsi": round(rsi, 2),
                "momentum_signals": {
                    "above_sma20": price_above_sma,
                    "above_ema50": price_above_ema,
                },
            })
        except Exception:
            continue

    if pattern_type == "bullish":
        pattern_coins.sort(key=lambda x: (x["pattern_strength"], x["current_change"]), reverse=True)
    else:
        pattern_coins.sort(key=lambda x: (x["pattern_strength"], -x["current_change"]), reverse=True)

    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "pattern_type": pattern_type,
        "candle_count": candle_count,
        "min_growth": min_growth,
        "total_found": len(pattern_coins),
        "data": pattern_coins[:limit],
    }


@mcp.tool()
def advanced_candle_pattern(
    exchange: str = "KUCOIN",
    base_timeframe: str = "15m",
    pattern_length: int = 3,
    min_size_increase: float = 10.0,
    limit: int = 15,
) -> dict:
    """Advanced candle pattern analysis using multi-timeframe data.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        base_timeframe: Base timeframe for analysis (5m, 15m, 1h, 4h)
        pattern_length: Number of consecutive periods to analyze (2-4)
        min_size_increase: Minimum percentage increase in candle size
        limit: Maximum number of results to return
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return {"error": err}

    exchange = sanitize_exchange(exchange, "KUCOIN")
    base_timeframe = sanitize_timeframe(base_timeframe, "15m")
    pattern_length = max(2, min(4, pattern_length))
    min_size_increase = max(5.0, min(50.0, min_size_increase))
    limit = max(1, min(30, limit))

    symbols = load_symbols(exchange)
    if not symbols:
        return {"error": f"No symbols found for exchange: {exchange}"}

    symbols = symbols[: min(limit * 2, 100)]

    # Try multi-timeframe via screener first
    if TRADINGVIEW_SCREENER_AVAILABLE:
        results = _fetch_multi_timeframe_patterns(
            exchange, symbols, base_timeframe, pattern_length, min_size_increase,
        )
        if results:
            return {
                "exchange": exchange,
                "base_timeframe": base_timeframe,
                "pattern_length": pattern_length,
                "min_size_increase": min_size_increase,
                "method": "multi-timeframe",
                "total_found": len(results),
                "data": results[:limit],
            }

    # Fallback: single timeframe via tradingview_ta
    try:
        analysis = _fetch_ta_batch(exchange, symbols, base_timeframe)
    except ConnectionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Advanced pattern analysis failed: {e}"}

    pattern_results: List[dict] = []
    for symbol, data in analysis.items():
        if data is None:
            continue
        try:
            indicators = data.indicators
            pattern_score = _calculate_candle_pattern_score(indicators, pattern_length, min_size_increase)
            if not pattern_score["detected"]:
                continue

            metrics = compute_metrics(indicators)
            pattern_results.append({
                "symbol": symbol,
                "pattern_score": pattern_score["score"],
                "pattern_details": pattern_score["details"],
                "current_price": pattern_score["price"],
                "total_change": pattern_score["total_change"],
                "volume": indicators.get("volume", 0),
                "bollinger_rating": metrics.get("rating", 0) if metrics else 0,
                "technical_strength": {
                    "rsi": round(indicators.get("RSI", 50), 2),
                    "momentum": "Strong" if abs(pattern_score["total_change"]) > min_size_increase else "Moderate",
                    "volume_trend": "High" if indicators.get("volume", 0) > 10000 else "Low",
                },
            })
        except Exception:
            continue

    pattern_results.sort(key=lambda x: (x["pattern_score"], abs(x["total_change"])), reverse=True)

    return {
        "exchange": exchange,
        "base_timeframe": base_timeframe,
        "pattern_length": pattern_length,
        "min_size_increase": min_size_increase,
        "method": "enhanced-single-timeframe",
        "total_found": len(pattern_results),
        "data": pattern_results[:limit],
    }


@mcp.tool()
def volume_breakout_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    volume_multiplier: float = 2.0,
    price_change_min: float = 3.0,
    limit: int = 25,
    compact: bool = True,
) -> list[dict]:
    """Detect coins with volume breakout + price breakout.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        volume_multiplier: How many times the volume should be above normal (default 2.0)
        price_change_min: Minimum price change percentage (default 3.0)
        limit: Number of rows to return (max 50)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    volume_multiplier = max(1.5, min(10.0, volume_multiplier))
    price_change_min = max(1.0, min(20.0, price_change_min))
    limit = max(1, min(limit, 50))

    symbols = load_symbols(exchange)
    if not symbols:
        return []

    volume_breakouts: List[dict] = []
    batch_size = 100
    max_symbols = min(len(symbols), 500)

    for i in range(0, max_symbols, batch_size):
        batch = symbols[i : i + batch_size]
        try:
            analysis = _fetch_ta_batch(exchange, batch, timeframe)
        except ConnectionError as e:
            return [{"error": str(e)}]
        except Exception:
            continue

        for symbol, data in analysis.items():
            if not data or not hasattr(data, "indicators"):
                continue
            try:
                indicators = data.indicators
                volume = indicators.get("volume", 0)
                close = indicators.get("close", 0)
                open_price = indicators.get("open", 0)
                sma20_volume = indicators.get("volume.SMA20", 0)

                if not all([volume, close, open_price]) or volume <= 0:
                    continue

                price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0

                if sma20_volume and sma20_volume > 0:
                    volume_ratio = volume / sma20_volume
                else:
                    volume_ratio = 2.0  # conservative default

                if abs(price_change) >= price_change_min and volume_ratio >= volume_multiplier:
                    rsi = indicators.get("RSI", 50)
                    bb_upper = indicators.get("BB.upper", 0)
                    bb_lower = indicators.get("BB.lower", 0)

                    volume_breakouts.append({
                        "symbol": symbol,
                        "changePercent": price_change,
                        "volume_ratio": round(volume_ratio, 2),
                        "volume_strength": round(min(10, volume_ratio), 1),
                        "current_volume": volume,
                        "breakout_type": "bullish" if price_change > 0 else "bearish",
                        "indicators": {
                            "close": close,
                            "RSI": rsi,
                            "BB_upper": bb_upper,
                            "BB_lower": bb_lower,
                            "volume": volume,
                        },
                    })
            except Exception:
                continue

        if len(volume_breakouts) >= limit * 3:
            break

    volume_breakouts.sort(key=lambda x: (x["volume_strength"], abs(x["changePercent"])), reverse=True)
    results = volume_breakouts[:limit]
    return [_compact_scan_row(r) for r in results] if compact else results


@mcp.tool()
def volume_confirmation_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    compact: bool = False,
) -> dict:
    """Detailed volume confirmation analysis for a specific coin.

    Args:
        symbol: Coin symbol (e.g., BTCUSDT)
        exchange: Exchange name
        timeframe: Time frame for analysis
        compact: If True, flatten to key signals only — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return {"error": err}

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")

    symbol = symbol.upper()
    # Only append USDT for crypto exchanges where symbols typically need it
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    if screener == "crypto" and not symbol.endswith(("USDT", "USDC", "BTC", "ETH", "BUSD")):
        symbol = symbol + "USDT"

    try:
        analysis = _fetch_ta_batch(exchange, [symbol], timeframe)
    except ConnectionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

    if not analysis or symbol not in analysis:
        return {"error": f"No data found for {symbol}"}

    data = analysis[symbol]
    if not data or not hasattr(data, "indicators"):
        return {"error": f"No indicator data for {symbol}"}

    indicators = data.indicators
    volume = indicators.get("volume", 0)
    close = indicators.get("close", 0)
    open_price = indicators.get("open", 0)
    high = indicators.get("high", 0)
    low = indicators.get("low", 0)

    price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
    candle_range = ((high - low) / low) * 100 if low > 0 else 0

    sma20_volume = indicators.get("volume.SMA20", 0)
    volume_ratio = volume / sma20_volume if sma20_volume > 0 else 1

    rsi = indicators.get("RSI", 50)
    bb_upper = indicators.get("BB.upper", 0)
    bb_lower = indicators.get("BB.lower", 0)

    signals: List[str] = []

    if volume_ratio >= 2.0 and abs(price_change) >= 3.0:
        signals.append(f"STRONG BREAKOUT: {volume_ratio:.1f}x volume + {price_change:.1f}% price")
    if volume_ratio >= 1.5 and abs(price_change) < 1.0:
        signals.append(f"VOLUME DIVERGENCE: High volume ({volume_ratio:.1f}x) but low price movement")
    if abs(price_change) >= 2.0 and volume_ratio < 0.8:
        signals.append(f"WEAK SIGNAL: Price moved but volume is low ({volume_ratio:.1f}x)")
    if close > bb_upper and volume_ratio >= 1.5:
        signals.append("BB BREAKOUT CONFIRMED: Upper band breakout + volume confirmation")
    elif close < bb_lower and volume_ratio >= 1.5:
        signals.append("BB SELL CONFIRMED: Lower band breakout + volume confirmation")
    if rsi > 70 and volume_ratio >= 2.0:
        signals.append(f"OVERBOUGHT + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")
    elif rsi < 30 and volume_ratio >= 2.0:
        signals.append(f"OVERSOLD + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")

    if volume_ratio >= 3.0:
        volume_strength = "VERY STRONG"
    elif volume_ratio >= 2.0:
        volume_strength = "STRONG"
    elif volume_ratio >= 1.5:
        volume_strength = "MEDIUM"
    elif volume_ratio >= 1.0:
        volume_strength = "NORMAL"
    else:
        volume_strength = "WEAK"

    result = {
        "symbol": symbol,
        "price_data": {
            "close": close,
            "change_percent": round(price_change, 2),
            "candle_range_percent": round(candle_range, 2),
        },
        "volume_analysis": {
            "current_volume": volume,
            "volume_ratio": round(volume_ratio, 2),
            "volume_strength": volume_strength,
            "average_volume": sma20_volume,
        },
        "technical_indicators": {
            "RSI": round(rsi, 1),
            "BB_position": "ABOVE" if close > bb_upper else "BELOW" if close < bb_lower else "WITHIN",
            "BB_upper": bb_upper,
            "BB_lower": bb_lower,
        },
        "signals": signals,
    }
    return _compact_vol_confirmation(result) if compact else result


@mcp.tool()
def smart_volume_scanner(
    exchange: str = "KUCOIN",
    min_volume_ratio: float = 2.0,
    min_price_change: float = 2.0,
    rsi_range: str = "any",
    limit: int = 20,
    compact: bool = True,
) -> list[dict]:
    """Smart volume + technical analysis combination scanner.

    Args:
        exchange: Exchange name
        min_volume_ratio: Minimum volume multiplier (default 2.0)
        min_price_change: Minimum price change percentage (default 2.0)
        rsi_range: "oversold" (<30), "overbought" (>70), "neutral" (30-70), "any"
        limit: Number of results (max 30)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    min_volume_ratio = max(1.2, min(10.0, min_volume_ratio))
    min_price_change = max(0.5, min(20.0, min_price_change))
    limit = max(1, min(limit, 30))

    breakouts = volume_breakout_scanner(
        exchange=exchange,
        volume_multiplier=min_volume_ratio,
        price_change_min=min_price_change,
        limit=limit * 2,
    )

    if not breakouts:
        return []
    if isinstance(breakouts[0], dict) and "error" in breakouts[0]:
        return breakouts

    filtered: List[dict] = []
    for coin in breakouts:
        rsi = coin.get("indicators", {}).get("RSI", 50)

        if rsi_range == "oversold" and rsi >= 30:
            continue
        elif rsi_range == "overbought" and rsi <= 70:
            continue
        elif rsi_range == "neutral" and (rsi <= 30 or rsi >= 70):
            continue

        recommendation = ""
        if coin["changePercent"] > 0 and coin["volume_ratio"] >= 2.0:
            recommendation = "STRONG BUY" if rsi < 70 else "OVERBOUGHT - CAUTION"
        elif coin["changePercent"] < 0 and coin["volume_ratio"] >= 2.0:
            recommendation = "STRONG SELL" if rsi > 30 else "OVERSOLD - OPPORTUNITY?"

        coin["trading_recommendation"] = recommendation
        filtered.append(coin)

    results = filtered[:limit]
    return [_compact_scan_row(r) for r in results] if compact else results


@mcp.tool()
def rsi_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "1h",
    condition: str = "oversold",
    rsi_threshold: Optional[float] = None,
    limit: int = 20,
    compact: bool = True,
) -> list[dict]:
    """Scan for coins matching a specific RSI condition.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        condition: "oversold" (RSI < 30), "overbought" (RSI > 70), or "custom" (use rsi_threshold)
        rsi_threshold: Custom RSI threshold — used only when condition="custom".
        limit: Number of rows to return (max 50)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "1h")
    limit = max(1, min(limit, 50))

    symbols = load_symbols(exchange)
    if not symbols:
        return [{"error": f"No symbols found for exchange: {exchange}"}]

    batch_size = 200
    max_symbols = min(len(symbols), 1000)
    matched: List[dict] = []

    for i in range(0, max_symbols, batch_size):
        if len(matched) >= limit * 3:
            break
        batch = symbols[i : i + batch_size]
        try:
            analysis = _fetch_ta_batch(exchange, batch, timeframe)
        except ConnectionError as e:
            return [{"error": str(e)}]
        except Exception:
            continue

        for key, value in analysis.items():
            if value is None:
                continue
            try:
                indicators = value.indicators
                rsi = indicators.get("RSI")
                if rsi is None:
                    continue

                if condition == "oversold" and rsi >= 30:
                    continue
                elif condition == "overbought" and rsi <= 70:
                    continue
                elif condition == "custom" and rsi_threshold is not None:
                    if rsi_threshold < 50:
                        if rsi >= rsi_threshold:
                            continue
                    else:
                        if rsi <= rsi_threshold:
                            continue

                metrics = compute_metrics(indicators)
                if not metrics:
                    continue

                matched.append({
                    "symbol": key,
                    "changePercent": metrics["change"],
                    "rsi": round(rsi, 2),
                    "rsi_signal": "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral",
                    "indicators": _make_indicators(indicators),
                })
            except Exception:
                continue

    if condition == "overbought":
        matched.sort(key=lambda x: x["rsi"], reverse=True)
    else:
        matched.sort(key=lambda x: x["rsi"])

    results = matched[:limit]
    return [_compact_scan_row(r) for r in results] if compact else results


@mcp.tool()
def trend_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    min_adx: float = 25.0,
    direction: str = "any",
    limit: int = 20,
    compact: bool = True,
) -> list[dict]:
    """Scan for strongly trending coins using ADX (Average Directional Index).

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        min_adx: Minimum ADX value to qualify as trending (default 25.0; >40 = very strong)
        direction: "bullish" (price above EMA50), "bearish" (below EMA50), or "any"
        limit: Number of rows to return (max 50)
        compact: If True (default), return only regime-critical fields — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return [{"error": err}]

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    min_adx = max(10.0, min(60.0, min_adx))
    limit = max(1, min(limit, 50))

    symbols = load_symbols(exchange)
    if not symbols:
        return [{"error": f"No symbols found for exchange: {exchange}"}]

    batch_size = 200
    max_symbols = min(len(symbols), 1000)
    matched: List[dict] = []

    for i in range(0, max_symbols, batch_size):
        if len(matched) >= limit * 3:
            break
        batch = symbols[i : i + batch_size]
        try:
            analysis = _fetch_ta_batch(exchange, batch, timeframe)
        except ConnectionError as e:
            return [{"error": str(e)}]
        except Exception:
            continue

        for key, value in analysis.items():
            if value is None:
                continue
            try:
                indicators = value.indicators
                adx = indicators.get("ADX")
                if adx is None or adx < min_adx:
                    continue

                close = indicators.get("close")
                ema50 = indicators.get("EMA50")
                if close is None or ema50 is None:
                    continue

                trend_dir = "bullish" if close > ema50 else "bearish"
                if direction != "any" and trend_dir != direction:
                    continue

                metrics = compute_metrics(indicators)
                if not metrics:
                    continue

                matched.append({
                    "symbol": key,
                    "changePercent": metrics["change"],
                    "adx": round(adx, 2),
                    "trend_direction": trend_dir,
                    "trend_strength": "Very Strong" if adx >= 40 else "Strong",
                    "indicators": _make_indicators(indicators),
                })
            except Exception:
                continue

    matched.sort(key=lambda x: x["adx"], reverse=True)
    results = matched[:limit]
    return [_compact_scan_row(r) for r in results] if compact else results


@mcp.tool()
def multi_timeframe_summary(
    symbol: str,
    exchange: str = "KUCOIN",
    compact: bool = False,
) -> dict:
    """Get a concise technical summary for a symbol across 4 timeframes (15m, 1h, 4h, 1D).

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSDT", "AAPL")
        exchange: Exchange name (KUCOIN, BINANCE, BYBIT, NASDAQ, etc.)
        compact: If True, return a flat minimal dict per timeframe — ~70% fewer tokens.
    """
    err = _require_ta() or _check_connectivity()
    if err:
        return {"error": err}

    exchange = sanitize_exchange(exchange, "KUCOIN")

    if ":" not in symbol:
        full_symbol = f"{exchange.upper()}:{symbol.upper()}"
    else:
        full_symbol = symbol.upper()

    timeframes = ["15m", "1h", "4h", "1D"]
    summary: dict = {"symbol": full_symbol, "exchange": exchange, "timeframes": {}}

    for tf in timeframes:
        try:
            analysis = _fetch_ta_batch(exchange, [full_symbol], tf)
            if full_symbol not in analysis or analysis[full_symbol] is None:
                summary["timeframes"][tf] = {"error": "no data"}
                continue

            indicators = analysis[full_symbol].indicators
            metrics = compute_metrics(indicators)
            if not metrics:
                summary["timeframes"][tf] = {"error": "metrics unavailable"}
                continue

            rsi = indicators.get("RSI", 0)
            adx = indicators.get("ADX", 0)
            macd = indicators.get("MACD.macd", 0)
            macd_signal = indicators.get("MACD.signal", 0)
            close = indicators.get("close") or metrics.get("price", 0)
            ema50 = indicators.get("EMA50", 0)

            tf_data = {
                "change_percent": metrics["change"],
                "rsi": round(rsi, 2),
                "rsi_signal": "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                "bb_rating": metrics["rating"],
                "bb_signal": metrics["signal"],
                "bbw": metrics["bbw"],
                "adx": round(adx, 2),
                "trend_strength": "Strong" if adx > 25 else "Weak",
                "trend_direction": "Bullish" if close > ema50 else "Bearish",
                "macd_divergence": round(macd - macd_signal, 6),
            }

            if compact:
                summary["timeframes"][tf] = {
                    "chg%": tf_data["change_percent"],
                    "rsi": tf_data["rsi"],
                    "rsi_sig": tf_data["rsi_signal"],
                    "bb": tf_data["bb_signal"],
                    "adx": tf_data["adx"],
                    "trend": tf_data["trend_direction"],
                }
            else:
                summary["timeframes"][tf] = tf_data

        except ConnectionError as e:
            return {"error": str(e)}
        except Exception as e:
            summary["timeframes"][tf] = {"error": str(e)}

    signals = [
        d.get("bb_signal", "")
        for d in summary["timeframes"].values()
        if isinstance(d, dict) and "bb_signal" in d
    ]
    bullish = sum(1 for s in signals if "Buy" in s)
    bearish = sum(1 for s in signals if "Sell" in s)
    summary["consensus"] = "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Mixed"
    summary["bullish_timeframes"] = bullish
    summary["bearish_timeframes"] = bearish

    return summary


@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """List available exchanges from coinlist directory."""
    current_dir = os.path.dirname(__file__)
    coinlist_dir = os.path.join(current_dir, "coinlist")

    if os.path.exists(coinlist_dir):
        exchanges = sorted(
            f[:-4].upper()
            for f in os.listdir(coinlist_dir)
            if f.endswith(".txt")
        )
        if exchanges:
            return f"Available exchanges: {', '.join(exchanges)}"

    return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView Screener MCP server")
    parser.add_argument(
        "transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport (default stdio)",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        import anyio
        import uvicorn
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        try:
            mcp.settings.host = args.host
            mcp.settings.port = args.port
        except Exception:
            pass

        api_key = os.environ.get("API_KEY")
        starlette_app = mcp.streamable_http_app()

        async def health(request: Request) -> Response:
            return Response("OK", status_code=200)

        if api_key:
            class ApiKeyMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    if request.url.path == "/health":
                        return await call_next(request)
                    auth = request.headers.get("Authorization", "")
                    if auth != f"Bearer {api_key}":
                        return Response("Unauthorized", status_code=401)
                    return await call_next(request)

            starlette_app = Starlette(
                routes=[Route("/health", health), Mount("/", app=starlette_app)],
                middleware=[Middleware(ApiKeyMiddleware)],
            )
        else:
            starlette_app = Starlette(
                routes=[Route("/health", health), Mount("/", app=starlette_app)],
            )

        async def _keep_alive():
            import httpx

            public_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
            ping_url = f"{public_url}/health" if public_url else f"http://127.0.0.1:{args.port}/health"
            await anyio.sleep(60)
            while True:
                try:
                    async with httpx.AsyncClient() as client:
                        await client.get(ping_url, timeout=10)
                except Exception:
                    pass
                await anyio.sleep(840)

        async def _serve():
            config = uvicorn.Config(
                starlette_app,
                host=args.host,
                port=args.port,
                log_level="info",
            )
            server = uvicorn.Server(config)

            async def _run_server():
                try:
                    await server.serve()
                finally:
                    tg.cancel_scope.cancel()

            async with anyio.create_task_group() as tg:
                tg.start_soon(_run_server)
                tg.start_soon(_keep_alive)

        anyio.run(_serve)


if __name__ == "__main__":
    main()
