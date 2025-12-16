"""LLM-aided news analysis that adjusts the base forecast."""
from __future__ import annotations

from typing import Any, Dict, List
import math

from . import config
from .llm_client import LLMClient, LLMNotConfigured, render_messages
from . import data_access


def _truncate_text(text: str, *, max_len: int = 2000) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...[truncated]"


def _heuristic_signal(base_signal: str, rel_change: float | None, sentiment: float | None) -> str:
    # Light-touch override using sentiment and the base delta.
    signal = base_signal
    if sentiment is None:
        return signal
    if sentiment >= 0.2 and base_signal == "HOLD":
        signal = "BUY"
    if sentiment <= -0.2 and base_signal == "HOLD":
        signal = "SELL"
    if sentiment <= -0.3 and base_signal == "BUY":
        signal = "HOLD"
    if sentiment >= 0.3 and base_signal == "SELL":
        signal = "HOLD"
    if rel_change is not None and abs(rel_change) < 0.002:
        signal = "HOLD"
    return signal


def _heuristic_risk(volatility: float | None, sentiment: float | None) -> float:
    vol_component = min(max((volatility or 0) / 100, 0.0), 1.0)
    sent_component = 0.5 if sentiment is None else 0.5 + (-sentiment * 0.3)
    sent_component = min(max(sent_component, 0.0), 1.0)
    return round(0.6 * vol_component + 0.4 * sent_component, 3)


def _build_prompt(base_advice: Dict[str, Any], news: List[Dict[str, Any]], market_ctx: Dict[str, Any], sentiment_ctx: Dict[str, Any], long_curve: Dict[str, Any]) -> str:
    headlines = "\n".join([
        f"- {item.get('headline','').strip()}: {item.get('summary','').strip()}" for item in news if item.get("headline")
    ])
    lc_label = long_curve.get("labels", [None])[-1]
    lc_price = long_curve.get("pred_price", [None])[-1]
    return (
        "You are an assistant that adjusts a crypto short-term forecast using news and sentiment. "
        "Respond with a compact JSON object with keys adjusted_signal, adjusted_change_pct, risk_score, explanation.\n"
        f"Base signal: {base_advice.get('signal')} with rel_change_pred={base_advice.get('rel_change_pred')} and next_price_pred={base_advice.get('next_price_pred')}\n"
        f"Latest sentiment={sentiment_ctx.get('latest')} avg_sentiment={sentiment_ctx.get('avg')} fear_greed={sentiment_ctx.get('fear_greed')}\n"
        f"Market last_close={market_ctx.get('last_close')} return_pct_{market_ctx.get('return_%')} vol_pct={market_ctx.get('volatility_%')}\n"
        f"Long_curve_anchor_year={lc_label} price={lc_price}\n"
        f"Recent news:\n{headlines or 'n/a'}"
    )


def build_adjusted_forecast(
    base_advice: Dict[str, Any],
    sentiment_series: Dict[str, Any],
    long_curve: Dict[str, Any],
    *,
    force_llm: bool = False,
) -> Dict[str, Any]:
    news_rows = data_access.load_recent_news(max_items=8)
    sentiment_ctx = data_access.load_sentiment_snapshot()
    market_ctx = data_access.load_market_context()

    base_signal = str(base_advice.get("signal", "HOLD"))
    rel_change = base_advice.get("rel_change_pred")
    # újabb advisor mező: pred_change_pct (%), ebből vissza tudjuk vezetni a rel_change-t
    if rel_change is None and base_advice.get("pred_change_pct") is not None:
        try:
            rel_change = float(base_advice.get("pred_change_pct")) / 100.0
        except Exception:
            rel_change = None
    rel_change = float(rel_change) if rel_change is not None and not math.isnan(rel_change) else None

    heuristic_signal = _heuristic_signal(base_signal, rel_change, sentiment_ctx.get("latest"))
    risk_score = _heuristic_risk(market_ctx.get("volatility_%"), sentiment_ctx.get("latest"))

    result = {
        "base_signal": base_signal,
        "base_rel_change_pred": rel_change,
        "adjusted_signal": heuristic_signal,
        "adjusted_change_pct": rel_change * 100 if rel_change is not None else None,
        # A specifikáció szerinti mező (összefoglaló)
        "adjusted_prediction": {
            "signal": heuristic_signal,
            "change_pct": rel_change * 100 if rel_change is not None else None,
        },
        "risk_score": risk_score,
        "explanation": "Heurisztikus módosítás a sentiment és volatilitás alapján.",
        "used_llm": False,
        "news_headlines": news_rows,
        "market_context": market_ctx,
        "sentiment_snapshot": sentiment_ctx,
    }

    # Biztonsági/UX szabály: LLM hívás kizárólag explicit felhasználói gombnyomásra.
    # (Dashboard: force_llm=1 query param.)
    if not force_llm:
        result["note"] = "LLM hívás kikapcsolva automatikus frissítésnél; nyomd meg az 'LLM frissítés' gombot."
        return result

    if not config.is_configured():
        result["note"] = "LLM_API_KEY nincs beállítva, heuristikus módot használtunk."
        if force_llm:
            result["note"] = "LLM kényszerítve lett, de LLM_API_KEY hiányzik."
        return result

    prompt = _build_prompt(base_advice, news_rows, market_ctx, sentiment_ctx, long_curve)
    messages = render_messages(
        "Legyél tömör. Előrejelzési tanácsadó vagy és az alábbi adatokból, adj JSON formátumot vissza ilyen mezőkkel: adjusted_signal (BUY/HOLD/SELL), adjusted_change_pct (float), risk_score (0-1), explanation (1-2 mondat).",
        prompt,
    )

    client = LLMClient()
    try:
        content = client.chat(messages)
        parsed = _safe_parse_json(content)
        if parsed:
            result.update(parsed)
            result["used_llm"] = True
            result["explanation"] = parsed.get("explanation") or result["explanation"]
            # ha az LLM adott adjusted_change_pct / adjusted_signal mezőt, képezzünk adjusted_prediction-t is
            if "adjusted_prediction" not in result:
                result["adjusted_prediction"] = {
                    "signal": result.get("adjusted_signal"),
                    "change_pct": result.get("adjusted_change_pct"),
                }
        else:
            raw = _truncate_text(content)
            result["llm_raw_response"] = raw
            result["note"] = "LLM válasz nem volt JSON, ezért a heuristikus eredmény maradt."
            if force_llm:
                result["note"] = "LLM kényszerítve lett, de a válasz nem volt parse-olható JSON; a nyers válasz ki van írva."
    except LLMNotConfigured:
        result["note"] = "LLM_API_KEY hiányzik, heuristika maradt."
    except Exception as exc:  # noqa: BLE001
        result["note"] = f"LLM hívás hiba: {exc}"[:240]
    return result


def _safe_parse_json(text: str) -> Dict[str, Any] | None:
    import json

    if not text:
        return None

    candidate = str(text).strip()

    # 1) Markdown code fence eltávolítása (```json ... ```)
    if candidate.startswith("```"):
        # levágjuk az első sort (``` vagy ```json)
        parts = candidate.splitlines()
        if parts:
            parts = parts[1:]
        candidate = "\n".join(parts).strip()
        if candidate.endswith("```"):
            candidate = candidate[: -3].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 2) Ha extra szöveg van körülötte, próbáljuk kivágni az első { ... } blokkot
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
        return None
