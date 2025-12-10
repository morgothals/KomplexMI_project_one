"""Simple retrieval layer for a crypto assistant ("CryptoGPT")."""
from __future__ import annotations

from typing import Any, Dict

from . import config
from .llm_client import LLMClient, LLMNotConfigured, render_messages
from . import data_access


DISCLAIMER = "A válasz nem minősül befektetési tanácsadásnak; kizárólag információs célra szolgál."


def _context_block() -> Dict[str, Any]:
    market = data_access.load_market_context()
    sentiment = data_access.load_sentiment_snapshot()
    labels, prices = data_access.load_long_curve()
    news = data_access.load_recent_news(max_items=5)
    return {
        "market": market,
        "sentiment": sentiment,
        "long_curve": {"labels": labels, "prices": prices},
        "news": news,
    }


def _context_text(ctx: Dict[str, Any]) -> str:
    parts = [
        f"Utolsó ár: {ctx['market'].get('last_close')}",
        f"180 napos hozam %: {ctx['market'].get('return_%')}",
        f"Évesített vol %: {ctx['market'].get('volatility_%')}",
        f"Legfrissebb news sentiment: {ctx['sentiment'].get('latest')}",
        f"Fear & Greed: {ctx['sentiment'].get('fear_greed')}",
    ]
    if ctx.get("long_curve", {}).get("labels"):
        parts.append(
            f"Hosszútávú görbe: {ctx['long_curve']['labels'][0]}-{ctx['long_curve']['labels'][-1]} első ár {ctx['long_curve']['prices'][0] if ctx['long_curve']['prices'] else None}"
        )
    if ctx.get("news"):
        top_news = " | ".join([n.get("headline", "") for n in ctx["news"] if n.get("headline")][:3])
        parts.append(f"Friss hírek: {top_news}")
    return "\n".join(parts)


def crypto_chat(question: str) -> Dict[str, Any]:
    ctx = _context_block()
    context_text = _context_text(ctx)

    if not config.is_configured():
        heuristic_answer = (
            f"LLM_API_KEY nincs beállítva, így összefoglaltuk a főbb adatokat.\n{context_text}\n{DISCLAIMER}"
        )
        return {"answer": heuristic_answer, "used_llm": False, "context": ctx}

    system = (
        "Te egy kripto asszisztens vagy. Válaszolj magyarul, legfeljebb 6 mondatban. "
        "Mindig add hozzá a figyelmeztetést: '" + DISCLAIMER + "'"
    )
    user = (
        f"Kérdés: {question}\n" +
        f"Kontekstus:\n{context_text}\n" +
        "Hivatkozz a számokra röviden, adj cselekvési irányt (buy/hold/sell) ha kérik, de ne adj pénzügyi tanácsot."
    )

    client = LLMClient()
    try:
        answer = client.chat(render_messages(system, user))
        return {"answer": answer.strip(), "used_llm": True, "context": ctx}
    except LLMNotConfigured:
        return {"answer": "LLM_API_KEY hiányzik, csak heurisztikus válasz elérhető.", "used_llm": False, "context": ctx}
    except Exception as exc:  # noqa: BLE001
        fallback = f"Nem sikerült az LLM hívás: {exc}.\n{context_text}\n{DISCLAIMER}"
        return {"answer": fallback, "used_llm": False, "context": ctx}
