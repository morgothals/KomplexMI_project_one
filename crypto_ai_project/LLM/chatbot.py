"""Simple retrieval layer for a crypto assistant ("CryptoGPT")."""
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime, timezone
import json

from . import config
from .llm_client import LLMClient, LLMNotConfigured, render_messages
from . import data_access


DISCLAIMER = "A válasz nem minősül befektetési tanácsadásnak; kizárólag információs célra szolgál."


def _context_block() -> Dict[str, Any]:
    market = data_access.load_market_context()
    sentiment = data_access.load_sentiment_snapshot()
    labels, prices = data_access.load_long_curve()
    news = data_access.load_recent_news(max_items=5)

    daily = data_access.load_last_day_bundle()
    training_last = data_access.load_training_features_last_row()
    longterm_last_year = data_access.load_longterm_features_last_year()

    now_utc = datetime.now(timezone.utc).isoformat()
    return {
        "server_time_utc": now_utc,
        "market": market,
        "sentiment": sentiment,
        "long_curve": {"labels": labels, "prices": prices},
        "news": news,
        "last_day": daily,
        "training_features_1h_last_row": training_last,
        "longterm_features_15d_last_year": longterm_last_year,
    }


def _truncate_text(text: str, *, max_len: int = 12000) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...[truncated]"


def _context_json(ctx: Dict[str, Any]) -> str:
    try:
        return _truncate_text(json.dumps(ctx, ensure_ascii=False, indent=2))
    except Exception:
        return _truncate_text(str(ctx))


def _context_text(ctx: Dict[str, Any]) -> str:
    parts = [
        f"Szerver idő (UTC): {ctx.get('server_time_utc')}",
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
    parts.append("\n--- Részletes adatok (JSON) ---")
    parts.append(_context_json(ctx))
    return "\n".join(parts)


def crypto_chat(question: str, *, allow_llm: bool = False) -> Dict[str, Any]:
    ctx = _context_block()
    context_text = _context_text(ctx)

    # Szabály: külső LLM hívás csak explicit felhasználói gombnyomásra.
    if not allow_llm:
        heuristic_answer = (
            f"LLM hívás ki van kapcsolva automatikus módban. (Nyomd meg a 'Küldés' gombot az asszisztensnél.)\n"
            f"{context_text}\n{DISCLAIMER}"
        )
        return {"answer": heuristic_answer, "used_llm": False, "context": ctx}

    if not config.is_configured():
        heuristic_answer = (
            f"LLM_API_KEY nincs beállítva, így összefoglaltuk a főbb adatokat.\n{context_text}\n{DISCLAIMER}"
        )
        return {"answer": heuristic_answer, "used_llm": False, "context": ctx}

    system = (
        "Te egy technikai, de csevegős kripto asszisztens vagy. Válaszolj magyarul. "
        "Mindig használd a megadott kontextusban lévő adatokat (számok/idősorok), és hivatkozz rájuk konkrétan. "
        "Ha a kérdés nem pénzügyi (pl. mennyi az idő), akkor is válaszolj a kontextus alapján (pl. szerver idő UTC) és röviden utalj a friss adatokra. "
        "Ne találj ki adatokat a kontextuson kívül. "
        "A végén mindig add hozzá a figyelmeztetést: '" + DISCLAIMER + "'"
    )
    user = (
        f"Kérdés: {question}\n" +
        f"Kontekstus:\n{context_text}\n" +
        "Stílus: technikai, de csevegős.\n"
        "Használj bulletpontokat ha segít, de maradj tömör.\n"
        "Ha kérik, adhatsz 'BUY/HOLD/SELL' jellegű irányt, de fogalmazd meg információs jelleggel (nem tanács)."
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
