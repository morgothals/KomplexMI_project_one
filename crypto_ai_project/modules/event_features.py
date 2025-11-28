# modules/event_features.py
from datetime import datetime, timedelta, timezone
import pandas as pd


def _dt(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


# Események definíciója
EVENTS = [
    # Halvingek (széles hatásablakkal, pl. ±90 nap)
    {
        "name": "halving_2012",
        "category": "halving",
        "date": _dt(2012, 11, 28),
        "window_before_days": 90,
        "window_after_days": 90,
        "impact": +1,
    },
    {
        "name": "halving_2016",
        "category": "halving",
        "date": _dt(2016, 7, 9),
        "window_before_days": 90,
        "window_after_days": 90,
        "impact": +1,
    },
    {
        "name": "halving_2020",
        "category": "halving",
        "date": _dt(2020, 5, 11),
        "window_before_days": 90,
        "window_after_days": 90,
        "impact": +1,
    },
    {
        "name": "halving_2024",
        "category": "halving",
        "date": _dt(2024, 4, 20),
        "window_before_days": 90,
        "window_after_days": 90,
        "impact": +1,
    },

    # COVID crash – rövidebb, intenzív időszak (pl. ±14 nap)
    {
        "name": "covid_crash_2020",
        "category": "covid_crash",
        "date": _dt(2020, 3, 12),
        "window_before_days": 14,
        "window_after_days": 14,
        "impact": -1,
    },

    # China crackdowns
    {
        "name": "china_exchanges_shutdown_2017",
        "category": "china_crackdown",
        "date": _dt(2017, 9, 15),  # kb. 2017 szeptember közepe
        "window_before_days": 14,
        "window_after_days": 30,
        "impact": -1,
    },
    {
        "name": "china_full_ban_2021",
        "category": "china_crackdown",
        "date": _dt(2021, 9, 24),
        "window_before_days": 14,
        "window_after_days": 30,
        "impact": -1,
    },

    # ETF események
    {
        "name": "futures_etf_2021",
        "category": "etf",
        "date": _dt(2021, 10, 19),
        "window_before_days": 7,
        "window_after_days": 30,
        "impact": +1,
    },
    {
        "name": "spot_etf_approval_2024",
        "category": "etf",
        "date": _dt(2024, 1, 10),
        "window_before_days": 7,
        "window_after_days": 60,
        "impact": +1,
    },
]


def build_event_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Kap egy DatetimeIndex-et (pl. 1H gyertyák időindexe),
    és visszaad egy DataFrame-et ugyanezzel az indexszel,
    plusz néhány esemény-jellegű feature-rel:

    - halving_window
    - covid_crash_window
    - china_crackdown_window
    - etf_event_window
    - event_impact_sum (összegzett hatás)
    """
    idx = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    df_ev = pd.DataFrame(index=idx)

    # kategória-szintű maskok + összhatás
    df_ev["halving_window"] = 0
    df_ev["covid_crash_window"] = 0
    df_ev["china_crackdown_window"] = 0
    df_ev["etf_event_window"] = 0
    df_ev["event_impact_sum"] = 0.0

    for ev in EVENTS:
        date = ev["date"]
        wb = ev["window_before_days"]
        wa = ev["window_after_days"]
        impact = ev["impact"]
        category = ev["category"]

        start = date - timedelta(days=wb)
        end = date + timedelta(days=wa)

        mask = (idx >= start) & (idx <= end)

        if category == "halving":
            df_ev.loc[mask, "halving_window"] = 1
        elif category == "covid_crash":
            df_ev.loc[mask, "covid_crash_window"] = 1
        elif category == "china_crackdown":
            df_ev.loc[mask, "china_crackdown_window"] = 1
        elif category == "etf":
            df_ev.loc[mask, "etf_event_window"] = 1

        df_ev.loc[mask, "event_impact_sum"] += impact

    return df_ev
