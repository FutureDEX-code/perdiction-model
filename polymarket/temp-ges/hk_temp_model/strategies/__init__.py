"""City paper-trade rules. Shared gates live in decide.py."""

from hk_temp_model.strategies import hk, nyc, paris

_CITIES = {"hk": hk, "paris": paris, "nyc": nyc}


def allow_yes(city: str, row: dict, **kwargs) -> str:
    """Empty string means the shared Yes gates may still buy this bracket."""

    return _CITIES.get(city, hk).allow_yes(row, **kwargs)
