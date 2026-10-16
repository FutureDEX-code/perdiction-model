"""Hong Kong daily-maximum model aligned to the Polymarket settlement station."""

from hk_temp_model.model import predict
from hk_temp_model.settlement import SETTLEMENT_STATION, bracket_for

__all__ = ["SETTLEMENT_STATION", "bracket_for", "predict"]
