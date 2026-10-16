"""Paris. Before the local peak, do not buy a degree below the forecast.

Those paper tickets lost, and none of them won. The forecast degree itself
stays eligible. After the peak the shared rule applies.
"""


def allow_yes(row: dict, *, hour: float, peak: float, forecast: float | None, span, **kwargs) -> str:
    if forecast is not None and hour < peak and span is not None and span[1] <= forecast:
        return "高峰前低于预报"
    return ""
