def spot_scenerio(base_spot, pct_range=0.20, step=0.05):
    """
    Spot price scenarios.
    Example: -20%, -15%, ..., +20%
    """
    scenarios = []
    multipliers = [
        1 + x * step
        for x in range(int(-pct_range / step), int(pct_range / step) + 1)
    ]

    for m in multipliers:
        scenarios.append({"S0": base_spot * m})

    return scenarios


def strike_scenerio(base_strike, pct_range=0.20, step=0.05):
    """
    Strike price scenarios.
    Example: -20%, -15%, ..., +20%
    """
    scenarios = []
    multipliers = [
        1 + x * step
        for x in range(int(-pct_range / step), int(pct_range / step) + 1)
    ]

    for m in multipliers:
        scenarios.append({"K": base_strike * m})

    return scenarios


def vol_scenerio(base_sigma, pct_range=0.50, step=0.10):
    """
    Volatility scenarios.
    Example: -50%, -40%, ..., +50%
    Sigma is floored to a small positive number.
    """
    scenarios = []
    multipliers = [
        1 + x * step
        for x in range(int(-pct_range / step), int(pct_range / step) + 1)
    ]

    for m in multipliers:
        sigma = max(base_sigma * m, 0.0001)
        scenarios.append({"sigma": sigma})

    return scenarios


def rate_scenerio(base_rate, pct_range=0.50, step=0.10):
    """
    Interest rate scenarios.
    Example: -50%, -40%, ..., +50% around base_rate
    Rate is floored to zero.
    """
    scenarios = []
    multipliers = [
        1 + x * step
        for x in range(int(-pct_range / step), int(pct_range / step) + 1)
    ]

    for m in multipliers:
        r = max(base_rate * m, 0.0)
        scenarios.append({"r": round(r, 6)})

    return scenarios



def maturity_scenerio(base_T, pct_range=0.50, step=0.25):
    """
    Time-to-maturity scenarios.
    Example: -50%, -25%, ..., +50% around base_T
    T is floored to a small positive number.
    """
    scenarios = []
    multipliers = [
        1 + x * step
        for x in range(int(-pct_range / step), int(pct_range / step) + 1)
    ]

    for m in multipliers:
        T = max(base_T * m, 1e-4)
        scenarios.append({"T": round(T, 6)})

    return scenarios

