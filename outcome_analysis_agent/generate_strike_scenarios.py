def strike_scenerio(base_strike, pct_range=0.20, step=0.05):
    """
    Creates multiple scenarios where only the strike price is bumped,
    while all other parameters remain the same.
    Example: -20%, -15%, ... +20%
    """
    scenarios = []
    multipliers = [1 + i for i in [x * step for x in range(int(-pct_range/step), int(pct_range/step)+1)]]

    for m in multipliers:
        scenarios.append({"S0": base_strike * m})

    return scenarios