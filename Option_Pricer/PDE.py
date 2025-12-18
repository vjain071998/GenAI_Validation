#! D:\GenAI_Validation\myenv\Scripts\python.exe

import numpy as np
from dataclasses import dataclass


@dataclass
class PricerResult:
    price: float
    grid_S: np.ndarray
    grid_V: np.ndarray
    S0_index: int  # index on grid closest to S0
    info: dict     # additional metadata (N_S, N_t, dt, dS, ...)


class AmericanPDEPricer:
    """
    American option pricer using finite differences (theta-scheme).
    - Theta=0.5 -> Crank-Nicolson (mix of implicit & explicit)
    - Early exercise enforced by V = max(V, payoff) at each time level

    Methods:
      - price(): returns PricerResult (price and grid)
      - greeks(): returns dict of Delta, Gamma, Theta, Vega (via bumping)
      - bumped_values(): compute price(s) for bumped parameters
    """

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        option_type: str = "call",
        S_max_multiplier: float = 3.0,
        N_S: int = 400,
        N_t: int = 400,
        theta: float = 0.5,
    ):
        """
        Params:
          S0: spot
          K: strike
          r: risk-free continuously compounded rate
          q: continuous dividend yield (0 if none)
          sigma: volatility (annual)
          T: time to maturity (in years)
          option_type: 'put' or 'call'
          S_max_multiplier: S_grid_max = S_max_multiplier * max(S0, K)
          N_S: number of spatial grid points (S dimension)
          N_t: number of time steps
          theta: 0.5 => Crank-Nicolson, 1.0 => implicit, 0.0 => explicit
        """
        assert option_type in ("put", "call")
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)
        self.T = float(T)
        self.option_type = option_type
        self.S_max_multiplier = float(S_max_multiplier)
        self.N_S = int(N_S)
        self.N_t = int(N_t)
        self.theta = float(theta)

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "put":
            return np.maximum(self.K - S, 0.0)
        else:
            return np.maximum(S - self.K, 0.0)

    def _setup_grid(self):
        S_max = self.S_max_multiplier * max(self.S0, self.K)
        dS = S_max / self.N_S
        S_grid = np.linspace(0.0, S_max, self.N_S + 1)
        dt = self.T / self.N_t if self.N_t > 0 else 0.0
        return S_grid, dS, dt

    def price(self, return_grid: bool = False) -> PricerResult:
        """
        Solve PDE and return a PricerResult object.
        """
        S_grid, dS, dt = self._setup_grid()
        M = self.N_S
        N = self.N_t

        # payoff at maturity
        V = self._payoff(S_grid)

        # boundary conditions: for S=0 and S=S_max
        # For a put: V(S=0)=K*exp(-r*t) (but we enforce payoff projection each time)
        # For a call: V(S_max) ~ S_max - K*exp(-r*(T-t))
        # We'll use Dirichlet conditions each time step.
        # Precompute coefficients for tri-diagonal matrix
        sigma2 = self.sigma * self.sigma

        # Index of S0 in grid (closest)
        S0_idx = int(round(self.S0 / dS))
        S0_idx = min(max(S0_idx, 0), M)

        if N == 0:
            price_at_S0 = np.interp(self.S0, S_grid, V)
            return PricerResult(price_at_S0, S_grid, V, S0_idx, {"N_S": M, "N_t": N, "dS": dS, "dt": dt})

        # Prepare tri-diagonal matrix coefficients (interior nodes 1..M-1)
        # Using standard FD discretization
        i = np.arange(1, M)
        a = 0.5 * dt * ( (self.sigma**2) * (i**2) - (self.r - self.q) * i )
        b = 1.0 + dt * ( self.sigma**2 * (i**2) + self.r )
        c = -0.5 * dt * ( (self.sigma**2) * (i**2) + (self.r - self.q) * i )

        # For theta-scheme we need two matrices: LHS and RHS; but to keep memory small we compute per-step vectors.
        # For Crank-Nicolson (theta=0.5): LHS = I - theta*dt*A, RHS = I + (1-theta)*dt*A
        # Build A's tri-diagonal representation (for interior indices)
        def build_A_coeffs():
            # A is tri-diagonal with subdiagonal alpha_i, diag beta_i, superdiag gamma_i
            alpha = 0.5 * sigma2 * (i**2) - 0.5 * (self.r - self.q) * i
            beta = - (sigma2 * (i**2) + self.r)
            gamma = 0.5 * sigma2 * (i**2) + 0.5 * (self.r - self.q) * i
            return alpha, beta, gamma

        alpha, beta, gamma = build_A_coeffs()

        # Convert A into dt-scaled tri-diagonal for RHS/LHS convenience
        # LHS tri-diagonal: diag_l = 1 - theta*dt*beta ; sub_l = -theta*dt*alpha ; sup_l = -theta*dt*gamma
        # RHS tri-diagonal: diag_r = 1 + (1-theta)*dt*beta ; sub_r = (1-theta)*dt*alpha ; sup_r = (1-theta)*dt*gamma

        theta = self.theta
        sub_l = -theta * dt * alpha
        diag_l = 1.0 - theta * dt * beta
        sup_l = -theta * dt * gamma

        sub_r = (1.0 - theta) * dt * alpha
        diag_r = 1.0 + (1.0 - theta) * dt * beta
        sup_r = (1.0 - theta) * dt * gamma

        # Time-stepping backwards
        # We'll solve LHS * V_new_interior = RHS * V_old_interior + boundary_terms
        # use Thomas algorithm (tridiagonal solver)
        def thomas_solve(a_sub, a_diag, a_sup, d_rhs):
            # solve tridiagonal a_sub(1..n-1), a_diag(0..n-1), a_sup(0..n-2)
            n = len(a_diag)
            cp = np.empty(n-1)
            dp = np.empty(n)
            cp[0] = a_sup[0] / a_diag[0]
            dp[0] = d_rhs[0] / a_diag[0]
            for j in range(1, n-1):
                denom = a_diag[j] - a_sub[j-1] * cp[j-1]
                cp[j] = a_sup[j] / denom
                dp[j] = (d_rhs[j] - a_sub[j-1] * dp[j-1]) / denom
            denom = a_diag[n-1] - a_sub[n-2] * cp[n-2]
            dp[n-1] = (d_rhs[n-1] - a_sub[n-2] * dp[n-2]) / denom
            x = np.empty(n)
            x[-1] = dp[-1]
            for j in range(n-2, -1, -1):
                x[j] = dp[j] - cp[j] * x[j+1]
            return x

        # interior indices correspond to grid points 1..M-1
        for n_step in range(N):
            t = self.T - n_step * dt  # current time level (backward)
            # Boundary values at this time (t-dt for LHS formation). Using direct Dirichlet approximations:
            # S=0
            if self.option_type == "put":
                V_0 = self.K * np.exp(-self.r * (t - dt))  # approximate
            else:
                V_0 = 0.0
            # S = S_max
            S_max = S_grid[-1]
            if self.option_type == "call":
                V_max = S_max - self.K * np.exp(-self.r * (t - dt))
            else:
                V_max = 0.0

            # RHS vector for interior nodes
            V_int_old = V[1:M]  # length M-1
            rhs = diag_r * V_int_old.copy()
            # add sub_r * V_{i-1} and sup_r * V_{i+1}
            rhs[1:] += sub_r[1:] * V[1:M-1]
            rhs[:-1] += sup_r[:-1] * V[2:M]

            # boundary contributions
            rhs[0] += sub_r[0] * V_0
            rhs[-1] += sup_r[-1] * V_max

            # Solve LHS system
            V_int_new = thomas_solve(sub_l, diag_l, sup_l, rhs)

            # construct full V_new
            V_new = np.empty_like(V)
            V_new[0] = V_0
            V_new[1:M] = V_int_new
            V_new[M] = V_max

            # Early exercise for American: enforce V >= payoff
            payoff_now = self._payoff(S_grid)  # payoff doesn't depend on t for American exercise
            V_new = np.maximum(V_new, payoff_now)

            V = V_new

        # price at S0 by interpolation
        price_at_S0 = np.interp(self.S0, S_grid, V)

        info = {"N_S": M, "N_t": N, "dS": dS, "dt": dt, "S_max": S_grid[-1]}
        if return_grid:
            return PricerResult(price_at_S0, S_grid, V, S0_idx, info)
        else:
            return PricerResult(price_at_S0, None, None, S0_idx, info)

    # ---------- Greeks via bumping ----------
    def greeks(self, eps: float = 1e-4) -> dict:
        """
        Compute Greeks (Delta, Gamma, Theta, Vega) using finite-difference bumping.
        eps: relative bump size for S0 and sigma; for time (theta) uses small deltaT.

        Returns: dict {'price':..., 'delta':..., 'gamma':..., 'theta':..., 'vega':...}
        """
        base = self.price()
        base_p = base.price

        # Delta: central bump on S0
        bump_S = max(self.S0 * eps, eps)
        pr_up = self._price_with_override(S0=self.S0 + bump_S)
        pr_down = self._price_with_override(S0=self.S0 - bump_S)
        delta = (pr_up - pr_down) / (2.0 * bump_S)
        gamma = (pr_up - 2.0 * base_p + pr_down) / (bump_S ** 2)

        # Theta: bump time forward (decrease T by small dt). Use absolute small dt
        bump_T = max(self.T * eps, 1e-6)
        if self.T - bump_T <= 0:
            theta_val = np.nan
        else:
            pr_theta = self._price_with_override(T=self.T - bump_T)
            theta_val = (pr_theta - base_p) / (-bump_T)  # price change per year

        # Vega: bump sigma
        bump_sigma = max(self.sigma * eps, 1e-6)
        pr_up_sigma = self._price_with_override(sigma=self.sigma + bump_sigma)
        pr_down_sigma = self._price_with_override(sigma=self.sigma - bump_sigma)
        vega = (pr_up_sigma - pr_down_sigma) / (2.0 * bump_sigma)

        return {"price": base_p, "delta": delta, "gamma": gamma, "theta": theta_val, "vega": vega}

    def bumped_values(self, bumps: dict) -> dict:
        """
        Compute prices for arbitrary parameter bumps.
        bumps: dict where keys are parameter names and values are bumped values or list of values.
               Supported parameter names: 'S0','K','r','q','sigma','T'
               If a list is provided, returns list of results for those values.

        Returns dict with same keys mapped to prices (float or list of floats).
        """
        out = {}
        for key, val in bumps.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                prices = []
                for v in val:
                    kwargs = {key: v}
                    prices.append(self._price_with_override(**kwargs))
                out[key] = prices
            else:
                out[key] = self._price_with_override(**{key: val})
        return out

    # ---------- internal helper ----------
    def _price_with_override(self, **overrides):
        """Return price after temporarily overriding some parameters."""
        saved = {}
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise KeyError(f"Unknown parameter to override: {k}")
            saved[k] = getattr(self, k)
            setattr(self, k, float(v))
        try:
            res = self.price(return_grid=False).price
        finally:
            # restore
            for k, v in saved.items():
                setattr(self, k, v)
        return res


# ---------- Example usage ----------
if __name__ == "__main__":
    # quick sanity check
    pricer = AmericanPDEPricer(
        S0=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        option_type="put",
        N_S=400,
        N_t=400,
    )
    res = pricer.price(return_grid=True)
    print("Price:", res.price)
    greeks = pricer.greeks()
    print("Greeks:", greeks)

    # bumped scan example
    bumps = {"S0": [90, 95, 100, 105, 110], "sigma": [0.15, 0.2, 0.25]}
    print("Bumped:", pricer.bumped_values(bumps))
