import math


class OneEuro:
    """One-Euro filter: adaptive low-pass. Heavy smoothing at low speed
    (kills jitter on small targets), light smoothing at high speed (no lag)."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._x = t, x
            return x
        dt = max(t - self._t, 1e-4)
        self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._t = None


class OneEuro2D:
    def __init__(self, **kw):
        self.fx = OneEuro(**kw)
        self.fy = OneEuro(**kw)

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
