# -*- coding: utf-8 -*-
"""
Physics core — Wu & Wang (2006) single-particle settling velocity
+ Richardson-Zaki concentration correction with Re-dependent index n.
No effective diameter (Deff); single particle is evaluated at D50.
"""
import numpy as np

G = 9.81


def wu_wang_velocity(d_mm, csf, rho_s, rho_f, nu):
    """Wu & Wang (2006) single-particle settling velocity [cm/s]."""
    d = d_mm / 1000.0
    R = (rho_s - rho_f) / rho_f
    Dstar = d * (G * R / nu ** 2) ** (1.0 / 3.0)
    M = 53.5 * np.exp(-0.65 * csf)
    N = 5.65 * np.exp(-2.5 * csf)
    n = 0.7 + 0.9 * csf
    inner = 0.25 + (4.0 * N / (3.0 * M ** 2) * Dstar ** 3) ** (1.0 / n)
    w = (M * nu) / (N * d) * (np.sqrt(inner) - 0.5) ** n
    return w * 100.0  # cm/s


def rz_index(Re):
    """Richardson-Zaki index n as a function of Re."""
    return 5.037 - 2.839 * (1.0 - np.exp(-0.1687 * Re ** 0.38))


def concentration_factor(w_single_cm_s, d_mm, nu, c):
    """
    Returns (Re, n, R_factor) where R_factor = (1-c)^n is the hindered-settling
    reduction factor. Re is computed from the single-particle velocity (prediction
    mode, no measured velocity available).
    """
    Re = (w_single_cm_s / 100.0) * (d_mm / 1000.0) / nu
    n = rz_index(Re)
    R = (1.0 - c) ** n
    return Re, n, R
