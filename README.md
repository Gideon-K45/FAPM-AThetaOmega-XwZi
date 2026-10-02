#!/usr/bin/env python3
"""
FAPM-AθΩ · XwZi Fusion — Alpha.theta.AΩ · v6.7
Model Creator: Gideon W. Kanyara
Copyright (c) 2026 Gideon W. Kanyara. All rights reserved.
Location: Nairobi, Kenya
Date of Creation: 09/2026
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, logging, math, os, pickle, tempfile, warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fapm_v67")

VERSION = "FAPM-AθΩ-XwZi-6.7"
SCHEMA_VERSION = 19
EPS = 1e-12
NANOSECONDS_PER_DAY = 86_400.0 * 1_000_000_000.0
MASTER_SEED = int(os.environ.get("FAPM_SEED", "20260922"))
OUTCOME_INV = {0: "H", 1: "D", 2: "A"}
BATCH_JOINT_MAX_ELEMS = 4_000_000
DIRICHLET_MAX_ELEMS = 1_000_000

# ============================================================================
# 1. Utilities
# ============================================================================

def clip01(x):
    try: f = float(x)
    except (TypeError, ValueError): return 0.0
    return float(np.clip(f, 0.0, 1.0)) if math.isfinite(f) else 0.0

def to_bool(v):
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if v is None: return False
    try:
        if isinstance(v, float) and math.isnan(v): return False
    except Exception: pass
    if isinstance(v, str): return v.strip().lower() in {"true", "1", "yes", "y", "t"}
    try: return bool(int(v))
    except (TypeError, ValueError): return False

def is_missing(v):
    if v is None: return True
    if isinstance(v, float) and math.isnan(v): return True
    if isinstance(v, str) and v.strip() == "": return True
    try: return bool(pd.isna(v))
    except Exception: return False

def to_num(v, default=0.0):
    if is_missing(v): return float(default)
    try:
        f = float(v); return f if math.isfinite(f) else float(default)
    except (TypeError, ValueError): return float(default)

def to_ts(v):
    t = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(t) else t

def clip_probs(p):
    """Strict: raises on non-finite input. Used at gateways where the caller
    must have already validated."""
    x = np.asarray(p, float)
    if not np.isfinite(x).all(): raise ValueError("non-finite probabilities")
    x = np.clip(x, EPS, None)
    return x / max(float(x.sum()), EPS)

def clip_matrix(P, strict=False):
    """Bulk NaN-safe row normaliser. Set strict=True to raise on all-NaN rows."""
    X = np.asarray(P, float)
    if strict and not np.isfinite(X).any(axis=1).all():
        bad = ~np.isfinite(X).any(axis=1)
        raise ValueError(f"clip_matrix strict: {int(bad.sum())} all-NaN rows")
    X = np.where(np.isfinite(X), X, EPS)
    X = np.clip(X, EPS, None)
    s = X.sum(axis=1, keepdims=True)
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
    out = X / s
    bad = ~np.isfinite(out).all(axis=1)
    if bad.any():
        out[bad] = 1.0 / out.shape[1]
    return out

def logloss_vec(p, y):
    p = np.asarray(p, float); y = np.asarray(y, int)
    if y.size and (y.min() < 0 or y.max() > p.shape[1] - 1):
        raise ValueError(f"y out of range: {y.min()}..{y.max()}")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))

def binary_logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def brier_vec(p, y):
    oh = np.eye(3)[np.asarray(y, int)]
    return float(np.mean(np.sum((np.asarray(p) - oh) ** 2, axis=1)))

def rps_vec(p, y):
    oh = np.eye(3)[np.asarray(y, int)]
    c1 = np.cumsum(np.asarray(p, float), axis=1)[:, :2]
    c2 = np.cumsum(oh, axis=1)[:, :2]
    return float(0.5 * np.mean(np.sum((c1 - c2) ** 2, axis=1)))

def ece_top_class(p, y, bins=12):
    p = np.asarray(p, float); y = np.asarray(y, int)
    conf = p.max(1); hit = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1); out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf <= hi) if hi == 1 else (conf < hi))
        if m.any(): out += float(m.mean()) * abs(float(hit[m].mean()) - float(conf[m].mean()))
    return float(out)

def _temper_1x2(p, T):
    p = np.atleast_2d(np.asarray(p, float))
    if abs(T - 1.0) < 1e-9: return p
    log_p = np.log(np.clip(p, EPS, 1.0)) / float(T)
    log_p -= logsumexp(log_p, axis=1, keepdims=True)
    return np.exp(log_p)

def _temper_binary(p, T):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    if abs(T - 1.0) < 1e-9: return p
    z = np.log(p / (1 - p)) / float(T)
    return 1.0 / (1.0 + np.exp(-z))

def poisson_pmf(lam, cap):
    lam = float(np.clip(lam, 1e-6, 30.0)); k = np.arange(cap + 1, dtype=float)
    log_pmf = -lam + k * math.log(lam) - gammaln(k + 1.0)
    pmf = np.exp(log_pmf); pmf[cap] += max(1.0 - float(pmf.sum()), 0.0)
    return pmf / max(pmf.sum(), EPS)

def nb_pmf(lam, phi, cap):
    lam = float(np.clip(lam, 1e-6, 30.0)); phi = float(np.clip(phi, 0.5, 500.0))
    p = phi / (phi + lam); p = float(np.clip(p, 1e-12, 1 - 1e-12))
    k = np.arange(cap + 1, dtype=float)
    log_pmf = (gammaln(k + phi) - gammaln(k + 1) - gammaln(phi)
               + phi * math.log(p) + k * math.log1p(-p))
    pmf = np.exp(log_pmf); pmf[cap] += max(1.0 - float(pmf.sum()), 0.0)
    return pmf / max(pmf.sum(), EPS)

def nb_surface(lam_h, lam_a, phi_h, phi_a, cap=20):
    ph = nb_pmf(lam_h, phi_h, cap); pa = nb_pmf(lam_a, phi_a, cap)
    J = ph[:, None] * pa[None, :]
    return J / max(float(J.sum()), EPS)

def _batch_poisson_pmf(lams, cap):
    lams = np.clip(np.asarray(lams, float), 1e-6, 30.0)
    k = np.arange(cap + 1, dtype=float)[None, :]
    log_pmf = -lams[:, None] + k * np.log(lams)[:, None] - gammaln(k + 1)
    pmf = np.exp(log_pmf)
    pmf[:, -1:] += np.maximum(1.0 - pmf.sum(axis=1, keepdims=True), 0.0)
    return pmf / np.maximum(pmf.sum(axis=1, keepdims=True), EPS)

def _batch_bp_joint(lam_h, lam_a, l3, cap):
    l3 = float(max(l3, 1e-6))
    l1 = np.maximum(np.asarray(lam_h, float) - l3, 1e-6)
    l2 = np.maximum(np.asarray(lam_a, float) - l3, 1e-6)
    P1 = _batch_poisson_pmf(l1, cap); P2 = _batch_poisson_pmf(l2, cap)
    P3 = poisson_pmf(l3, cap); n = len(lam_h)
    J = np.zeros((n, cap + 1, cap + 1))
    for kk in range(cap + 1):
        n1 = cap + 1 - kk
        if n1 <= 0: break
        J[:, kk:, kk:] += P3[kk] * (P1[:, :n1, None] * P2[:, None, :n1])
    s = J.sum(axis=(1, 2), keepdims=True)
    return J / np.maximum(s, EPS)

def bivariate_poisson_surface(lam_h, lam_a, lam_3, cap=20):
    return _batch_bp_joint(np.array([float(lam_h)]), np.array([float(lam_a)]), float(lam_3), cap)[0]

def outcome_from_matrix(J):
    ii = np.arange(J.shape[0])[:, None]; jj = np.arange(J.shape[1])[None, :]
    return np.array([J[ii > jj].sum(), J[ii == jj].sum(), J[ii < jj].sum()], dtype=float)

def outcome_from_batch(J):
    L = J.shape[1]; idx = np.arange(L)
    mh = idx[:, None] > idx[None, :]; md = idx[:, None] == idx[None, :]; ma = idx[:, None] < idx[None, :]
    pH = (J * mh).sum(axis=(1, 2)); pD = (J * md).sum(axis=(1, 2)); pA = (J * ma).sum(axis=(1, 2))
    out = np.stack([pH, pD, pA], axis=1)
    return out / np.maximum(out.sum(axis=1, keepdims=True), EPS)

def apply_dc(J, lam_h, lam_a, rho):
    lh = max(float(lam_h), EPS); la = max(float(lam_a), EPS)
    lo = max(-1.0 / lh, -1.0 / la) + 1e-8; hi = min(1.0, 1.0 / (lh * la)) - 1e-8
    rho = float(np.clip(rho, lo, hi)) if lo <= hi else 0.0
    out = np.asarray(J, float).copy()
    if out.shape[0] < 2 or out.shape[1] < 2 or rho == 0.0:
        return out / max(float(out.sum()), EPS)
    for (i, j), f in {(0, 0): 1 - lh * la * rho, (0, 1): 1 + lh * rho,
                      (1, 0): 1 + la * rho, (1, 1): 1 - rho}.items():
        out[i, j] *= max(float(f), 0.0)
    return np.clip(out, 0.0, None) / max(float(out.sum()), EPS)

def _batch_apply_dc(J, lam_h, lam_a, rho):
    lh = np.maximum(np.asarray(lam_h, float), EPS)
    la = np.maximum(np.asarray(lam_a, float), EPS)
    lo = np.maximum(-1.0 / lh, -1.0 / la) + 1e-8
    hi = np.minimum(1.0, 1.0 / (lh * la)) - 1e-8
    rho_arr = np.where(lo <= hi, np.clip(rho, lo, hi), 0.0)
    out = J.copy()
    out[:, 0, 0] *= np.maximum(1 - lh * la * rho_arr, 0.0)
    out[:, 0, 1] *= np.maximum(1 + lh * rho_arr, 0.0)
    out[:, 1, 0] *= np.maximum(1 + la * rho_arr, 0.0)
    out[:, 1, 1] *= np.maximum(1 - rho_arr, 0.0)
    s = out.sum(axis=(1, 2), keepdims=True)
    return out / np.maximum(s, EPS)

def score_temperature(J, T):
    T = float(np.clip(T, 0.25, 4.0))
    logJ = np.log(np.clip(np.asarray(J, float), EPS, None)) / T
    logJ -= logsumexp(logJ)
    return np.exp(logJ)

def project_to_joint(J0, target_1x2=None, iters=1, tol=1e-9):
    J = np.asarray(J0, float).copy(); L = J.shape[0]
    H, A = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    m1 = [(H > A), (H == A), (H < A)]
    if J.sum() <= 0: J = np.ones_like(J) / J.size
    max_dev = 0.0
    for _ in range(max(1, int(iters))):
        J = np.maximum(J, 1e-15)
        if target_1x2 is not None:
            for k, mk in enumerate(m1):
                mass = J[mk].sum()
                if mass > 1e-12: J[mk] *= float(target_1x2[k]) / mass
        J /= max(J.sum(), EPS)
        dev = 0.0
        if target_1x2 is not None:
            for k, mk in enumerate(m1): dev = max(dev, abs(float(J[mk].sum()) - float(target_1x2[k])))
        max_dev = dev
        if dev < tol: break
    return J, (max_dev < 1e-6), float(max_dev)

def _vectorized_match_keys(dates, leagues, homes, aways):
    dts = pd.to_datetime(pd.Series(dates).to_numpy(), errors="coerce", utc=True)
    dts = pd.Series(dts).dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")
    lg = pd.Series(leagues).astype(str).str.strip().str.upper()
    hm = pd.Series(homes).astype(str).str.strip().str.upper()
    aw = pd.Series(aways).astype(str).str.strip().str.upper()
    raw = dts + "|" + lg + "|" + hm + "|" + aw
    return [hashlib.sha256(r.encode("utf-8")).hexdigest()[:24] for r in raw.tolist()]

def stable_match_key(date, league, home, away):
    return _vectorized_match_keys([date], [league], [home], [away])[0]

def sha256_json(obj):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def atomic_write(path, blob, compress=False):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if compress: blob = gzip.compress(blob)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp_", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f: f.write(blob)
        os.replace(tmp, str(p))
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def chronological_block_bootstrap(delta, dates, block_days=14, n_boot=2000, seed=MASTER_SEED):
    d = pd.to_datetime(pd.Series(dates), errors="coerce"); x = np.asarray(delta, float)
    if len(x) != len(d) or len(x) == 0 or d.isna().any(): raise ValueError("invalid loss/dates")
    day = d.dt.normalize().to_numpy(); days = pd.unique(day)
    L = max(1, min(int(block_days), len(days)))
    blocks = [np.flatnonzero(np.isin(day, days[i:i + L])) for i in range(0, len(days) - L + 1)]
    if not blocks:
        m = float(x.mean()); return {"mean": m, "lo": m, "hi": m}
    rng = np.random.default_rng(seed); boots = np.empty(max(400, int(n_boot)))
    for b in range(len(boots)):
        idx = []; total = 0
        while total < len(x):
            z = blocks[int(rng.integers(0, len(blocks)))]; idx.append(z); total += len(z)
        boots[b] = float(x[np.concatenate(idx)[:len(x)]].mean())
    return {"mean": float(x.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}

def _odds_to_prob_vec(v, prob_threshold=1.01):
    v = np.asarray(v, float)
    is_odds = np.isfinite(v) & (v > prob_threshold)
    out = np.where(is_odds, 1.0 / np.maximum(v, prob_threshold), v)
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)

# ============================================================================
# 2. Copulas
# ============================================================================

def _frank_C_vec(u, v, th):
    small = np.abs(th) < 1e-8
    th_safe = np.where(small, 1.0, th)
    au = np.clip(-th_safe * u, -700.0, 700.0)
    av = np.clip(-th_safe * v, -700.0, 700.0)
    emt = np.expm1(-th_safe); emt = np.where(np.abs(emt) < 1e-15, -1e-15, emt)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inner = 1.0 + (np.expm1(au) * np.expm1(av)) / emt
        inner = np.maximum(inner, 1e-12)
        stable = -np.log(inner) / th_safe
    return np.where(small, u * v, stable)

def _clayton_C_vec(u, v, th):
    th_pos = np.maximum(th, 1e-6)
    u = np.maximum(u, 1e-12); v = np.maximum(v, 1e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        base = np.maximum(np.power(u, -th_pos) + np.power(v, -th_pos) - 1.0, 1e-12)
        return np.power(base, -1.0 / th_pos)

def _gumbel_C_vec(u, v, th):
    th_pos = np.maximum(th, 1.0 + 1e-6)
    lu = -np.log(np.maximum(u, 1e-12)); lv = -np.log(np.maximum(v, 1e-12))
    with np.errstate(over="ignore", invalid="ignore"):
        inner = np.power(lu, th_pos) + np.power(lv, th_pos)
        return np.exp(-np.power(inner, 1.0 / th_pos))

def _copula_C_vec(u, v, theta, family):
    if family == "frank": return _frank_C_vec(u, v, theta)
    if family == "clayton": return _clayton_C_vec(u, v, theta)
    if family == "gumbel": return _gumbel_C_vec(u, v, theta)
    return u * v

def _cumulative_from_pmf(pmf_row):
    return np.concatenate([[0.0], np.cumsum(pmf_row)])

def copula_joint_vec(Fh, Fa, theta, family="frank"):
    u_hi = Fh[:, 1:]; u_lo = Fh[:, :-1]; v_hi = Fa[:, 1:]; v_lo = Fa[:, :-1]
    u_hi = u_hi[:, :, None]; u_lo = u_lo[:, :, None]
    v_hi = v_hi[:, None, :]; v_lo = v_lo[:, None, :]
    th = np.asarray(theta).reshape(-1, 1, 1)
    J = (_copula_C_vec(u_hi, v_hi, th, family) - _copula_C_vec(u_lo, v_hi, th, family)
         - _copula_C_vec(u_hi, v_lo, th, family) + _copula_C_vec(u_lo, v_lo, th, family))
    J = np.clip(J, 0.0, None)
    s = J.sum(axis=(1, 2), keepdims=True)
    return J / np.maximum(s, EPS)

def copula_joint(marg_h, marg_a, theta, family="frank"):
    Fh = _cumulative_from_pmf(marg_h)[None, :]; Fa = _cumulative_from_pmf(marg_a)[None, :]
    return copula_joint_vec(Fh, Fa, np.array([float(theta)]), family)[0]

_COPULA_BOUNDS = {"frank": (-25.0, 25.0), "clayton": (1e-4, 25.0), "gumbel": (1.0001, 25.0)}

def _batch_nb_pmf(lams, phi, cap):
    lams = np.clip(np.asarray(lams, float), 1e-6, 30.0)
    phi_c = float(np.clip(phi, 0.5, 500.0))
    k = np.arange(cap + 1, dtype=float)[None, :]
    p = np.clip(phi_c / (phi_c + lams), 1e-12, 1 - 1e-12)
    log_pmf = (gammaln(k + phi_c) - gammaln(k + 1) - gammaln(phi_c)
               + phi_c * np.log(p)[:, None] + k * np.log1p(-p)[:, None])
    pmf = np.exp(log_pmf)
    pmf[:, -1:] += np.maximum(1.0 - pmf.sum(axis=1, keepdims=True), 0.0)
    return pmf / np.maximum(pmf.sum(axis=1, keepdims=True), EPS)

def _batch_nb_pmf_var(lams, phis, cap):
    lams = np.clip(np.asarray(lams, float), 1e-6, 30.0)
    phis = np.clip(np.asarray(phis, float), 0.5, 500.0)
    k = np.arange(cap + 1, dtype=float)[None, :]
    p = np.clip(phis[:, None] / (phis[:, None] + lams[:, None]), 1e-12, 1 - 1e-12)
    log_pmf = (gammaln(k + phis[:, None]) - gammaln(k + 1) - gammaln(phis[:, None])
               + phis[:, None] * np.log(p) + k * np.log1p(-p))
    pmf = np.exp(log_pmf)
    pmf[:, -1:] += np.maximum(1.0 - pmf.sum(axis=1, keepdims=True), 0.0)
    return pmf / np.maximum(pmf.sum(axis=1, keepdims=True), EPS)

def fit_copula_theta(lam_h, lam_a, obs_h, obs_a, phi_h, phi_a, cap, family="frank", max_n=4000, seed=MASTER_SEED):
    n = len(lam_h)
    if n == 0: return 0.0
    if n > max_n:
        rng = np.random.default_rng(seed); idx = rng.choice(n, size=max_n, replace=False)
        lam_h = np.asarray(lam_h)[idx]; lam_a = np.asarray(lam_a)[idx]
        obs_h = np.asarray(obs_h)[idx]; obs_a = np.asarray(obs_a)[idx]
        n = max_n
    PH = _batch_nb_pmf(lam_h, phi_h, cap); PA = _batch_nb_pmf(lam_a, phi_a, cap)
    Fh = np.concatenate([np.zeros((n, 1)), np.cumsum(PH, axis=1)], axis=1)
    Fa = np.concatenate([np.zeros((n, 1)), np.cumsum(PA, axis=1)], axis=1)
    gi = np.clip(np.asarray(obs_h, int), 0, cap); gj = np.clip(np.asarray(obs_a, int), 0, cap)
    idx_flat = gi * (cap + 1) + gj
    def nll(theta):
        J = copula_joint_vec(Fh, Fa, np.full(n, float(theta)), family)
        prob = J.reshape(n, -1)[np.arange(n), idx_flat]
        return -float(np.sum(np.log(np.maximum(prob, EPS))))
    lo, hi = _COPULA_BOUNDS.get(family, (-25.0, 25.0))
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    return float(res.x) if np.isfinite(res.x) else 0.0

def fit_bp_lambda3(lam_h, lam_a, obs_h, obs_a, cap, max_n=4000, seed=MASTER_SEED):
    n = len(lam_h)
    if n == 0: return 0.0
    if n > max_n:
        rng = np.random.default_rng(seed); idx = rng.choice(n, size=max_n, replace=False)
        lam_h = np.asarray(lam_h)[idx]; lam_a = np.asarray(lam_a)[idx]
        obs_h = np.asarray(obs_h)[idx]; obs_a = np.asarray(obs_a)[idx]
        n = max_n
    gi = np.clip(np.asarray(obs_h, int), 0, cap); gj = np.clip(np.asarray(obs_a, int), 0, cap)
    idx_flat = gi * (cap + 1) + gj
    upper = float(np.minimum(np.percentile(lam_h, 5.0), np.percentile(lam_a, 5.0))) if n else 0.5
    upper = max(upper, 1e-3)
    def nll(l3):
        J = _batch_bp_joint(lam_h, lam_a, float(l3), cap)
        prob = J.reshape(n, -1)[np.arange(n), idx_flat]
        return -float(np.sum(np.log(np.maximum(prob, EPS))))
    res = minimize_scalar(nll, bounds=(1e-6, upper), method="bounded", options={"xatol": 1e-5})
    return float(res.x) if np.isfinite(res.x) else 0.0

# ============================================================================
# 3. PIT gate + evidence adapter + ledger
# ============================================================================

@dataclass
class PITConfig:
    min_latency_seconds: float = 0.0
    max_future_tolerance_seconds: float = 0.0
    reject_unknown_publication_time: bool = True
    reject_post_cutoff: bool = True
    reject_cumulative_season_snapshots: bool = True
    require_ingestion_timestamp: bool = True
    require_source_before_ingestion: bool = True

TRUSTED_EXACT_SOURCES = {"rotowire", "lineupslabs", "official-leagues", "uefa",
                          "bundesliga-official", "ff-scout", "api-football"}
EVIDENCE_TYPE_MAP = {
    "expected_xi": "expected_xi", "predicted_xi": "expected_xi",
    "lineup_confirmation": "lineup_confirmation", "availability": "availability",
    "transfer_event": "transfer", "transfer": "transfer",
    "registration_snapshot": "registration", "registration": "registration",
    "market_snapshot": "market_snapshot", "process_snapshot": "process_snapshot",
    "process_match": "process_snapshot", "fixture": "fixture",
    "player_match_performance": "player_match_performance",
}

class Top15EvidenceAdapter:
    MARKET_LEADS = (1440, 360, 60, 15, 5)
    def __init__(self, strict=True): self.strict = bool(strict)
    @staticmethod
    def load_jsonl(paths):
        rows = []
        for p in paths:
            p = Path(p)
            if not p.exists(): continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: continue
        return pd.DataFrame(rows)
    @staticmethod
    def load_csv(paths):
        frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    def normalize(self, evidence):
        if evidence.empty: return evidence.copy()
        x = evidence.copy()
        def first(row, names, default=None):
            for n in names:
                if n in row.index and pd.notna(row[n]) and str(row[n]).strip() not in {"", "nan", "None"}:
                    return row[n]
            return default
        out = []
        for _, r in x.iterrows():
            raw_rt = first(r, ['record_type', 'evidence_type'], '')
            if not raw_rt:
                if 'expected_xi_json' in r.index: raw_rt = 'expected_xi'
                elif 'player_match_rating' in r.index or ('minutes' in r.index and 'player_id' in r.index):
                    raw_rt = 'player_match_performance'
                elif 'market_p_home' in r.index or 'market_home_odds' in r.index: raw_rt = 'market_snapshot'
                elif 'status' in r.index and 'player' in r.index: raw_rt = 'availability'
                else: raw_rt = 'process_snapshot'
            rt = EVIDENCE_TYPE_MAP.get(str(raw_rt), str(raw_rt))
            out.append({
                'record_type': rt,
                'event_id': first(r, ['event_id', 'match_id', 'fixture_id']),
                'event_time_utc': first(r, ['event_time_utc', 'kickoff', 'kickoff_utc']),
                'source_time_utc': first(r, ['effective_time_utc', 'source_publication_time_utc',
                                              'source_observation_timestamp', 'source_timestamp_utc']),
                'collector_time_utc': first(r, ['collector_observed_at_utc', 'retrieved_at', 'retrieved_at_utc']),
                'forecast_cutoff_utc': first(r, ['forecast_cutoff_utc', 'information_cutoff_utc']),
                'timestamp_precision': str(first(r, ['timestamp_precision', 'source_timestamp_precision'], '')),
                'data_quality_grade': str(first(r, ['data_quality_grade', 'evidence_grade'], '')),
                'usage_policy': str(first(r, ['usage_policy'], '')),
                'team': first(r, ['team', 'home_team']),
                'player': first(r, ['player', 'player_name']),
                'player_id': first(r, ['player_id', 'playerId']),
                'minutes': first(r, ['minutes', 'min']),
                'rating': first(r, ['rating', 'player_rating', 'match_rating']),
                'goals': first(r, ['goals', 'goals_scored']),
                'assists': first(r, ['assists']),
                'shots': first(r, ['shots', 'total_shots']),
                'shots_on_target': first(r, ['shots_on_target', 'sot']),
                'key_passes': first(r, ['key_passes', 'keypasses']),
                'source': first(r, ['source', 'source_id']),
                'pit_admissible_raw': first(r, ['pit_admissible'], False),
                'home_probability': first(r, ['market_p_home']),
                'draw_probability': first(r, ['market_p_draw']),
                'away_probability': first(r, ['market_p_away']),
                'market_home_odds': first(r, ['market_home_odds']),
                'market_draw_odds': first(r, ['market_draw_odds']),
                'market_away_odds': first(r, ['market_away_odds']),
                'expected_xi_json': first(r, ['expected_xi_json']),
                'xi_n': first(r, ['expected_xi_n']),
                'xi_change_from_prior': first(r, ['xi_change_from_prior_observation']),
                'status': first(r, ['status']),
                'notes': first(r, ['notes']),
            })
        return pd.DataFrame(out)
    @staticmethod
    def _prob_tuple(row):
        p = []
        for k in ('home_probability', 'draw_probability', 'away_probability'):
            try: p.append(float(row[k]))
            except Exception: p.append(np.nan)
        if any(not np.isfinite(v) or v <= 0 for v in p):
            odds = []
            for k in ('market_home_odds', 'market_draw_odds', 'market_away_odds'):
                try: odds.append(float(row[k]))
                except Exception: odds.append(np.nan)
            if all(np.isfinite(v) and v > 1 for v in odds): p = [1 / v for v in odds]
        if not all(np.isfinite(v) and v > 0 for v in p): return None
        s = sum(p)
        return tuple(v / s for v in p) if s > 0 else None

class ProspectiveEvidenceLedger:
    def __init__(self, strict=True, require_source_before_ingestion=True):
        self.strict = bool(strict)
        self.require_source_before_ingestion = bool(require_source_before_ingestion)
        self.adapter = Top15EvidenceAdapter(strict=self.strict)
        self.records = pd.DataFrame(); self._strict_eligible = None
    @staticmethod
    def _canonical_hash(row):
        keep = {str(k): str(v) for k, v in row.items() if k not in {"retrieved_at", "collector_observed_at_utc", "ingestion_time_utc"}}
        return sha256_json(keep)
    def ingest(self, evidence):
        x = self.adapter.normalize(evidence.copy()) if not evidence.empty else pd.DataFrame()
        if x.empty:
            self.records = x; self._strict_eligible = None
            return {"accepted": 0, "duplicates": 0, "records": 0}
        x["canonical_record_hash"] = x.apply(self._canonical_hash, axis=1)
        x["_source_time"] = x["source_time_utc"].map(to_ts)
        x["_collector_time"] = x["collector_time_utc"].map(to_ts)
        x["_effective_time"] = x["_source_time"].combine_first(x["_collector_time"])
        x["match_id"] = x["event_id"]
        before = len(x)
        x = x.drop_duplicates(subset=["canonical_record_hash"], keep="last").reset_index(drop=True)
        self.records = x.sort_values(["_source_time"], kind="mergesort", na_position="last").reset_index(drop=True)
        exact = self.records["timestamp_precision"].astype(str).str.contains("exact", case=False, na=False)
        grade = self.records["data_quality_grade"].astype(str).isin(["A", "A_or_B_exact", "B", "B_exact"])
        trusted = self.records["source"].astype(str).str.lower().isin(TRUSTED_EXACT_SOURCES)
        policy = self.records["usage_policy"].astype(str).eq("strict_PIT_eligible") | grade | (exact & trusted)
        raw = self.records["pit_admissible_raw"].map(to_bool)
        if self.require_source_before_ingestion:
            src_ok = self.records["_source_time"].notna() & self.records["_collector_time"].notna() & \
                     (self.records["_source_time"] <= self.records["_collector_time"])
        else:
            src_ok = pd.Series([True] * len(self.records), index=self.records.index)
        self._strict_eligible = raw & exact & policy & self.records["_source_time"].notna() & src_ok
        return {"accepted": int(len(x)), "duplicates": int(before - len(x)), "records": int(len(self.records))}
    def feature_frame(self, fixtures):
        if fixtures is None or fixtures.empty: return pd.DataFrame()
        fx = fixtures.copy()
        if "match_id" not in fx.columns: fx["match_id"] = fx.get("fixture_id", fx.index).astype(str)
        fx["match_id"] = fx["match_id"].astype(str)
        if "cutoff_ts" not in fx.columns:
            for c in ("forecast_cutoff_utc", "kickoff_utc", "date"):
                if c in fx.columns: fx["cutoff_ts"] = fx[c].map(to_ts); break
        rec = self.records
        if rec.empty:
            return pd.DataFrame({"match_id": fx["match_id"].astype(str), "market_ladder_count": 0, "evidence_present": 0.0})
        strict_ok = self._strict_eligible if self._strict_eligible is not None else pd.Series([True] * len(rec), index=rec.index)
        market = rec[strict_ok & rec["record_type"].eq("market_snapshot")].copy()
        market["_event_time"] = market["event_time_utc"].map(to_ts)
        market["_lead"] = (market["_event_time"] - market["_source_time"]).dt.total_seconds() / 60.0
        market["_probs"] = market.apply(lambda r: self.adapter._prob_tuple(r), axis=1)
        market = market[market["_probs"].notna()].copy()
        xi = rec[strict_ok & rec["record_type"].isin(["expected_xi", "lineup_confirmation", "availability"])].copy()
        xi["team_lc"] = xi["team"].astype(str).str.casefold()
        process = rec[rec["record_type"].eq("process_snapshot")].copy()
        process["team_lc"] = process["team"].astype(str).str.casefold()
        process["_notes_shots"] = process["notes"].astype(str).str.extract(r"shots\s*=\s*([0-9.]+)", expand=False).astype(float)
        process["_notes_xg"] = process["notes"].astype(str).str.extract(r"(?:xG|xg)\s*=\s*([0-9.]+)", expand=False).astype(float)
        state_ev = rec[rec["record_type"].isin(["transfer", "registration"])].copy()
        state_ev["team_lc"] = state_ev["team"].astype(str).str.casefold()
        player_perf = rec[rec["record_type"].eq("player_match_performance")].copy()
        player_perf["team_lc"] = player_perf["team"].astype(str).str.casefold()
        rows = []
        for _, f in fx.iterrows():
            fid = str(f["match_id"]); cts = f.get("cutoff_ts")
            home = str(f.get("home_team", f.get("home", ""))); away = str(f.get("away_team", f.get("away", "")))
            h_lc = home.casefold(); a_lc = away.casefold()
            r = {"match_id": fid, "home_team": home, "away_team": away,
                 "market_ladder_count": 0, "market_near_kickoff_entropy": np.nan, "market_drift_home": np.nan,
                 "xi_home_freshness_min": np.nan, "xi_away_freshness_min": np.nan,
                 "xi_home_change": 0.0, "xi_away_change": 0.0,
                 "availability_home_events": 0, "availability_away_events": 0,
                 "transfer_home_30d": 0, "transfer_away_30d": 0,
                 "registration_home_30d": 0, "registration_away_30d": 0,
                 "process_home_n": 0, "process_away_n": 0,
                 "process_home_xg_ewma": np.nan, "process_away_xg_ewma": np.nan,
                 "process_home_shots_ewma": np.nan, "process_away_shots_ewma": np.nan,
                 "player_perf_home_n": 0, "player_perf_away_n": 0, "evidence_present": 0.0}
            if cts is not None and not pd.isna(cts):
                m_rows = market[(market["match_id"].astype(str) == fid) & (market["_source_time"] <= cts)]
                if not m_rows.empty:
                    leads = []
                    tol_map = {1440: 360, 360: 90, 60: 15, 15: 5, 5: 7.5}
                    for _, mr in m_rows.iterrows():
                        lead = max(float(mr["_lead"]), 0.0)
                        nearest = min(self.adapter.MARKET_LEADS, key=lambda z: abs(lead - z))
                        if abs(lead - nearest) <= tol_map[nearest]: leads.append((nearest, mr["_probs"], lead))
                    if leads:
                        r["market_ladder_count"] = len(leads)
                        near = min(leads, key=lambda l: abs(l[2] - 5.0))
                        ph, pdr, pa = near[1]; probs = np.array([ph, pdr, pa])
                        r["market_near_kickoff_entropy"] = float(-np.sum(probs * np.log(np.maximum(probs, EPS))) / math.log(3.0))
                        if len(leads) >= 2:
                            home_ps = [l[1][0] for l in sorted(leads, key=lambda l: -l[2])]
                            r["market_drift_home"] = float(home_ps[-1] - home_ps[0])
                for side, team_lc in (("home", h_lc), ("away", a_lc)):
                    xr = xi[(xi["match_id"].astype(str) == fid) & (xi["team_lc"] == team_lc) & (xi["_source_time"] <= cts)]
                    if not xr.empty:
                        latest = xr.sort_values("_source_time").iloc[-1]
                        r[f"xi_{side}_freshness_min"] = max((cts - latest["_source_time"]).total_seconds() / 60.0, 0.0)
                        try: r[f"xi_{side}_change"] = abs(float(latest.get("xi_change_from_prior", 0.0)))
                        except Exception: pass
                    ar = xi[(xi["match_id"].astype(str) == fid) & (xi["team_lc"] == team_lc)
                             & (xi["record_type"] == "availability") & (xi["_source_time"] <= cts)]
                    r[f"availability_{side}_events"] = int(len(ar))
                start = cts - pd.Timedelta(days=30)
                for side, team_lc in (("home", h_lc), ("away", a_lc)):
                    st_ev = state_ev[(state_ev["team_lc"] == team_lc) & (state_ev["_effective_time"].between(start, cts, inclusive="both"))]
                    r[f"transfer_{side}_30d"] = int((st_ev["record_type"] == "transfer").sum())
                    r[f"registration_{side}_30d"] = int((st_ev["record_type"] == "registration").sum())
                for side, team_lc in (("home", h_lc), ("away", a_lc)):
                    pr = process[(process["team_lc"] == team_lc) & (process["_effective_time"] <= cts)]
                    r[f"process_{side}_n"] = len(pr)
                    if not pr.empty:
                        ages = ((cts - pr["_effective_time"]).dt.total_seconds() / 86400.0).clip(lower=0)
                        w = 0.5 ** (ages / 14.0)
                        sw = float(w.sum())
                        if sw > 0 and pr["_notes_xg"].notna().any():
                            r[f"process_{side}_xg_ewma"] = float((w * pr["_notes_xg"].fillna(0)).sum() / sw)
                        if sw > 0 and pr["_notes_shots"].notna().any():
                            r[f"process_{side}_shots_ewma"] = float((w * pr["_notes_shots"].fillna(0)).sum() / sw)
                for side, team_lc in (("home", h_lc), ("away", a_lc)):
                    pp = player_perf[(player_perf["match_id"].astype(str) == fid) & (player_perf["team_lc"] == team_lc)]
                    r[f"player_perf_{side}_n"] = len(pp)
                r["evidence_present"] = 1.0 if any([
                    r["market_ladder_count"] > 0,
                    pd.notna(r["xi_home_freshness_min"]) or pd.notna(r["xi_away_freshness_min"]),
                    r["availability_home_events"] > 0 or r["availability_away_events"] > 0,
                    r["process_home_n"] > 0 or r["process_away_n"] > 0,
                    r["player_perf_home_n"] > 0 or r["player_perf_away_n"] > 0]) else 0.0
            rows.append(r)
        return pd.DataFrame(rows)
    def readiness(self):
        x = self.records
        if x.empty: return {"records": 0, "status": "NOT_READY"}
        strict = x[self._strict_eligible] if self._strict_eligible is not None else x.iloc[0:0]
        return {"records": int(len(x)), "strict_source_records": int(len(strict)),
                "status": "PROSPECTIVE_COLLECTION_REQUIRED" if len(strict) < 100 else "READY_FOR_SHADOW_REPLAY"}

@dataclass
class EvidenceCoverageConfig:
    min_pit_matches_shadow: int = 100
    min_pit_matches_promotion: int = 200
    min_folds_promotion: int = 5
    max_missing_channel_fraction: float = 0.80
    min_market_matches_promotion: int = 100
    min_player_performance_matches_promotion: int = 100
    min_xi_matches_promotion: int = 100
    max_abs_logit_correction: float = 0.15
    temporal_purge_days: int = 1
    min_out_of_sample_gain: float = 0.0005

class EvidenceCoverageController:
    def __init__(self, cfg=None): self.cfg = cfg or EvidenceCoverageConfig()
    def assess(self, fixtures, features):
        n = max(len(fixtures), 1)
        def s(name):
            return pd.to_numeric(features[name], errors="coerce") if name in features.columns \
                else pd.Series(index=features.index, dtype=float)
        market = int((s("market_ladder_count").fillna(0) > 0).sum()) if len(features) else 0
        xi = int((s("xi_home_freshness_min").notna() | s("xi_away_freshness_min").notna()).sum()) if len(features) else 0
        availability = int((s("availability_home_events").fillna(0) + s("availability_away_events").fillna(0) > 0).sum()) if len(features) else 0
        process = int((s("process_home_n").fillna(0) + s("process_away_n").fillna(0) > 0).sum()) if len(features) else 0
        player_perf = int((s("player_perf_home_n").fillna(0) + s("player_perf_away_n").fillna(0) > 0).sum()) if len(features) else 0
        counts = {"market": market, "xi": xi, "availability": availability, "process": process, "player_performance": player_perf}
        return {"fixtures": int(len(fixtures)), "channel_counts": counts,
                "channel_rates": {k: v / n for k, v in counts.items()},
                "shadow_ready": len(fixtures) >= self.cfg.min_pit_matches_shadow,
                "promotion_sample_ready": len(fixtures) >= self.cfg.min_pit_matches_promotion}

class CrossFittedEvidenceResidualExpert:
    BASE_FEATURES = ("market_ladder_count", "market_near_kickoff_entropy", "market_drift_home",
                      "xi_home_freshness_min", "xi_away_freshness_min", "xi_home_change", "xi_away_change",
                      "availability_home_events", "availability_away_events",
                      "transfer_home_30d", "transfer_away_30d",
                      "registration_home_30d", "registration_away_30d",
                      "process_home_n", "process_away_n",
                      "process_home_xg_ewma", "process_away_xg_ewma",
                      "process_home_shots_ewma", "process_away_shots_ewma",
                      "player_perf_home_n", "player_perf_away_n")
    DERIVED = ("market_ladder_entropy_range", "xi_disagreement_proxy", "availability_intensity",
               "process_xg_gap", "process_shots_gap", "registration_gap", "player_perf_gap")
    FEATURES = BASE_FEATURES + DERIVED
    def __init__(self, cfg=None):
        self.cfg = cfg or EvidenceCoverageConfig()
        self.model = None; self.fitted = False; self.n_fit = 0
        self.feature_medians = {}; self.feature_scales = {}; self.fold_reports = []
    @staticmethod
    def _augment(x):
        y = x.copy()
        def num(c): return pd.to_numeric(y.get(c, np.nan), errors="coerce")
        y["market_ladder_entropy_range"] = num("market_near_kickoff_entropy")
        y["xi_disagreement_proxy"] = (num("xi_home_change") - num("xi_away_change")).abs()
        y["availability_intensity"] = num("availability_home_events") + num("availability_away_events")
        y["process_xg_gap"] = num("process_home_xg_ewma") - num("process_away_xg_ewma")
        y["process_shots_gap"] = num("process_home_shots_ewma") - num("process_away_shots_ewma")
        y["registration_gap"] = num("registration_home_30d") - num("registration_away_30d")
        y["player_perf_gap"] = num("player_perf_home_n") - num("player_perf_away_n")
        return y
    def _matrix(self, features, base_p, fit):
        x = self._augment(features).reindex(columns=self.FEATURES, fill_value=np.nan).apply(pd.to_numeric, errors="coerce")
        if fit:
            self.feature_medians = {c: float(x[c].median()) if np.isfinite(x[c]).any() else 0.0 for c in self.FEATURES}
            x = x.fillna(pd.Series(self.feature_medians))
            self.feature_scales = {c: max(float(x[c].std()), 0.25) for c in self.FEATURES}
        else:
            x = x.fillna(pd.Series(self.feature_medians))
        z = ((x - pd.Series(self.feature_medians)) / pd.Series(self.feature_scales))
        z = np.clip(z.to_numpy(float), -5.0, 5.0)
        bp = clip_matrix(base_p); rel = np.log(np.maximum(bp, EPS)); rel = rel - rel[:, [2]]
        return np.hstack([z, rel[:, :2]])
    def _split_indices(self, dates, n_splits):
        d = pd.to_datetime(dates, errors="coerce", utc=True)
        unique = np.array(sorted(pd.Series(d.dropna().dt.normalize().unique())))
        if len(unique) < n_splits + 1: return []
        folds = []
        chunks = np.array_split(unique, n_splits + 1)
        for i in range(1, len(chunks)):
            val_days = set(chunks[i].tolist()); train_days = set(np.concatenate(chunks[:i]).tolist())
            if self.cfg.temporal_purge_days:
                vmin = min(val_days)
                train_days = {x for x in train_days if (vmin - x).days > self.cfg.temporal_purge_days}
            tr = np.where(d.dt.normalize().isin(train_days))[0]
            va = np.where(d.dt.normalize().isin(val_days))[0]
            if len(tr) and len(va): folds.append((tr, va))
        return folds
    def fit_cross_fitted(self, features, base_p, y, dates):
        y = np.asarray(y, dtype=int); base_p = clip_matrix(np.asarray(base_p, dtype=float))
        folds = self._split_indices(pd.Series(dates), self.cfg.min_folds_promotion)
        if len(y) < self.cfg.min_pit_matches_promotion or len(folds) < self.cfg.min_folds_promotion:
            self.fitted = False; self.n_fit = len(y)
            return {"fitted": False, "n": int(len(y)), "folds": len(folds), "reason": "insufficient_sample_or_folds"}
        oof = np.full_like(base_p, np.nan, dtype=float)
        self.fold_reports = []
        for fold_id, (tr, va) in enumerate(folds, 1):
            saved = (self.feature_medians, self.feature_scales)
            self.feature_medians, self.feature_scales = {}, {}
            Xtr = self._matrix(features.iloc[tr], base_p[tr], fit=True)
            Xva = self._matrix(features.iloc[va], base_p[va], fit=False)
            if len(np.unique(y[tr])) < 3:
                self.fold_reports.append({"fold": fold_id, "status": "skipped_single_class"})
                self.feature_medians, self.feature_scales = saved; continue
            clf = LogisticRegression(C=0.25, max_iter=3000).fit(Xtr, y[tr])
            oof[va] = clf.predict_proba(Xva)
            self.fold_reports.append({"fold": fold_id, "n_train": int(len(tr)), "n_test": int(len(va))})
            self.feature_medians, self.feature_scales = saved
        valid = np.isfinite(oof).all(axis=1)
        if valid.sum() < self.cfg.min_pit_matches_promotion:
            self.fitted = False
            return {"fitted": False, "n_oof": int(valid.sum()), "folds": len(folds), "reason": "insufficient_oof"}
        self.feature_medians, self.feature_scales = {}, {}
        Xall = self._matrix(features, base_p, fit=True)
        clf = LogisticRegression(C=0.25, max_iter=3000).fit(Xall, y)
        self.model = clf; self.fitted = True; self.n_fit = len(y)
        return {"fitted": True, "n": int(len(y)), "oof_n": int(valid.sum()), "folds": len(folds), "fold_reports": self.fold_reports}
    def predict(self, features, base_p):
        bp = clip_matrix(base_p)
        if not self.fitted or self.model is None: return bp
        X = self._matrix(features, bp, fit=False)
        p = self.model.predict_proba(X)
        delta = np.log(np.maximum(p, EPS)) - np.log(np.maximum(bp, EPS))
        cap = float(self.cfg.max_abs_logit_correction)
        return clip_matrix(bp * np.exp(np.clip(delta, -cap, cap)))
    def state(self):
        if not self.fitted or self.model is None: return {"fitted": False, "n_fit": self.n_fit}
        return {"fitted": True, "n_fit": self.n_fit, "feature_medians": self.feature_medians,
                "feature_scales": self.feature_scales,
                "coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist(),
                "classes": self.model.classes_.tolist()}
    @classmethod
    def restore(cls, s):
        obj = cls()
        obj.n_fit = int(s.get("n_fit", 0))
        if not s.get("fitted"): return obj
        obj.feature_medians = dict(s.get("feature_medians", {}))
        obj.feature_scales = dict(s.get("feature_scales", {}))
        m = LogisticRegression(C=0.25, max_iter=3000)
        m.classes_ = np.asarray(s["classes"], int)
        m.coef_ = np.asarray(s["coef"], float); m.intercept_ = np.asarray(s["intercept"], float)
        m.n_features_in_ = int(m.coef_.shape[1]); m.n_iter_ = np.array([1], dtype=int)
        obj.model = m; obj.fitted = True
        return obj

# ============================================================================
# 4. Feature contract
# ============================================================================

CORE_FEATURES = (
    "true_elo_diff", "goal_form_diff", "shot_process_diff", "sot_process_diff",
    "form_points_diff10", "rest_diff", "attack_defence_mismatch",
    "elo_form_interaction", "strength_rest_interaction",
    "season_index", "home_games_prior", "away_games_prior",
    "h_matches_last7d", "a_matches_last7d", "h_matches_last14d", "a_matches_last14d",
    "congestion_diff7", "congestion_diff14",
    "h_roll_attack", "a_roll_attack", "h_roll_defence", "a_roll_defence",
    "roll_attack_diff", "roll_defence_diff", "market_strength_diff",
)
EVIDENCE_FEATURES = (
    "market_ladder_count", "market_near_kickoff_entropy", "market_drift_home",
    "xi_home_freshness_min", "xi_away_freshness_min", "xi_home_change", "xi_away_change",
    "availability_home_events", "availability_away_events",
    "transfer_home_30d", "transfer_away_30d", "registration_home_30d", "registration_away_30d",
    "process_home_n", "process_away_n", "process_home_xg_ewma", "process_away_xg_ewma",
    "process_home_shots_ewma", "process_away_shots_ewma",
    "player_perf_home_n", "player_perf_away_n", "evidence_present",
)
EXTENDED_FEATURES = (
    "ref_cards_h", "ref_cards_a", "ref_totals_h", "ref_totals_a",
    "mgr_change_h", "mgr_change_a", "weather_precip", "weather_wind", "weather_temp",
    "is_cup", "is_friendly", "travel_log", "tz_delta",
)
ALL_FEATURES = CORE_FEATURES + EVIDENCE_FEATURES + EXTENDED_FEATURES

# ============================================================================
# 5. Vectorized causal feature builder
# ============================================================================

class CausalFeatureBuilder:
    def __init__(self, half_life_days=35.0, ewma_half_life_days=45.0, roll_half_life_days=90.0,
                 market_half_life_days=180.0, market_ridge=1.0, market_refit_every=200):
        self.half_life_days = float(half_life_days)
        self.ewma_half_life_days = float(ewma_half_life_days)
        self.roll_half_life_days = float(roll_half_life_days)
        self.market_half_life_days = float(market_half_life_days)
        self.market_ridge = float(market_ridge); self.market_refit_every = int(market_refit_every)
        self.reset()

    def reset(self):
        self._team_days: Dict[str, np.ndarray] = {}
        self._team_gf: Dict[str, np.ndarray] = {}
        self._team_ga: Dict[str, np.ndarray] = {}
        self._team_pts: Dict[str, np.ndarray] = {}
        self._cumsum_a_goal: Dict[str, np.ndarray] = {}
        self._cumsum_av_gf_goal: Dict[str, np.ndarray] = {}
        self._cumsum_av_ga_goal: Dict[str, np.ndarray] = {}
        self._cumsum_a_roll: Dict[str, np.ndarray] = {}
        self._cumsum_av_gf_roll: Dict[str, np.ndarray] = {}
        self._cumsum_av_ga_roll: Dict[str, np.ndarray] = {}
        self._cumsum_a_pts: Dict[str, np.ndarray] = {}
        self._cumsum_av_pts: Dict[str, np.ndarray] = {}
        self._roll_pts10: Dict[str, np.ndarray] = {}
        self._team_ref_days: Dict[str, int] = {}
        self._elo: Dict[str, float] = {}
        self._last_date: Dict[str, int] = {}
        self._games: Dict[str, int] = {}
        self._manager_events: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        self._reference_date: Optional[pd.Timestamp] = None
        self._market_snapshot: Dict[str, float] = {}
        self._market_sd_snapshot: float = 1.0

    def register_manager_change(self, team, date, sign=1.0):
        self._manager_events.setdefault(str(team), []).append((pd.Timestamp(date), float(sign)))

    def _manager_impulse(self, team, d_int):
        evs = self._manager_events.get(str(team), [])
        if not evs: return 0.0
        best = 0.0
        for ts, sgn in evs:
            age = (pd.Timestamp(d_int, unit="D") - ts).days
            if 0 <= age <= 60: best = max(best, float(sgn) * math.exp(-age / 45.0))
        return best

    def _build_histories(self, homes, aways, days_int, hg, ag, has_outcome):
        n = len(homes)
        if has_outcome:
            pts_h = np.where(hg > ag, 3.0, np.where(hg == ag, 1.0, 0.0))
            pts_a = np.where(ag > hg, 3.0, np.where(hg == ag, 1.0, 0.0))
        else:
            pts_h = np.zeros(n); pts_a = np.zeros(n)
        team_days: Dict[str, List[int]] = defaultdict(list)
        team_gf: Dict[str, List[float]] = defaultdict(list)
        team_ga: Dict[str, List[float]] = defaultdict(list)
        team_pts: Dict[str, List[float]] = defaultdict(list)
        for i in range(n):
            h = homes[i]; a = aways[i]; d = int(days_int[i])
            team_days[h].append(d); team_gf[h].append(float(hg[i])); team_ga[h].append(float(ag[i])); team_pts[h].append(float(pts_h[i]))
            team_days[a].append(d); team_gf[a].append(float(ag[i])); team_ga[a].append(float(hg[i])); team_pts[a].append(float(pts_a[i]))
        c_goal = math.log(2.0) / self.ewma_half_life_days
        c_roll = math.log(2.0) / self.roll_half_life_days
        for t in list(team_days.keys()):
            d = np.asarray(team_days[t], dtype=np.int64)
            gf = np.asarray(team_gf[t], dtype=np.float64)
            ga = np.asarray(team_ga[t], dtype=np.float64)
            pts = np.asarray(team_pts[t], dtype=np.float64)
            self._team_days[t] = d
            self._team_gf[t] = gf; self._team_ga[t] = ga; self._team_pts[t] = pts
            if len(d) == 0: continue
            d_ref = int(d[0]); self._team_ref_days[t] = d_ref
            a_g = np.exp(c_goal * (d - d_ref).astype(np.float64))
            a_r = np.exp(c_roll * (d - d_ref).astype(np.float64))
            self._cumsum_a_goal[t] = np.cumsum(a_g)
            self._cumsum_av_gf_goal[t] = np.cumsum(a_g * gf)
            self._cumsum_av_ga_goal[t] = np.cumsum(a_g * ga)
            self._cumsum_a_roll[t] = np.cumsum(a_r)
            self._cumsum_av_gf_roll[t] = np.cumsum(a_r * gf)
            self._cumsum_av_ga_roll[t] = np.cumsum(a_r * ga)
            self._cumsum_a_pts[t] = np.cumsum(a_g)
            self._cumsum_av_pts[t] = np.cumsum(a_g * pts)
            W = 10
            csum = np.concatenate([[0.0], np.cumsum(pts)])
            i_arr = np.arange(len(pts))
            lo = np.maximum(0, i_arr - W + 1)
            counts = np.minimum(i_arr + 1, W)
            self._roll_pts10[t] = (csum[i_arr + 1] - csum[lo]) / counts
        self._elo = {}; self._last_date = {}; self._games = {}
        for i in range(n):
            h = homes[i]; a = aways[i]; d = int(days_int[i])
            rh = self._elo.get(h, 1500.0); ra = self._elo.get(a, 1500.0)
            if has_outcome:
                y = 0 if hg[i] > ag[i] else (1 if hg[i] == ag[i] else 2)
                expected = 1.0 / (1.0 + 10.0 ** (-(rh + 60.0 - ra) / 400.0))
                zh = 1.0 if y == 0 else (0.5 if y == 1 else 0.0)
                ch = 20.0 * (zh - expected)
                self._elo[h] = rh + ch; self._elo[a] = ra - ch
            self._last_date[h] = d; self._last_date[a] = d
            self._games[h] = self._games.get(h, 0) + 1
            self._games[a] = self._games.get(a, 0) + 1

    def build(self, master, update_state=True, evidence_features=None):
        n = len(master)
        if update_state:
            ms_h, ms_a, ms_sd, snap, snap_sd = _compute_rolling_market_strength(
                master, half_life_days=self.market_half_life_days, ridge=self.market_ridge,
                refit_every=self.market_refit_every, return_final=True)
            if snap: self._market_snapshot = snap; self._market_sd_snapshot = float(snap_sd)
        else:
            ms_h = np.zeros(n); ms_a = np.zeros(n); ms_sd = np.ones(n)
            snap = self._market_snapshot
            sd_snap = self._market_sd_snapshot if self._market_sd_snapshot > 1e-9 else 1.0
            if snap:
                homes_raw = master["home"].astype(str).tolist(); aways_raw = master["away"].astype(str).tolist()
                ms_h = np.array([snap.get(h, 0.0) for h in homes_raw])
                ms_a = np.array([snap.get(a, 0.0) for a in aways_raw])
                ms_sd = np.full(n, sd_snap)
        x = master.copy()
        for c in ALL_FEATURES:
            if c not in x.columns: x[c] = np.nan
            else: x[c] = pd.to_numeric(x[c], errors="coerce")
        if self._reference_date is None and n:
            self._reference_date = pd.Timestamp(x["date"].min())
        x = x.sort_values(["date", "match_key"], kind="mergesort").reset_index(drop=True)
        sort_idx = master.sort_values(["date", "match_key"], kind="mergesort").index.to_numpy()
        if len(sort_idx) == n:
            ms_h = ms_h[sort_idx]; ms_a = ms_a[sort_idx]; ms_sd = ms_sd[sort_idx]
        homes = x["home"].astype(str).to_numpy(); aways = x["away"].astype(str).to_numpy()
        days_int = x["date"].to_numpy("datetime64[D]").astype(np.int64)
        has_outcome = ("home_goals" in x.columns and "away_goals" in x.columns and update_state)
        hg = x["home_goals"].to_numpy(float) if "home_goals" in x.columns else np.zeros(n)
        ag = x["away_goals"].to_numpy(float) if "away_goals" in x.columns else np.zeros(n)
        neutral = x["is_neutral"].to_numpy(float) if "is_neutral" in x.columns else np.zeros(n)
        mt_arr = (x["match_type"].astype(str).str.lower().to_numpy()
                   if "match_type" in x.columns else np.array(["league"] * n))
        if update_state:
            self._build_histories(homes, aways, days_int, hg, ag, has_outcome)
        out_keys = ("true_elo_diff", "home_games_prior", "away_games_prior", "season_index",
                     "goal_form_diff", "form_points_diff10", "attack_defence_mismatch",
                     "rest_diff", "strength_rest_interaction", "elo_form_interaction",
                     "mgr_change_h", "mgr_change_a", "is_cup", "is_friendly",
                     "h_matches_last7d", "a_matches_last7d", "h_matches_last14d", "a_matches_last14d",
                     "congestion_diff7", "congestion_diff14",
                     "h_roll_attack", "a_roll_attack", "h_roll_defence", "a_roll_defence",
                     "roll_attack_diff", "roll_defence_diff", "market_strength_diff")
        out = {k: np.zeros(n) for k in out_keys}
        ref_day = int(self._reference_date.to_numpy().astype("datetime64[D]").astype(np.int64)) if self._reference_date is not None else 0
        for i in range(n):
            h = homes[i]; a = aways[i]; d = int(days_int[i])
            out["home_games_prior"][i] = self._games.get(h, 0)
            out["away_games_prior"][i] = self._games.get(a, 0)
            out["season_index"][i] = max((d - ref_day) / 365.0, 0.0)
            elo_h = self._elo.get(h, 1500.0); elo_a = self._elo.get(a, 1500.0)
            elo_diff = elo_h - elo_a
            out["true_elo_diff"][i] = elo_diff
            k_h = int(np.searchsorted(self._team_days.get(h, np.zeros(0, dtype=np.int64)), d, side="left"))
            k_a = int(np.searchsorted(self._team_days.get(a, np.zeros(0, dtype=np.int64)), d, side="left"))
            if k_h > 0:
                ca = self._cumsum_a_goal[h]; cg = self._cumsum_av_gf_goal[h]; cd = self._cumsum_av_ga_goal[h]
                cp = self._cumsum_av_pts[h]
                h_gf = float(cg[k_h - 1] / max(ca[k_h - 1], EPS))
                h_ga = float(cd[k_h - 1] / max(ca[k_h - 1], EPS))
                h_pts_ewma = float(cp[k_h - 1] / max(ca[k_h - 1], EPS))
                h_att = float(self._cumsum_av_gf_roll[h][k_h - 1] / max(self._cumsum_a_roll[h][k_h - 1], EPS))
                h_def = float(self._cumsum_av_ga_roll[h][k_h - 1] / max(self._cumsum_a_roll[h][k_h - 1], EPS))
                h_pts10 = float(self._roll_pts10[h][k_h - 1])
            else:
                h_gf = h_ga = h_pts_ewma = h_att = h_def = h_pts10 = None
            if k_a > 0:
                ca = self._cumsum_a_goal[a]; cg = self._cumsum_av_gf_goal[a]; cd = self._cumsum_av_ga_goal[a]
                cp = self._cumsum_av_pts[a]
                a_gf = float(cg[k_a - 1] / max(ca[k_a - 1], EPS))
                a_ga = float(cd[k_a - 1] / max(ca[k_a - 1], EPS))
                a_pts_ewma = float(cp[k_a - 1] / max(ca[k_a - 1], EPS))
                a_att = float(self._cumsum_av_gf_roll[a][k_a - 1] / max(self._cumsum_a_roll[a][k_a - 1], EPS))
                a_def = float(self._cumsum_av_ga_roll[a][k_a - 1] / max(self._cumsum_a_roll[a][k_a - 1], EPS))
                a_pts10 = float(self._roll_pts10[a][k_a - 1])
            else:
                a_gf = a_ga = a_pts_ewma = a_att = a_def = a_pts10 = None
            if h_gf is not None and a_gf is not None:
                out["goal_form_diff"][i] = (h_gf - h_ga) - (a_gf - a_ga)
                out["attack_defence_mismatch"][i] = (h_gf - a_ga) - (a_gf - h_ga)
            if h_pts10 is not None and a_pts10 is not None:
                out["form_points_diff10"][i] = h_pts10 - a_pts10
            if k_h > 0 and k_a > 0:
                h_rest = float(d - self._team_days[h][k_h - 1])
                a_rest = float(d - self._team_days[a][k_a - 1])
                out["rest_diff"][i] = h_rest - a_rest
                out["strength_rest_interaction"][i] = (elo_diff / 100.0) * (h_rest - a_rest) / 7.0
            out["elo_form_interaction"][i] = elo_diff * out["form_points_diff10"][i]
            out["mgr_change_h"][i] = self._manager_impulse(h, d)
            out["mgr_change_a"][i] = self._manager_impulse(a, d)
            mt = mt_arr[i]
            out["is_cup"][i] = 1.0 if "cup" in mt else 0.0
            out["is_friendly"][i] = 1.0 if "friend" in mt else 0.0
            td_h = self._team_days.get(h, np.zeros(0, dtype=np.int64))
            td_a = self._team_days.get(a, np.zeros(0, dtype=np.int64))
            k_h_7 = int(np.searchsorted(td_h, d - 7, side="left"))
            k_a_7 = int(np.searchsorted(td_a, d - 7, side="left"))
            k_h_14 = int(np.searchsorted(td_h, d - 14, side="left"))
            k_a_14 = int(np.searchsorted(td_a, d - 14, side="left"))
            out["h_matches_last7d"][i] = k_h - k_h_7
            out["a_matches_last7d"][i] = k_a - k_a_7
            out["h_matches_last14d"][i] = k_h - k_h_14
            out["a_matches_last14d"][i] = k_a - k_a_14
            out["congestion_diff7"][i] = out["h_matches_last7d"][i] - out["a_matches_last7d"][i]
            out["congestion_diff14"][i] = out["h_matches_last14d"][i] - out["a_matches_last14d"][i]
            out["h_roll_attack"][i] = h_att if h_att is not None else 0.0
            out["a_roll_attack"][i] = a_att if a_att is not None else 0.0
            out["h_roll_defence"][i] = h_def if h_def is not None else 0.0
            out["a_roll_defence"][i] = a_def if a_def is not None else 0.0
            out["roll_attack_diff"][i] = out["h_roll_attack"][i] - out["a_roll_attack"][i]
            out["roll_defence_diff"][i] = out["h_roll_defence"][i] - out["a_roll_defence"][i]
            sd_i = ms_sd[i] if ms_sd[i] > 1e-9 else 1.0
            out["market_strength_diff"][i] = (ms_h[i] - ms_a[i]) / sd_i
        for k, v in out.items(): x[k] = v
        x["is_neutral"] = neutral
        if evidence_features is not None and not evidence_features.empty:
            ev = evidence_features.copy()
            ev["match_id"] = ev["match_id"].astype(str)
            ev = ev.drop_duplicates(subset=["match_id"], keep="last")
            cols = ["match_id"] + [c for c in EVIDENCE_FEATURES if c in ev.columns and c != "evidence_present"]
            x["match_id"] = x["match_key"].astype(str) if "match_key" in x.columns else x.index.astype(str)
            if "match_id" in master.columns:
                x["match_id"] = master.sort_values(["date", "match_key"], kind="mergesort")["match_id"].astype(str).to_numpy()
            x = x.merge(ev[cols], on="match_id", how="left")
            for c in EVIDENCE_FEATURES:
                if c not in x.columns: x[c] = np.nan
            x["evidence_present"] = x.get("evidence_present", pd.Series(0.0, index=x.index)).fillna(0.0)
        else:
            for c in EVIDENCE_FEATURES:
                if c not in x.columns: x[c] = np.nan
            x["evidence_present"] = 0.0
        return x

    def state(self):
        return {
            "half_life_days": self.half_life_days, "ewma_half_life_days": self.ewma_half_life_days,
            "roll_half_life_days": self.roll_half_life_days,
            "market_half_life_days": self.market_half_life_days,
            "market_ridge": self.market_ridge, "market_refit_every": self.market_refit_every,
            "team_days": dict(self._team_days),
            "team_gf": dict(self._team_gf), "team_ga": dict(self._team_ga), "team_pts": dict(self._team_pts),
            "cumsum_a_goal": dict(self._cumsum_a_goal), "cumsum_av_gf_goal": dict(self._cumsum_av_gf_goal),
            "cumsum_av_ga_goal": dict(self._cumsum_av_ga_goal),
            "cumsum_a_roll": dict(self._cumsum_a_roll), "cumsum_av_gf_roll": dict(self._cumsum_av_gf_roll),
            "cumsum_av_ga_roll": dict(self._cumsum_av_ga_roll),
            "cumsum_a_pts": dict(self._cumsum_a_pts), "cumsum_av_pts": dict(self._cumsum_av_pts),
            "roll_pts10": dict(self._roll_pts10),
            "team_ref_days": dict(self._team_ref_days),
            "elo": dict(self._elo), "last_date": dict(self._last_date), "games": dict(self._games),
            "manager_events": {k: [(str(t), float(s)) for t, s in evs] for k, evs in self._manager_events.items()},
            "reference_date": self._reference_date,
            "market_snapshot": dict(self._market_snapshot),
            "market_sd_snapshot": float(self._market_sd_snapshot),
        }
    @classmethod
    def from_state(cls, s):
        obj = cls(half_life_days=s.get("half_life_days", 35.0),
                  ewma_half_life_days=s.get("ewma_half_life_days", 45.0),
                  roll_half_life_days=s.get("roll_half_life_days", 90.0),
                  market_half_life_days=s.get("market_half_life_days", 180.0),
                  market_ridge=s.get("market_ridge", 1.0),
                  market_refit_every=s.get("market_refit_every", 200))
        def _arr(v, dt): return {k: np.asarray(a, dtype=dt) for k, a in v.items()}
        obj._team_days = _arr(s.get("team_days", {}), np.int64)
        obj._team_gf = _arr(s.get("team_gf", {}), np.float64)
        obj._team_ga = _arr(s.get("team_ga", {}), np.float64)
        obj._team_pts = _arr(s.get("team_pts", {}), np.float64)
        obj._cumsum_a_goal = _arr(s.get("cumsum_a_goal", {}), np.float64)
        obj._cumsum_av_gf_goal = _arr(s.get("cumsum_av_gf_goal", {}), np.float64)
        obj._cumsum_av_ga_goal = _arr(s.get("cumsum_av_ga_goal", {}), np.float64)
        obj._cumsum_a_roll = _arr(s.get("cumsum_a_roll", {}), np.float64)
        obj._cumsum_av_gf_roll = _arr(s.get("cumsum_av_gf_roll", {}), np.float64)
        obj._cumsum_av_ga_roll = _arr(s.get("cumsum_av_ga_roll", {}), np.float64)
        obj._cumsum_a_pts = _arr(s.get("cumsum_a_pts", {}), np.float64)
        obj._cumsum_av_pts = _arr(s.get("cumsum_av_pts", {}), np.float64)
        obj._roll_pts10 = _arr(s.get("roll_pts10", {}), np.float64)
        obj._team_ref_days = {k: int(v) for k, v in s.get("team_ref_days", {}).items()}
        obj._elo = {str(k): float(v) for k, v in s.get("elo", {}).items()}
        obj._last_date = {str(k): int(v) for k, v in s.get("last_date", {}).items()}
        obj._games = {str(k): int(v) for k, v in s.get("games", {}).items()}
        obj._manager_events = {str(k): [(pd.Timestamp(t), float(sc)) for t, sc in evs]
                                for k, evs in s.get("manager_events", {}).items()}
        rd = s.get("reference_date")
        if rd is None: obj._reference_date = None
        elif isinstance(rd, pd.Timestamp): obj._reference_date = rd
        else: obj._reference_date = pd.Timestamp(rd)
        obj._market_snapshot = dict(s.get("market_snapshot", {}))
        obj._market_sd_snapshot = float(s.get("market_sd_snapshot", 1.0))
        return obj

# ============================================================================
# 6. Rolling market strength
# ============================================================================

def _extract_market_probabilities(df):
    hcols = ("market_p_home", "b365h", "home_odds", "psh", "home_prob")
    acols = ("market_p_away", "b365a", "away_odds", "psa", "away_prob")
    n = len(df); ph = np.full(n, np.nan); pa = np.full(n, np.nan)
    for hc in hcols:
        if hc in df.columns:
            v = pd.to_numeric(df[hc], errors="coerce").to_numpy(float)
            inv = _odds_to_prob_vec(v)
            ph = np.where(np.isfinite(inv), inv, ph)
    for ac in acols:
        if ac in df.columns:
            v = pd.to_numeric(df[ac], errors="coerce").to_numpy(float)
            inv = _odds_to_prob_vec(v)
            pa = np.where(np.isfinite(inv), inv, pa)
    return ph, pa

def _compute_rolling_market_strength(df, half_life_days=180.0, ridge=1.0, refit_every=200, return_final=False):
    n = len(df)
    if n == 0:
        empty = (np.zeros(0), np.zeros(0), np.zeros(0))
        return (*empty, {}, 1.0) if return_final else empty
    ph, pa = _extract_market_probabilities(df)
    valid = np.isfinite(ph) & np.isfinite(pa) & (ph > 0) & (pa > 0)
    if valid.sum() < 30:
        z = np.zeros(n); o = np.ones(n)
        return (z, z, o, {}, 1.0) if return_final else (z, z, o)
    teams = sorted(set(df["home"].astype(str)) | set(df["away"].astype(str)))
    team_idx = {t: i for i, t in enumerate(teams)}; n_t = len(teams)
    hi = df["home"].astype(str).map(team_idx).to_numpy(np.int64)
    ai = df["away"].astype(str).map(team_idx).to_numpy(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(valid, np.log(np.clip(ph, 1e-3, 1 - 1e-3)) - np.log(np.clip(pa, 1e-3, 1 - 1e-3)), np.nan)
    dates = pd.to_datetime(df["date"]).to_numpy("datetime64[ns]").astype(np.int64)
    order = np.argsort(dates, kind="mergesort")
    strengths = np.zeros(n_t); sd = 1.0
    decay_w = math.log(2.0) / max(half_life_days, 1e-6)
    s_home = np.zeros(n); s_away = np.zeros(n); s_sd = np.zeros(n)
    last_refit = -10**9
    past_h: List[int] = []; past_a: List[int] = []; past_y: List[float] = []; past_d: List[int] = []
    for pos, i in enumerate(order):
        if valid[i] and (pos - last_refit) >= refit_every and len(past_y) >= 30:
            current = dates[i]
            ages_ns = np.array([current - d for d in past_d], dtype=float)
            ages_days = ages_ns / NANOSECONDS_PER_DAY
            w = np.exp(-decay_w * np.maximum(ages_days, 0.0))
            y = np.asarray(past_y, float)
            h_arr = np.asarray(past_h, np.int64); a_arr = np.asarray(past_a, np.int64); sw = np.sqrt(w)
            rows_c = np.concatenate([np.arange(len(y)), np.arange(len(y))])
            cols_c = np.concatenate([h_arr, a_arr]); vals_c = np.concatenate([sw, -sw])
            A = sparse.coo_matrix((vals_c, (rows_c, cols_c)), shape=(len(y), n_t)).tocsr()
            G = (A.T @ A + ridge * sparse.eye(n_t)).tocsc()
            rhs = np.asarray(A.T @ (sw * y)).ravel()
            try: sol = sparse.linalg.spsolve(G, rhs)
            except Exception: sol = np.linalg.lstsq(G.toarray(), rhs, rcond=None)[0]
            sol = np.asarray(sol, float).ravel(); sol -= sol.mean()
            strengths = sol; sd = float(strengths.std()) if strengths.size else 1.0; last_refit = pos
        s_home[i] = strengths[hi[i]] if hi[i] >= 0 else 0.0
        s_away[i] = strengths[ai[i]] if ai[i] >= 0 else 0.0
        s_sd[i] = sd if sd > 1e-9 else 1.0
        if valid[i]:
            past_h.append(int(hi[i])); past_a.append(int(ai[i]))
            past_y.append(float(log_ratio[i])); past_d.append(int(dates[i]))
    if return_final:
        final = {t: float(strengths[team_idx[t]]) for t in teams}
        return s_home, s_away, s_sd, final, float(sd)
    return s_home, s_away, s_sd

# ============================================================================
# 7. Causal contract
# ============================================================================

_LEAK_EXACT = {"home_goals", "away_goals", "hg", "ag", "FTHG", "FTAG",
               "home_score", "away_score", "result", "y", "score", "outcome"}
_LEAK_SUFFIXES = ("_result", "_score", "_outcome")

class CausalContract:
    RENAME = {"Div": "league", "League": "league", "division": "league", "competition": "league",
              "Competition": "league", "competition_code": "league",
              "Date": "date", "kickoff_utc": "date", "kickoff": "date",
              "HomeTeam": "home", "Home": "home", "home_team": "home",
              "AwayTeam": "away", "Away": "away", "away_team": "away",
              "FTHG": "home_goals", "HG": "home_goals", "hg": "home_goals", "home_goals": "home_goals",
              "FTAG": "away_goals", "AG": "away_goals", "ag": "away_goals", "away_goals": "away_goals",
              "Referee": "referee", "MatchType": "match_type", "isNeutral": "is_neutral"}
    @classmethod
    def normalize(cls, df, require_outcome=True):
        if df is None or df.empty: raise ValueError("empty match dataframe")
        x = df.copy()
        for a, b in cls.RENAME.items():
            if a in x.columns and b not in x.columns: x[b] = x[a]
        if "league" not in x.columns: raise ValueError("missing league")
        if not {"date", "home", "away"}.issubset(x.columns): raise ValueError("missing date/home/away")
        x["date"] = pd.to_datetime(x["date"], errors="coerce", utc=True).dt.tz_convert(None)
        if x["date"].isna().any(): raise ValueError("invalid match dates")
        for c in ("home", "away", "league"):
            x[c] = x[c].astype(str).str.strip()
            if (x[c] == "").any(): raise ValueError(f"blank {c}")
        if require_outcome:
            if not {"home_goals", "away_goals"}.issubset(x.columns): raise ValueError("outcomes required")
            x["home_goals"] = pd.to_numeric(x["home_goals"], errors="coerce")
            x["away_goals"] = pd.to_numeric(x["away_goals"], errors="coerce")
            if x[["home_goals", "away_goals"]].isna().any().any(): raise ValueError("missing goals")
            x["y"] = np.where(x.home_goals > x.away_goals, 0,
                              np.where(x.home_goals == x.away_goals, 1, 2)).astype(int)
        x["match_key"] = _vectorized_match_keys(x.date, x.league, x.home, x.away)
        dup = x[x.duplicated("match_key", keep=False)]
        if not dup.empty and require_outcome:
            conflict = dup.groupby("match_key")[["home_goals", "away_goals"]].nunique(dropna=False).max(axis=1)
            if bool((conflict > 1).any()):
                raise ValueError(f"conflicting duplicates: {conflict[conflict > 1].index.tolist()[:5]}")
        x = x.sort_values(["date", "match_key"], kind="mergesort").drop_duplicates("match_key", keep="first")
        return x.reset_index(drop=True)
    @staticmethod
    def leak_check(df):
        hits = set(_LEAK_EXACT.intersection(map(str, df.columns)))
        for c in df.columns:
            name = str(c).lower()
            for suf in _LEAK_SUFFIXES:
                if name.endswith(suf): hits.add(str(c)); break
        return sorted(hits)

def ensure_features(df, cols, fill=None):
    x = df.copy()
    for i, c in enumerate(cols):
        if c not in x.columns: x[c] = float(fill[i]) if fill is not None else np.nan
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")
            if fill is not None: x[c] = x[c].fillna(float(fill[i]))
    return x

# ============================================================================
# 8. Maher model
# ============================================================================

@dataclass
class MaherConfig:
    l2_team: float = 0.035
    l2_league: float = 0.02
    l2_ha: float = 0.02
    l2_covariate: float = 0.05
    l2_evidence: float = 0.10
    l2_ident: float = 5.0
    recency_half_life_days: Optional[float] = 150.0
    max_iter: int = 500
    score_cap: int = 22
    use_nb: bool = True
    nb_phi_min: float = 0.5
    nb_phi_max: float = 200.0
    eta_lo: float = -8.0
    eta_hi: float = 3.0
    pool_kappa: float = 10.0
    use_mad_scale: bool = True
    phi_uncertainty: bool = True
    gamma_uncertainty: bool = True
    evidence_weight_boost: float = 0.5

class MaherPoissonModel:
    def __init__(self, cfg=None):
        self.cfg = cfg or MaherConfig()
        self.teams = {}; self.leagues = {}
        self.active_features = []
        self.evidence_feature_set = set(EVIDENCE_FEATURES)
        self.theta = np.zeros(0); self.theta_cov = np.zeros((0, 0)); self.theta_chol = np.zeros((0, 0))
        self.feature_mean = np.zeros(0); self.feature_scale = np.ones(0)
        self.phi_h = 8.0; self.phi_a = 8.0; self.n_h_obs = 0; self.n_a_obs = 0
        self.log_lambda_total_mean = 0.0
        self.dispersion_gamma_h = 0.0; self.dispersion_gamma_a = 0.0
        self.dispersion_gamma_sd_h = 0.0; self.dispersion_gamma_sd_a = 0.0
        self.n_l = 0; self.n_t = 0; self.p = 0; self.n_params = 0
        self.fitted = False; self.design_blocks = {}
        self.kappa_mean = 0.0; self.ha_l_mean = 0.0
        self.attack_mean = 0.0; self.defence_mean = 0.0
        self.league_attack_mean = {}; self.league_defence_mean = {}
        self.team_league = {}; self.team_games = np.zeros(0)
        self.team_prior_attack = np.zeros(0); self.team_prior_defence = np.zeros(0)
        self.league_prior_attack = np.zeros(0); self.league_prior_defence = np.zeros(0)
        self.needs_refit = False
        self._dc_copula_warned = False

    def _feature_matrix(self, df):
        if not self.active_features: return np.zeros((len(df), 0))
        x = ensure_features(df, self.active_features, fill=self.feature_mean)
        X = x[self.active_features].to_numpy(float)
        X = np.where(np.isfinite(X), X, self.feature_mean[None, :])
        return (X - self.feature_mean) / self.feature_scale

    def _build_long_design(self, df):
        n = len(df); n_l = self.n_l; n_t = self.n_t; p = self.p
        off_ha_global = 0; off_kappa = 1; off_ha_l = 1 + n_l
        off_attack = 1 + 2 * n_l; off_defence = off_attack + n_t
        off_beta_h = off_defence + n_t; off_beta_a = off_beta_h + p
        n_cols = off_beta_a + p
        lg = df.league.map(self.leagues).fillna(-1).astype(int).to_numpy()
        hi = df.home.map(self.teams).fillna(-1).astype(int).to_numpy()
        ai = df.away.map(self.teams).fillna(-1).astype(int).to_numpy()
        Xf = self._feature_matrix(df)
        neutral = df["is_neutral"].to_numpy(float) if "is_neutral" in df.columns else np.zeros(n)
        rows_h = np.arange(n); rows_a = np.arange(n) + n
        rl, cl, vl = [], [], []
        nf = (1.0 - (neutral > 0.5).astype(float))
        rl.append(rows_h); cl.append(np.full(n, off_ha_global, dtype=np.int64)); vl.append(nf)
        vlg = lg >= 0
        if vlg.any():
            rl.append(np.concatenate([rows_h[vlg], rows_a[vlg]]))
            cl.append(np.concatenate([off_kappa + lg[vlg], off_kappa + lg[vlg]]))
            vl.append(np.ones(2 * int(vlg.sum())))
        vha = vlg & (neutral > 0.5)
        if vha.any():
            rl.append(np.concatenate([rows_h[vha], rows_a[vha]]))
            cl.append(np.concatenate([off_ha_l + lg[vha], off_ha_l + lg[vha]]))
            vl.append(np.concatenate([0.5 * np.ones(int(vha.sum())), -0.5 * np.ones(int(vha.sum()))]))
        vh = hi >= 0; va = ai >= 0
        if vh.any(): rl.append(rows_h[vh]); cl.append(off_attack + hi[vh]); vl.append(np.ones(int(vh.sum())))
        if va.any(): rl.append(rows_a[va]); cl.append(off_attack + ai[va]); vl.append(np.ones(int(va.sum())))
        if va.any(): rl.append(rows_h[va]); cl.append(off_defence + ai[va]); vl.append(-np.ones(int(va.sum())))
        if vh.any(): rl.append(rows_a[vh]); cl.append(off_defence + hi[vh]); vl.append(-np.ones(int(vh.sum())))
        if p > 0:
            for j in range(p):
                xj = Xf[:, j]; nz = xj != 0.0
                if not nz.any(): continue
                k = int(nz.sum())
                rl.append(rows_h[nz]); cl.append(np.full(k, off_beta_h + j, dtype=np.int64)); vl.append(xj[nz])
                rl.append(rows_a[nz]); cl.append(np.full(k, off_beta_a + j, dtype=np.int64)); vl.append(xj[nz])
        if rl:
            rr = np.concatenate(rl); cc = np.concatenate(cl).astype(np.int64); vv = np.concatenate(vl).astype(float)
        else:
            rr = np.array([], dtype=np.int64); cc = np.array([], dtype=np.int64); vv = np.array([], dtype=float)
        X = sparse.coo_matrix((vv, (rr, cc)), shape=(2 * n, n_cols), dtype=float).tocsr()
        y = np.r_[df.home_goals.to_numpy(float), df.away_goals.to_numpy(float)]
        self.design_blocks = {
            "ha_global": slice(off_ha_global, off_ha_global + 1),
            "kappa": slice(off_kappa, off_kappa + n_l), "ha_l": slice(off_ha_l, off_ha_l + n_l),
            "attack": slice(off_attack, off_attack + n_t), "defence": slice(off_defence, off_defence + n_t),
            "beta_h": slice(off_beta_h, off_beta_h + p), "beta_a": slice(off_beta_a, off_beta_a + p)}
        return X, y

    def _penalty_vector(self):
        pen = np.zeros(self.n_params)
        pen[self.design_blocks["ha_global"]] = 1.0 / max(self.cfg.l2_team, 1e-6)
        pen[self.design_blocks["kappa"]] = 1.0 / max(self.cfg.l2_league, 1e-6)
        pen[self.design_blocks["ha_l"]] = 1.0 / max(self.cfg.l2_ha, 1e-6)
        pen[self.design_blocks["attack"]] = 1.0 / max(self.cfg.l2_team, 1e-6)
        pen[self.design_blocks["defence"]] = 1.0 / max(self.cfg.l2_team, 1e-6)
        bh = self.design_blocks["beta_h"]; ba = self.design_blocks["beta_a"]
        for j, name in enumerate(self.active_features):
            lam = 1.0 / max(self.cfg.l2_evidence if name in self.evidence_feature_set else self.cfg.l2_covariate, 1e-6)
            pen[bh.start + j] = lam; pen[ba.start + j] = lam
        return pen

    def _nll_and_grad(self, theta, X, y, w, pen, ida, idd):
        eta = X @ theta; eta_c = np.clip(eta, self.cfg.eta_lo, self.cfg.eta_hi)
        active = ((eta > self.cfg.eta_lo) & (eta < self.cfg.eta_hi)).astype(float)
        mu = np.exp(eta_c)
        nll = float(np.sum(w * (mu - y * eta_c)) + 0.5 * np.sum(pen * theta * theta))
        grad = np.asarray(X.T @ (w * active * (mu - y))).ravel() + pen * theta
        if ida.sum() > 0:
            s = float(theta[ida].sum()); nll += 0.5 * self.cfg.l2_ident * s * s; grad[ida] += self.cfg.l2_ident * s
        if idd.sum() > 0:
            s = float(theta[idd].sum()); nll += 0.5 * self.cfg.l2_ident * s * s; grad[idd] += self.cfg.l2_ident * s
        return nll, grad

    def fit(self, df):
        x = CausalContract.normalize(df, True)
        teams = list(pd.unique(pd.concat([x.home, x.away], ignore_index=True)))
        leagues = list(pd.unique(x.league))
        self.teams = {t: i for i, t in enumerate(teams)}; self.leagues = {l: i for i, l in enumerate(leagues)}
        self.n_t = len(teams); self.n_l = len(leagues)
        active = []
        for c in ALL_FEATURES:
            if c in x.columns:
                v = pd.to_numeric(x[c], errors="coerce")
                if float(np.nanstd(v)) > 1e-9: active.append(c)
        self.active_features = active
        if self.active_features:
            raw = x[self.active_features].apply(pd.to_numeric, errors="coerce")
            self.feature_mean = raw.mean().fillna(0.0).to_numpy(float)
            if self.cfg.use_mad_scale:
                med = raw.median(); mad = (raw - med).abs().median()
                scale = 1.4826 * mad; std = raw.std(ddof=0)
                scale = scale.replace(0.0, np.nan).fillna(std).replace(0.0, 1.0).fillna(1.0)
                self.feature_scale = scale.to_numpy(float)
            else:
                self.feature_scale = raw.std(ddof=0).replace(0, 1.0).fillna(1.0).to_numpy(float)
        else:
            self.feature_mean = np.zeros(0); self.feature_scale = np.ones(0)
        self.p = len(self.active_features)
        self.n_params = 1 + 2 * self.n_l + 2 * self.n_t + 2 * self.p
        X, y = self._build_long_design(x)
        pen = self._penalty_vector()
        if self.cfg.recency_half_life_days is None: w_base = np.ones(len(y))
        else:
            age = (x["date"].max() - x["date"]).dt.total_seconds() / 86400.0
            wh = np.exp(-math.log(2.0) * age / max(self.cfg.recency_half_life_days, 1e-6)).to_numpy(float)
            w_base = np.tile(wh, 2)
        if self.cfg.evidence_weight_boost > 0 and "evidence_present" in x.columns:
            ev = pd.to_numeric(x["evidence_present"], errors="coerce").fillna(0.0).to_numpy(float)
            boost = 1.0 + self.cfg.evidence_weight_boost * np.clip(ev, 0.0, 1.0)
            w = w_base * np.tile(boost, 2)
        else:
            w = w_base
        theta0 = np.zeros(self.n_params)
        theta0[self.design_blocks["ha_global"]] = 0.1
        theta0[self.design_blocks["kappa"]] = math.log(max(float(np.clip(y.mean(), 0.1, None)), 0.1))
        ida = np.zeros(self.n_params, bool); ida[self.design_blocks["attack"]] = True
        idd = np.zeros(self.n_params, bool); idd[self.design_blocks["defence"]] = True
        res = minimize(self._nll_and_grad, theta0, args=(X, y, w, pen, ida, idd), method="L-BFGS-B", jac=True,
                       options={"maxiter": int(self.cfg.max_iter), "ftol": 1e-10, "gtol": 1e-7})
        self.theta = np.asarray(res.x, float)
        att = self.theta[self.design_blocks["attack"]].copy(); dfn = self.theta[self.design_blocks["defence"]].copy()
        self.attack_mean = float(att.mean()); self.defence_mean = float(dfn.mean())
        self.theta[self.design_blocks["attack"]] -= self.attack_mean
        self.theta[self.design_blocks["defence"]] -= self.defence_mean
        self.theta[self.design_blocks["ha_global"]] += (self.attack_mean - self.defence_mean)
        kappa_arr = self.theta[self.design_blocks["kappa"]]; ha_arr = self.theta[self.design_blocks["ha_l"]]
        self.kappa_mean = float(kappa_arr.mean()) if self.n_l > 0 else 0.0
        self.ha_l_mean = float(ha_arr.mean()) if self.n_l > 0 else 0.0
        counts = {}
        for _, r in x.iterrows():
            li = self.leagues.get(str(r.league), -1)
            if li < 0: continue
            for t in (str(r.home), str(r.away)):
                ti = self.teams.get(t, -1)
                if ti < 0: continue
                counts.setdefault(ti, {}); counts[ti][li] = counts[ti].get(li, 0) + 1
        for ti, c in counts.items():
            if c: self.team_league[ti] = max(c, key=c.get)
        for li in range(self.n_l):
            members = [ti for ti, ll in self.team_league.items() if ll == li]
            if not members: continue
            self.league_attack_mean[li] = float(np.mean([self.theta[self.design_blocks["attack"]][ti] for ti in members]))
            self.league_defence_mean[li] = float(np.mean([self.theta[self.design_blocks["defence"]][ti] for ti in members]))
        self.team_games = np.zeros(self.n_t, dtype=float)
        for _, r in x.iterrows():
            for t in (str(r.home), str(r.away)):
                ti = self.teams.get(t, -1)
                if ti >= 0: self.team_games[ti] += 1.0
        self.team_prior_attack = np.full(self.n_t, self.attack_mean)
        self.team_prior_defence = np.full(self.n_t, self.defence_mean)
        for ti, li in self.team_league.items():
            if li in self.league_attack_mean: self.team_prior_attack[ti] = self.league_attack_mean[li]
            if li in self.league_defence_mean: self.team_prior_defence[ti] = self.league_defence_mean[li]
        self.league_prior_attack = np.array([self.league_attack_mean.get(li, self.attack_mean) for li in range(self.n_l)]) if self.n_l > 0 else np.zeros(0)
        self.league_prior_defence = np.array([self.league_defence_mean.get(li, self.defence_mean) for li in range(self.n_l)]) if self.n_l > 0 else np.zeros(0)
        eta = np.clip(X @ self.theta, self.cfg.eta_lo, self.cfg.eta_hi); mu = np.exp(eta); W = w * mu
        H = (X.T @ sparse.diags(W) @ X).toarray() + np.diag(pen)
        for blk in ("attack", "defence"):
            s = self.design_blocks[blk]
            if s.stop > s.start: H[s, s] += self.cfg.l2_ident * np.ones(s.stop - s.start)
        H = H + 1e-8 * np.eye(self.n_params)
        try:
            c, low = cho_factor(H, lower=True); Sigma = cho_solve((c, low), np.eye(self.n_params))
        except Exception: Sigma = np.linalg.pinv(H)
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigs = np.linalg.eigvalsh(Sigma)
        if eigs.min() <= 0: Sigma += (abs(eigs.min()) + 1e-8) * np.eye(self.n_params)
        self.theta_cov = Sigma
        try: self.theta_chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(self.n_params))
        except np.linalg.LinAlgError: self.theta_chol = np.eye(self.n_params) * math.sqrt(1e-6)
        if self.cfg.use_nb:
            n_rows = len(y) // 2
            y_h = y[:n_rows]; y_a = y[n_rows:]
            mu_h = mu[:n_rows]; mu_a = mu[n_rows:]
            self.n_h_obs = int(n_rows); self.n_a_obs = int(n_rows)
            pear_h = (y_h - mu_h) ** 2 / np.maximum(mu_h, EPS)
            pear_a = (y_a - mu_a) ** 2 / np.maximum(mu_a, EPS)
            over_h = max(float(pear_h.mean()) - 1.0, 0.05); over_a = max(float(pear_a.mean()) - 1.0, 0.05)
            self.phi_h = float(np.clip(float(np.maximum(mu_h, EPS).mean()) / over_h, self.cfg.nb_phi_min, self.cfg.nb_phi_max))
            self.phi_a = float(np.clip(float(np.maximum(mu_a, EPS).mean()) / over_a, self.cfg.nb_phi_min, self.cfg.nb_phi_max))
            log_lam_h = np.log(np.maximum(mu_h, EPS)); log_lam_a = np.log(np.maximum(mu_a, EPS))
            self.log_lambda_total_mean = float(np.mean(np.concatenate([log_lam_h, log_lam_a])))
            try:
                gh, gh_sd = self._gamma_glm_fit(log_lam_h, ((y_h - mu_h) ** 2) / np.maximum(mu_h, EPS))
                ga, ga_sd = self._gamma_glm_fit(log_lam_a, ((y_a - mu_a) ** 2) / np.maximum(mu_a, EPS))
                self.dispersion_gamma_h = float(np.clip(gh, -0.5, 0.5)); self.dispersion_gamma_a = float(np.clip(ga, -0.5, 0.5))
                self.dispersion_gamma_sd_h = float(np.clip(gh_sd, 0.0, 0.5)); self.dispersion_gamma_sd_a = float(np.clip(ga_sd, 0.0, 0.5))
            except Exception:
                self.dispersion_gamma_h = self.dispersion_gamma_a = 0.0
                self.dispersion_gamma_sd_h = self.dispersion_gamma_sd_a = 0.0
        else:
            self.phi_h = self.phi_a = 1e6
            self.dispersion_gamma_h = self.dispersion_gamma_a = 0.0
            self.dispersion_gamma_sd_h = self.dispersion_gamma_sd_a = 0.0
            self.log_lambda_total_mean = 0.0
        self.fitted = True
        self.needs_refit = False
        return {"converged": bool(res.success), "n": int(len(x)),
                "teams": self.n_t, "leagues": self.n_l,
                "features": len(self.active_features),
                "evidence_features": len([f for f in self.active_features if f in self.evidence_feature_set]),
                "phi_h": self.phi_h, "phi_a": self.phi_a, "iters": int(res.nit)}

    @staticmethod
    def _gamma_glm_fit(log_lams, resid2):
        X = np.column_stack([np.ones_like(log_lams), log_lams - log_lams.mean()])
        y = np.maximum(resid2, 1e-6); coef = np.zeros(2)
        for _ in range(8):
            eta = X @ coef; mu_hat = np.exp(np.clip(eta, -8, 4))
            W = np.maximum(mu_hat ** 2, 1e-6)
            z = eta + (y - mu_hat) / np.maximum(mu_hat, 1e-6)
            XW = X * W[:, None]
            try: coef_new = np.linalg.solve(X.T @ XW + 1e-6 * np.eye(2), XW.T @ z)
            except np.linalg.LinAlgError: break
            if np.max(np.abs(coef_new - coef)) < 1e-8: coef = coef_new; break
            coef = coef_new
        try:
            eta = X @ coef; mu_hat = np.exp(np.clip(eta, -8, 4))
            W = np.maximum(mu_hat ** 2, 1e-6)
            XWX = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
            inv = np.linalg.inv(XWX); e = y - mu_hat
            meat = X.T @ ((W ** 2 * e ** 2)[:, None] * X)
            cov = inv @ meat @ inv
            return float(coef[1]), float(math.sqrt(max(cov[1, 1], 0.0)))
        except Exception:
            return float(coef[1]), 0.0

    def _unpack(self, theta):
        b = self.design_blocks
        return {"ha_global": float(theta[b["ha_global"]][0]), "kappa": theta[b["kappa"]],
                "ha_l": theta[b["ha_l"]], "attack": theta[b["attack"]],
                "defence": theta[b["defence"]], "beta_h": theta[b["beta_h"]], "beta_a": theta[b["beta_a"]]}

    def _indices(self, x):
        lg = x.league.map(self.leagues).fillna(-1).astype(int).to_numpy()
        hi = x.home.map(self.teams).fillna(-1).astype(int).to_numpy()
        ai = x.away.map(self.teams).fillna(-1).astype(int).to_numpy()
        neutral = x["is_neutral"].to_numpy(float) if "is_neutral" in x.columns else np.zeros(len(x))
        X = self._feature_matrix(x)
        return lg, hi, ai, neutral, X

    def _pooled_from_theta(self, attack, defence):
        n_t = attack.shape[-1]
        w = (self.team_games / (self.team_games + self.cfg.pool_kappa)) if self.team_games.size == n_t else np.zeros(n_t)
        pa = self.team_prior_attack if self.team_prior_attack.size == n_t else np.full(n_t, self.attack_mean)
        pd_ = self.team_prior_defence if self.team_prior_defence.size == n_t else np.full(n_t, self.defence_mean)
        return w * attack + (1 - w) * pa, w * defence + (1 - w) * pd_

    def predict_intensity_precomp(self, lg, hi, ai, neutral, X, theta):
        u = self._unpack(theta); lg_ok = lg >= 0
        n_l = max(self.n_l, 1); n_t = max(self.n_t, 1)
        kappa_eff = np.where(lg_ok, u["kappa"][np.clip(lg, 0, n_l - 1)], self.kappa_mean)
        ha_l_eff = np.where(lg_ok, u["ha_l"][np.clip(lg, 0, n_l - 1)], self.ha_l_mean)
        pooled_a, pooled_d = self._pooled_from_theta(u["attack"], u["defence"])
        lg_clip = np.clip(lg, 0, n_l - 1)
        lpa = self.league_prior_attack[lg_clip] if self.league_prior_attack.size > 0 else np.full(len(lg), self.attack_mean)
        lpd = self.league_prior_defence[lg_clip] if self.league_prior_defence.size > 0 else np.full(len(lg), self.defence_mean)
        lpa = np.where(~lg_ok, self.attack_mean, lpa); lpd = np.where(~lg_ok, self.defence_mean, lpd)
        hc = np.clip(hi, 0, n_t - 1); ac = np.clip(ai, 0, n_t - 1)
        A_h = np.where(hi >= 0, pooled_a[hc], lpa)
        D_a = np.where(ai >= 0, pooled_d[ac], lpd)
        A_a = np.where(ai >= 0, pooled_a[ac], lpa)
        D_h = np.where(hi >= 0, pooled_d[hc], lpd)
        nm = (neutral > 0.5).astype(float)
        ha_eff = ha_l_eff * (1 - nm); ha_global = u["ha_global"] * (1 - nm)
        Xh = (X @ u["beta_h"]) if self.p else np.zeros(len(lg))
        Xa = (X @ u["beta_a"]) if self.p else np.zeros(len(lg))
        eta_h = ha_global + kappa_eff + 0.5 * ha_eff + A_h - D_a + Xh
        eta_a = kappa_eff - 0.5 * ha_eff + A_a - D_h + Xa
        cold_h = (hi < 0).astype(float); cold_a = (ai < 0).astype(float)
        return (np.exp(np.clip(eta_h, self.cfg.eta_lo, self.cfg.eta_hi)),
                np.exp(np.clip(eta_a, self.cfg.eta_lo, self.cfg.eta_hi)), cold_h, cold_a)

    def predict_intensity_batch(self, lg, hi, ai, neutral, X, theta_batch):
        S = theta_batch.shape[0]
        b = self.design_blocks
        n_l = max(self.n_l, 1); n_t = max(self.n_t, 1)
        ha_global = theta_batch[:, b["ha_global"].start]
        kappa = theta_batch[:, b["kappa"]]
        ha_l = theta_batch[:, b["ha_l"]]
        attack = theta_batch[:, b["attack"]]
        defence = theta_batch[:, b["defence"]]
        beta_h = theta_batch[:, b["beta_h"]]
        beta_a = theta_batch[:, b["beta_a"]]
        n_t_cur = attack.shape[1]
        w = (self.team_games / (self.team_games + self.cfg.pool_kappa)) if self.team_games.size == n_t_cur else np.zeros(n_t_cur)
        pa = self.team_prior_attack if self.team_prior_attack.size == n_t_cur else np.full(n_t_cur, self.attack_mean)
        pd_ = self.team_prior_defence if self.team_prior_defence.size == n_t_cur else np.full(n_t_cur, self.defence_mean)
        pooled_a = attack * w[None, :] + pa[None, :] * (1 - w[None, :])
        pooled_d = defence * w[None, :] + pd_[None, :] * (1 - w[None, :])
        lg_ok = lg >= 0
        lg_clip = np.clip(lg, 0, n_l - 1)
        kappa_eff = np.where(lg_ok[None, :], kappa[:, lg_clip], self.kappa_mean)
        ha_l_eff = np.where(lg_ok[None, :], ha_l[:, lg_clip], self.ha_l_mean)
        lpa_arr = self.league_prior_attack[lg_clip] if self.league_prior_attack.size > 0 else np.full(len(lg), self.attack_mean)
        lpd_arr = self.league_prior_defence[lg_clip] if self.league_prior_defence.size > 0 else np.full(len(lg), self.defence_mean)
        lpa_arr = np.where(lg_ok, lpa_arr, self.attack_mean)[None, :]
        lpd_arr = np.where(lg_ok, lpd_arr, self.defence_mean)[None, :]
        hc = np.clip(hi, 0, n_t - 1); ac = np.clip(ai, 0, n_t - 1)
        A_h = np.where((hi >= 0)[None, :], pooled_a[:, hc], lpa_arr)
        D_a = np.where((ai >= 0)[None, :], pooled_d[:, ac], lpd_arr)
        A_a = np.where((ai >= 0)[None, :], pooled_a[:, ac], lpa_arr)
        D_h = np.where((hi >= 0)[None, :], pooled_d[:, hc], lpd_arr)
        nm = (neutral > 0.5).astype(float)[None, :]
        ha_global_eff = ha_global[:, None] * (1 - nm)
        ha_l_eff = ha_l_eff * (1 - nm)
        Xh = (X @ beta_h.T).T if self.p else np.zeros((S, len(lg)))
        Xa = (X @ beta_a.T).T if self.p else np.zeros((S, len(lg)))
        eta_h = ha_global_eff + kappa_eff + 0.5 * ha_l_eff + A_h - D_a + Xh
        eta_a = kappa_eff - 0.5 * ha_l_eff + A_a - D_h + Xa
        lam_h = np.exp(np.clip(eta_h, self.cfg.eta_lo, self.cfg.eta_hi))
        lam_a = np.exp(np.clip(eta_a, self.cfg.eta_lo, self.cfg.eta_hi))
        cold_h = np.broadcast_to((hi < 0).astype(float)[None, :], (S, len(lg)))
        cold_a = np.broadcast_to((ai < 0).astype(float)[None, :], (S, len(lg)))
        return lam_h, lam_a, cold_h, cold_a

    def predict_intensity(self, df, theta=None):
        lg, hi, ai, neutral, X = self._indices(df)
        return self.predict_intensity_precomp(lg, hi, ai, neutral, X, self.theta if theta is None else theta)

    def phi_h_vec(self, lams):
        lams = np.maximum(np.asarray(lams, float), EPS)
        if not self.cfg.use_nb: return np.full_like(lams, 1e6)
        if abs(self.dispersion_gamma_h) < 1e-9: return np.full_like(lams, self.phi_h)
        log_lam = np.log(lams)
        return np.clip(self.phi_h * np.exp(self.dispersion_gamma_h * (log_lam - self.log_lambda_total_mean)),
                       self.cfg.nb_phi_min, self.cfg.nb_phi_max)

    def phi_a_vec(self, lams):
        lams = np.maximum(np.asarray(lams, float), EPS)
        if not self.cfg.use_nb: return np.full_like(lams, 1e6)
        if abs(self.dispersion_gamma_a) < 1e-9: return np.full_like(lams, self.phi_a)
        log_lam = np.log(lams)
        return np.clip(self.phi_a * np.exp(self.dispersion_gamma_a * (log_lam - self.log_lambda_total_mean)),
                       self.cfg.nb_phi_min, self.cfg.nb_phi_max)

    def phi_h_for(self, lam_h): return float(self.phi_h_vec(np.array([lam_h]))[0])
    def phi_a_for(self, lam_a): return float(self.phi_a_vec(np.array([lam_a]))[0])

    def _joint_matrix(self, lh, la, T_score, family, theta_c, bp_l3, joint, rho, use_dc,
                       phi_h_override=None, phi_a_override=None):
        ph = phi_h_override if phi_h_override is not None else self.phi_h_for(lh)
        pa = phi_a_override if phi_a_override is not None else self.phi_a_for(la)
        if joint in ("copula", "bp"):
            if use_dc and rho != 0.0 and not self._dc_copula_warned:
                log.warning("[joint] Dixon-Coles ignored when joint_type ∈ {copula, bp}; "
                            "use joint_type='none' with use_dc=True if DC adjustment is required")
                self._dc_copula_warned = True
            use_dc = False; rho = 0.0
        if joint == "bp" and bp_l3 > 0:
            J = bivariate_poisson_surface(lh, la, bp_l3, self.cfg.score_cap)
        elif joint == "copula" and abs(theta_c) > 1e-8:
            mh = nb_pmf(lh, ph, self.cfg.score_cap) if self.cfg.use_nb else poisson_pmf(lh, self.cfg.score_cap)
            ma = nb_pmf(la, pa, self.cfg.score_cap) if self.cfg.use_nb else poisson_pmf(la, self.cfg.score_cap)
            J = copula_joint(mh, ma, theta_c, family)
        else:
            J = nb_surface(lh, la, ph, pa, self.cfg.score_cap) if self.cfg.use_nb \
                else np.outer(poisson_pmf(lh, self.cfg.score_cap), poisson_pmf(la, self.cfg.score_cap))
            if use_dc and rho != 0.0: J = apply_dc(J, lh, la, rho)
        return score_temperature(J, T_score)

    def predict_1x2(self, df, T_score=1.0, theta=None, family="frank", theta_c=0.0, bp_l3=0.0,
                    joint="copula", use_dc=False, rho=0.0):
        lh, la, _, _ = self.predict_intensity(df, theta=theta)
        cap = self.cfg.score_cap
        ph_arr = self.phi_h_vec(lh); pa_arr = self.phi_a_vec(la)
        if joint in ("copula", "bp"): use_dc = False; rho = 0.0
        p = _batch_joint_1x2(lh[None, :], la[None, :], T_score, family, theta_c, bp_l3,
                              joint, rho, use_dc, ph_arr[None, :], pa_arr[None, :], cap)[0]
        return clip_matrix(p)

    def posterior_draws(self, n_samples, rng=None):
        rng = rng or np.random.default_rng(MASTER_SEED)
        z = rng.standard_normal((n_samples, self.n_params))
        return self.theta[None, :] + z @ self.theta_chol.T

    def posterior_phi_draws(self, n_samples, rng=None):
        rng = rng or np.random.default_rng(MASTER_SEED)
        if not self.cfg.phi_uncertainty:
            return np.full(n_samples, self.phi_h), np.full(n_samples, self.phi_a)
        n_h = max(self.n_h_obs, 20); n_a = max(self.n_a_obs, 20)
        ph = rng.gamma(shape=n_h / 2.0, scale=2.0 * self.phi_h / n_h, size=n_samples)
        pa = rng.gamma(shape=n_a / 2.0, scale=2.0 * self.phi_a / n_a, size=n_samples)
        return np.clip(ph, self.cfg.nb_phi_min, self.cfg.nb_phi_max), np.clip(pa, self.cfg.nb_phi_min, self.cfg.nb_phi_max)

    def posterior_gamma_draws(self, n_samples, rng=None):
        rng = rng or np.random.default_rng(MASTER_SEED + 7)
        if not self.cfg.gamma_uncertainty or self.dispersion_gamma_sd_h <= 0:
            gh = np.full(n_samples, self.dispersion_gamma_h)
        else:
            gh = np.clip(rng.normal(self.dispersion_gamma_h, self.dispersion_gamma_sd_h, size=n_samples), -0.5, 0.5)
        if not self.cfg.gamma_uncertainty or self.dispersion_gamma_sd_a <= 0:
            ga = np.full(n_samples, self.dispersion_gamma_a)
        else:
            ga = np.clip(rng.normal(self.dispersion_gamma_a, self.dispersion_gamma_sd_a, size=n_samples), -0.5, 0.5)
        return gh, ga

    def state(self):
        return {"cfg": asdict(self.cfg), "teams": self.teams, "leagues": self.leagues,
                "active_features": self.active_features,
                "feature_mean": self.feature_mean.tolist(), "feature_scale": self.feature_scale.tolist(),
                "theta": self.theta.tolist(), "theta_cov": self.theta_cov.tolist(),
                "theta_chol": self.theta_chol.tolist(),
                "phi_h": self.phi_h, "phi_a": self.phi_a, "n_h_obs": self.n_h_obs, "n_a_obs": self.n_a_obs,
                "log_lambda_total_mean": self.log_lambda_total_mean,
                "dispersion_gamma_h": self.dispersion_gamma_h, "dispersion_gamma_a": self.dispersion_gamma_a,
                "dispersion_gamma_sd_h": self.dispersion_gamma_sd_h, "dispersion_gamma_sd_a": self.dispersion_gamma_sd_a,
                "n_l": self.n_l, "n_t": self.n_t, "p": self.p, "n_params": self.n_params,
                "design_blocks": {k: [s.start, s.stop] for k, s in self.design_blocks.items()},
                "kappa_mean": self.kappa_mean, "ha_l_mean": self.ha_l_mean,
                "attack_mean": self.attack_mean, "defence_mean": self.defence_mean,
                "league_attack_mean": {int(k): float(v) for k, v in self.league_attack_mean.items()},
                "league_defence_mean": {int(k): float(v) for k, v in self.league_defence_mean.items()},
                "team_league": {int(k): int(v) for k, v in self.team_league.items()},
                "team_games": self.team_games.tolist(),
                "team_prior_attack": self.team_prior_attack.tolist(),
                "team_prior_defence": self.team_prior_defence.tolist(),
                "league_prior_attack": self.league_prior_attack.tolist(),
                "league_prior_defence": self.league_prior_defence.tolist(),
                "needs_refit": bool(self.needs_refit),
                "fitted": self.fitted}

    @classmethod
    def restore(cls, s):
        obj = cls(MaherConfig(**s["cfg"]))
        obj.teams = {str(k): int(v) for k, v in s["teams"].items()}
        obj.leagues = {str(k): int(v) for k, v in s["leagues"].items()}
        obj.active_features = list(s["active_features"])
        obj.feature_mean = np.asarray(s["feature_mean"], float); obj.feature_scale = np.asarray(s["feature_scale"], float)
        obj.theta = np.asarray(s["theta"], float); obj.theta_cov = np.asarray(s["theta_cov"], float)
        obj.theta_chol = np.asarray(s["theta_chol"], float)
        obj.phi_h = float(s.get("phi_h", 8.0)); obj.phi_a = float(s.get("phi_a", 8.0))
        obj.n_h_obs = int(s.get("n_h_obs", 0)); obj.n_a_obs = int(s.get("n_a_obs", 0))
        obj.log_lambda_total_mean = float(s.get("log_lambda_total_mean", 0.0))
        obj.dispersion_gamma_h = float(s.get("dispersion_gamma_h", 0.0)); obj.dispersion_gamma_a = float(s.get("dispersion_gamma_a", 0.0))
        obj.dispersion_gamma_sd_h = float(s.get("dispersion_gamma_sd_h", 0.0)); obj.dispersion_gamma_sd_a = float(s.get("dispersion_gamma_sd_a", 0.0))
        obj.n_l = int(s["n_l"]); obj.n_t = int(s["n_t"]); obj.p = int(s["p"]); obj.n_params = int(s["n_params"])
        obj.design_blocks = {k: slice(int(v[0]), int(v[1])) for k, v in s["design_blocks"].items()}
        obj.kappa_mean = float(s.get("kappa_mean", 0.0)); obj.ha_l_mean = float(s.get("ha_l_mean", 0.0))
        obj.attack_mean = float(s.get("attack_mean", 0.0)); obj.defence_mean = float(s.get("defence_mean", 0.0))
        obj.league_attack_mean = {int(k): float(v) for k, v in s.get("league_attack_mean", {}).items()}
        obj.league_defence_mean = {int(k): float(v) for k, v in s.get("league_defence_mean", {}).items()}
        obj.team_league = {int(k): int(v) for k, v in s.get("team_league", {}).items()}
        obj.team_games = np.asarray(s.get("team_games", []), float)
        obj.team_prior_attack = np.asarray(s.get("team_prior_attack", []), float)
        obj.team_prior_defence = np.asarray(s.get("team_prior_defence", []), float)
        obj.league_prior_attack = np.asarray(s.get("league_prior_attack", []), float)
        obj.league_prior_defence = np.asarray(s.get("league_prior_defence", []), float)
        obj.needs_refit = bool(s.get("needs_refit", False))
        obj.fitted = bool(s.get("fitted", True)); return obj

    def expand_teams(self, new_teams, league_of):
        new_teams = [str(t) for t in new_teams if str(t) not in self.teams]
        if not new_teams: return {"added": 0}
        n_add = len(new_teams); n_t_old = self.n_t; n_t_new = n_t_old + n_add
        p = self.p; n_l = self.n_l; b = self.design_blocks
        off_ha_global = 0; off_kappa = 1; off_ha_l = 1 + n_l
        off_attack = 1 + 2 * n_l; off_defence = off_attack + n_t_new
        off_beta_h = off_defence + n_t_new; off_beta_a = off_beta_h + p
        n_params_new = off_beta_a + p
        theta_new = np.zeros(n_params_new)
        theta_new[off_ha_global] = self.theta[b["ha_global"].start]
        theta_new[off_kappa:off_kappa + n_l] = self.theta[b["kappa"]]
        theta_new[off_ha_l:off_ha_l + n_l] = self.theta[b["ha_l"]]
        theta_new[off_attack:off_attack + n_t_old] = self.theta[b["attack"]]
        theta_new[off_defence:off_defence + n_t_old] = self.theta[b["defence"]]
        theta_new[off_beta_h:off_beta_h + p] = self.theta[b["beta_h"]]
        theta_new[off_beta_a:off_beta_a + p] = self.theta[b["beta_a"]]
        for i, t in enumerate(new_teams):
            lg_name = league_of.get(str(t), None)
            li = self.leagues.get(str(lg_name), -1) if lg_name is not None else -1
            att_prior = self.league_attack_mean.get(li, self.attack_mean) if li >= 0 else self.attack_mean
            dfn_prior = self.league_defence_mean.get(li, self.defence_mean) if li >= 0 else self.defence_mean
            theta_new[off_attack + n_t_old + i] = att_prior
            theta_new[off_defence + n_t_old + i] = dfn_prior
        cov_new = np.eye(n_params_new) * 0.05
        cov_new[off_ha_global:off_ha_global + 1, off_ha_global:off_ha_global + 1] = self.theta_cov[b["ha_global"], b["ha_global"]]
        cov_new[off_kappa:off_kappa + n_l, off_kappa:off_kappa + n_l] = self.theta_cov[b["kappa"], b["kappa"]]
        cov_new[off_ha_l:off_ha_l + n_l, off_ha_l:off_ha_l + n_l] = self.theta_cov[b["ha_l"], b["ha_l"]]
        cov_new[off_attack:off_attack + n_t_old, off_attack:off_attack + n_t_old] = self.theta_cov[b["attack"], b["attack"]]
        cov_new[off_defence:off_defence + n_t_old, off_defence:off_defence + n_t_old] = self.theta_cov[b["defence"], b["defence"]]
        cov_new[off_beta_h:off_beta_h + p, off_beta_h:off_beta_h + p] = self.theta_cov[b["beta_h"], b["beta_h"]]
        cov_new[off_beta_a:off_beta_a + p, off_beta_a:off_beta_a + p] = self.theta_cov[b["beta_a"], b["beta_a"]]
        self.theta = theta_new; self.theta_cov = cov_new
        try: self.theta_chol = np.linalg.cholesky(cov_new + 1e-10 * np.eye(n_params_new))
        except np.linalg.LinAlgError: self.theta_chol = np.eye(n_params_new) * 1e-3
        self.n_t = n_t_new; self.n_params = n_params_new
        self.design_blocks = {
            "ha_global": slice(off_ha_global, off_ha_global + 1),
            "kappa": slice(off_kappa, off_kappa + n_l),
            "ha_l": slice(off_ha_l, off_ha_l + n_l),
            "attack": slice(off_attack, off_attack + n_t_new),
            "defence": slice(off_defence, off_defence + n_t_new),
            "beta_h": slice(off_beta_h, off_beta_h + p),
            "beta_a": slice(off_beta_a, off_beta_a + p)}
        for i, t in enumerate(new_teams):
            ti = n_t_old + i
            self.teams[str(t)] = ti
            lg_name = league_of.get(str(t), None)
            li = self.leagues.get(str(lg_name), -1) if lg_name is not None else -1
            self.team_league[ti] = li
            att_prior = self.league_attack_mean.get(li, self.attack_mean) if li >= 0 else self.attack_mean
            dfn_prior = self.league_defence_mean.get(li, self.defence_mean) if li >= 0 else self.defence_mean
            self.team_prior_attack = np.append(self.team_prior_attack, att_prior)
            self.team_prior_defence = np.append(self.team_prior_defence, dfn_prior)
            self.team_games = np.append(self.team_games, 0.0)
        self.needs_refit = True
        return {"added": n_add}


def _batch_joint_1x2(lam_h, lam_a, T_score, family, theta_c, bp_l3, joint, rho, use_dc,
                     phi_h_arr, phi_a_arr, cap, max_elems=BATCH_JOINT_MAX_ELEMS):
    if joint in ("copula", "bp"): use_dc = False; rho = 0.0
    S, n = lam_h.shape
    total = S * n
    out = np.empty((S, n, 3), dtype=float)
    rows_per_chunk = max(1, int(max_elems) // ((cap + 1) * (cap + 1)))
    flat_h = np.asarray(lam_h, float).reshape(-1)
    flat_a = np.asarray(lam_a, float).reshape(-1)
    flat_ph = np.asarray(phi_h_arr, float).reshape(-1)
    flat_pa = np.asarray(phi_a_arr, float).reshape(-1)
    for start in range(0, total, rows_per_chunk):
        end = min(total, start + rows_per_chunk)
        fh = flat_h[start:end]; fa = flat_a[start:end]
        fph = flat_ph[start:end]; fpa = flat_pa[start:end]
        N = end - start
        if joint == "bp" and bp_l3 > 0:
            J = _batch_bp_joint(fh, fa, float(bp_l3), cap)
        elif joint == "copula" and abs(theta_c) > 1e-8:
            PH = _batch_nb_pmf_var(fh, fph, cap); PA = _batch_nb_pmf_var(fa, fpa, cap)
            Fh = np.concatenate([np.zeros((N, 1)), np.cumsum(PH, axis=1)], axis=1)
            Fa = np.concatenate([np.zeros((N, 1)), np.cumsum(PA, axis=1)], axis=1)
            J = copula_joint_vec(Fh, Fa, np.full(N, float(theta_c)), family)
        else:
            PH = _batch_nb_pmf_var(fh, fph, cap); PA = _batch_nb_pmf_var(fa, fpa, cap)
            J = PH[:, :, None] * PA[:, None, :]
            if use_dc and rho != 0.0: J = _batch_apply_dc(J, fh, fa, float(rho))
        if abs(T_score - 1.0) > 1e-9:
            logJ = np.log(np.clip(J, EPS, None)) / float(T_score)
            logJ -= logsumexp(logJ, axis=(1, 2), keepdims=True)
            J = np.exp(logJ)
        else:
            s = J.sum(axis=(1, 2), keepdims=True)
            J = J / np.maximum(s, EPS)
        out.reshape(-1, 3)[start:end] = outcome_from_batch(J)
    return out

# ============================================================================
# 9. Hierarchical dynamic goal state
# ============================================================================

@dataclass
class HierStateConfig:
    enabled: bool = True
    process_sd: float = 0.07
    observation_sd: float = 0.40
    prior_sd: float = 0.35
    prior_dispersion: float = 8.0
    use_fitted_dispersion: bool = True
    min_games_for_activation: int = 8
    max_abs_state: float = 1.50
    n_mc_samples: int = 64
    refresh_every_n_assimilations: int = 0

class HierarchicalDynamicGoalState:
    def __init__(self, cfg=None, structural_cfg=None):
        self.cfg = cfg or HierStateConfig()
        self.scfg = structural_cfg or MaherConfig()
        self.states = {}; self.home_adv = defaultdict(lambda: 0.17)
        self.global_log_goal = math.log(1.30); self.fitted = False; self.fit_cutoff = None
        self._phi_h = float(self.cfg.prior_dispersion)
        self._phi_a = float(self.cfg.prior_dispersion)
    def _key(self, comp, team): return (str(comp), str(team))
    def _state(self, comp, team):
        k = self._key(comp, team)
        if k not in self.states:
            self.states[k] = {"attack": 0.0, "defence": 0.0,
                              "attack_var": self.cfg.prior_sd ** 2, "defence_var": self.cfg.prior_sd ** 2,
                              "games": 0.0, "last_date": None}
        return self.states[k]
    def _decay(self, st, date):
        last = st.get("last_date")
        if isinstance(last, pd.Timestamp):
            dt = max((date - last).total_seconds() / 86400.0, 0.0)
            if dt > 0:
                q = self.cfg.process_sd ** 2 * min(dt / 30.0, 4.0)
                cap = self.cfg.prior_sd ** 2 * 9.0
                st["attack_var"] = min(st["attack_var"] + q, cap)
                st["defence_var"] = min(st["defence_var"] + q, cap)
    @staticmethod
    def _kalman(mean, var, resid, obs_var, sign=1.0):
        obs_var = max(obs_var, 0.02)
        gain = var / max(var + obs_var, EPS)
        updated = mean + sign * gain * resid
        post = max((1 - gain) * var, 0.001)
        return float(np.clip(updated, -1.5, 1.5)), float(post)
    def _observation_var(self, lam, side="home"):
        """Delta-method variance of log(Y + ε) for Y ~ NB(λ, φ):
        Var(log Y) ≈ 1/λ + 1/φ. Uses the fitted φ from the Maher layer when available."""
        lam = max(float(lam), 0.05)
        if self.cfg.use_fitted_dispersion:
            phi = self._phi_h if side == "home" else self._phi_a
            phi = float(np.clip(phi, 1.0, 500.0))
        else:
            phi = max(float(self.cfg.prior_dispersion), 1.0)
        return 1.0 / lam + 1.0 / phi
    def fit(self, frame, cutoff=None, phi_h=None, phi_a=None):
        if phi_h is not None: self._phi_h = float(phi_h)
        if phi_a is not None: self._phi_a = float(phi_a)
        if frame is None or frame.empty: self.fitted = False; return {"fitted": False, "n": 0}
        x = frame.copy(); x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x[x["date"].notna()].sort_values("date", kind="mergesort")
        c = to_ts(cutoff) if cutoff is not None else x["date"].max()
        x = x[x["date"] < c].copy()
        if len(x) < 100: self.fitted = False; return {"fitted": False, "n": int(len(x))}
        self.states = {}; self.home_adv = defaultdict(lambda: 0.17)
        global_goal = float(np.clip(pd.concat([x["home_goals"], x["away_goals"]]).mean(), 0.2, 4.0))
        self.global_log_goal = math.log(global_goal)
        for _, r in x.iterrows():
            comp = str(r.get("competition", r.get("league", "UNKNOWN"))); date = r["date"]
            home = str(r.get("home_team", r.get("home"))); away = str(r.get("away_team", r.get("away")))
            hs = self._state(comp, home); aas = self._state(comp, away)
            self._decay(hs, date); self._decay(aas, date)
            lh = math.exp(np.clip(self.global_log_goal + self.home_adv[comp] + hs["attack"] - aas["defence"],
                                  math.log(0.05), math.log(6.0)))
            la = math.exp(np.clip(self.global_log_goal + aas["attack"] - hs["defence"],
                                  math.log(0.05), math.log(6.0)))
            yh = math.log((max(to_num(r.get("home_goals")), 0.0) + 0.25) / max(global_goal + 0.25, EPS))
            ya = math.log((max(to_num(r.get("away_goals")), 0.0) + 0.25) / max(global_goal + 0.25, EPS))
            rh = float(yh - math.log(max(lh, EPS))); ra = float(ya - math.log(max(la, EPS)))
            oh = self._observation_var(lh, side="home"); oa = self._observation_var(la, side="away")
            hs["attack"], hs["attack_var"] = self._kalman(hs["attack"], hs["attack_var"], rh, oh, +1.0)
            aas["defence"], aas["defence_var"] = self._kalman(aas["defence"], aas["defence_var"], rh, oh, -1.0)
            aas["attack"], aas["attack_var"] = self._kalman(aas["attack"], aas["attack_var"], ra, oa, +1.0)
            hs["defence"], hs["defence_var"] = self._kalman(hs["defence"], hs["defence_var"], ra, oa, -1.0)
            actual_diff = yh - ya; expected_diff = math.log(max(lh, EPS)) - math.log(max(la, EPS))
            ha_gain = 0.015 / math.sqrt(max(hs["games"] + aas["games"] + 2.0, 1.0))
            self.home_adv[comp] = float(np.clip(self.home_adv[comp] + ha_gain * (actual_diff - expected_diff), -0.5, 0.5))
            hs["games"] += 1.0; aas["games"] += 1.0; hs["last_date"] = date; aas["last_date"] = date
            global_goal = 0.995 * global_goal + 0.005 * ((to_num(r.get("home_goals")) + to_num(r.get("away_goals"))) / 2.0)
        self.global_log_goal = math.log(max(global_goal, 0.20)); self.fitted = True
        self.fit_cutoff = c.isoformat()
        return {"fitted": True, "n": int(len(x)), "teams": len(self.states),
                "phi_h": self._phi_h, "phi_a": self._phi_a}
    def predict_mc(self, frame, n_samples=None, rng=None):
        n = len(frame)
        if not self.fitted or n == 0:
            S = int(n_samples or self.cfg.n_mc_samples)
            z = np.full((S, n), 0.05)
            return z, z.copy()
        rng = rng or np.random.default_rng(MASTER_SEED + 11)
        S = int(n_samples or self.cfg.n_mc_samples)
        lam_h = np.zeros((S, n)); lam_a = np.zeros((S, n))
        for i, (_, r) in enumerate(frame.iterrows()):
            comp = str(r.get("competition", r.get("league", "UNKNOWN")))
            home = str(r.get("home_team", r.get("home"))); away = str(r.get("away_team", r.get("away")))
            hs = self._state(comp, home); aas = self._state(comp, away)
            av_h = max(hs["attack_var"], 1e-6); dv_h = max(hs["defence_var"], 1e-6)
            av_a = max(aas["attack_var"], 1e-6); dv_a = max(aas["defence_var"], 1e-6)
            ah_s = rng.normal(hs["attack"], math.sqrt(av_h), size=S)
            dh_s = rng.normal(hs["defence"], math.sqrt(dv_h), size=S)
            aa_s = rng.normal(aas["attack"], math.sqrt(av_a), size=S)
            da_s = rng.normal(aas["defence"], math.sqrt(dv_a), size=S)
            ha_sd = max(self.cfg.process_sd / math.sqrt(max(hs["games"] + aas["games"] + 2.0, 1.0)), 1e-4)
            ha_s = rng.normal(self.home_adv[comp], ha_sd, size=S)
            log_lh = self.global_log_goal + ha_s + ah_s - da_s
            log_la = self.global_log_goal + aa_s - dh_s
            var_lh = av_h + dv_a + ha_sd ** 2
            var_la = av_a + dv_h
            log_lh -= 0.5 * var_lh
            log_la -= 0.5 * var_la
            lam_h[:, i] = np.exp(np.clip(log_lh, math.log(0.05), math.log(6.0)))
            lam_a[:, i] = np.exp(np.clip(log_la, math.log(0.05), math.log(6.0)))
        return lam_h, lam_a
    def probabilities(self, frame, phi_h=None, phi_a=None, cap=None, n_samples=None):
        phi_h = float(phi_h) if phi_h is not None else float(self._phi_h)
        phi_a = float(phi_a) if phi_a is not None else float(self._phi_a)
        cap = int(cap) if cap is not None else 12
        lam_h, lam_a = self.predict_mc(frame, n_samples=n_samples)
        S = lam_h.shape[0]; n = lam_h.shape[1]
        p_draws = _batch_joint_1x2(lam_h, lam_a, 1.0, "none", 0.0, 0.0, "none", 0.0, False,
                                    np.full((S, n), phi_h), np.full((S, n), phi_a), cap)
        return clip_matrix(p_draws.mean(axis=0))
    def state(self):
        return {"fitted": self.fitted, "n_teams": len(self.states),
                "global_log_goal": self.global_log_goal,
                "home_adv": {str(k): float(v) for k, v in self.home_adv.items()},
                "fit_cutoff": self.fit_cutoff,
                "phi_h": float(self._phi_h), "phi_a": float(self._phi_a),
                "states": {f"{k[0]}|{k[1]}": v for k, v in self.states.items()}}
    @classmethod
    def restore(cls, s):
        obj = cls()
        obj.global_log_goal = float(s.get("global_log_goal", math.log(1.3)))
        obj.home_adv = defaultdict(lambda: 0.17, {str(k): float(v) for k, v in s.get("home_adv", {}).items()})
        obj.fitted = bool(s.get("fitted", False)); obj.fit_cutoff = s.get("fit_cutoff")
        obj._phi_h = float(s.get("phi_h", obj.cfg.prior_dispersion))
        obj._phi_a = float(s.get("phi_a", obj.cfg.prior_dispersion))
        for key, v in s.get("states", {}).items():
            if "|" in key:
                comp, team = key.split("|", 1)
                obj.states[(comp, team)] = v
        return obj

# ============================================================================
# 10. Regime-conditional ensemble (with evidence quantile edges)
# ============================================================================

class RegimeConditionalEnsemble:
    def __init__(self, n_buckets=3, freeze_edges=True, evidence_buckets=2):
        self.n_buckets = int(n_buckets)
        self.evidence_buckets = int(evidence_buckets)
        self.edges = None
        self.evidence_edges = None
        self.freeze_edges = bool(freeze_edges)
        self.names = []; self.weights_by_regime = {}
        self.fitted = False
    @staticmethod
    def _softmax(z):
        z = z - np.max(z); e = np.exp(z)
        return e / max(e.sum(), EPS)
    def _evidence_bucket(self, evidence_present):
        if self.evidence_edges is None or len(self.evidence_edges) < 3:
            return int(evidence_present > 0.5)
        b = int(np.searchsorted(self.evidence_edges, evidence_present, side="left")) - 1
        return max(0, min(b, len(self.evidence_edges) - 2))
    def _regime(self, evidence_present, uncertainty):
        if not np.isfinite(uncertainty): uncertainty = 0.0
        ev_b = self._evidence_bucket(evidence_present)
        if self.edges is None: return (ev_b, 0)
        b = int(np.searchsorted(self.edges, uncertainty, side="left")) - 1
        b = max(0, min(b, len(self.edges) - 2))
        return (ev_b, b)
    def fit(self, expert_probs, y, regime_keys, cv_for_edges=None, evidence_for_edges=None,
            min_rows_per_regime=40):
        names = [k for k, v in expert_probs.items() if v is not None and len(v) == len(y)]
        if not names: self.fitted = False; return {"fitted": False, "reason": "no_experts"}
        if self.edges is None and cv_for_edges is not None:
            qs = np.quantile(cv_for_edges, np.linspace(0, 1, self.n_buckets + 1))
            qs[0] = 0.0; qs[-1] = max(qs[-1], qs[0] + 1e-6)
            self.edges = qs
        if self.evidence_edges is None and evidence_for_edges is not None:
            eq = np.quantile(evidence_for_edges, np.linspace(0, 1, self.evidence_buckets + 1))
            eq[0] = min(eq[0], 0.0); eq[-1] = max(eq[-1], eq[0] + 1e-6)
            self.evidence_edges = eq
        P = np.stack([clip_matrix(expert_probs[n]) for n in names], axis=1)
        y = np.asarray(y, int); n = len(y); self.names = names
        regimes = defaultdict(list)
        for i, rk in enumerate(regime_keys): regimes[rk].append(i)
        self.weights_by_regime = {}
        for rk, idxs in regimes.items():
            idx = np.array(idxs, dtype=int)
            if len(idx) < min_rows_per_regime: continue
            P_r = P[idx]; y_r = y[idx]
            def obj(w_raw):
                w = self._softmax(w_raw); p = np.einsum("k,nkc->nc", w, P_r)
                return float(-np.mean(np.log(np.clip(p[np.arange(len(y_r)), y_r], EPS, 1.0))))
            r = minimize(obj, np.zeros(len(names)), method="L-BFGS-B")
            self.weights_by_regime[rk] = self._softmax(r.x if r.success else np.zeros(len(names)))
        def obj_g(w_raw):
            w = self._softmax(w_raw); p = np.einsum("k,nkc->nc", w, P)
            return float(-np.mean(np.log(np.clip(p[np.arange(n), y], EPS, 1.0))))
        rg = minimize(obj_g, np.zeros(len(names)), method="L-BFGS-B")
        self.weights_by_regime[("GLOBAL", 0)] = self._softmax(rg.x if rg.success else np.zeros(len(names)))
        self.fitted = True
        return {"fitted": True, "n": n, "names": names, "n_regimes": len(self.weights_by_regime),
                "edges": self.edges.tolist() if self.edges is not None else None,
                "evidence_edges": self.evidence_edges.tolist() if self.evidence_edges is not None else None}
    def predict(self, expert_probs, regime_keys):
        if not self.fitted or not self.names:
            first = next(iter(expert_probs.values())); return clip_matrix(first)
        n = len(regime_keys)
        P = []
        for nm in self.names:
            p = expert_probs.get(nm)
            if p is None or len(p) != n: p = np.full((n, 3), 1.0 / 3.0)
            P.append(clip_matrix(p))
        A = np.stack(P, axis=1)
        groups = defaultdict(list)
        for i, rk in enumerate(regime_keys): groups[rk].append(i)
        out = np.empty((n, 3))
        for rk, idxs in groups.items():
            idx = np.asarray(idxs, int)
            w = self.weights_by_regime.get(rk, self.weights_by_regime[("GLOBAL", 0)])
            out[idx] = np.einsum("k,nkc->nc", w, A[idx])
        return clip_matrix(out)
    def state(self):
        return {"fitted": self.fitted, "names": self.names, "n_buckets": self.n_buckets,
                "evidence_buckets": self.evidence_buckets, "freeze_edges": self.freeze_edges,
                "edges": None if self.edges is None else list(self.edges),
                "evidence_edges": None if self.evidence_edges is None else list(self.evidence_edges),
                "weights_by_regime": [{"ev": int(k[0]), "bucket": int(k[1]), "w": v.tolist()}
                                       for k, v in self.weights_by_regime.items()]}
    @classmethod
    def restore(cls, s):
        obj = cls(int(s.get("n_buckets", 3)), bool(s.get("freeze_edges", True)),
                  int(s.get("evidence_buckets", 2)))
        obj.names = list(s.get("names", []))
        edges = s.get("edges"); obj.edges = None if edges is None else np.asarray(edges, float)
        eedges = s.get("evidence_edges")
        obj.evidence_edges = None if eedges is None else np.asarray(eedges, float)
        obj.weights_by_regime = {}
        for rec in s.get("weights_by_regime", []):
            obj.weights_by_regime[(int(rec["ev"]), int(rec["bucket"]))] = np.asarray(rec["w"], float)
        obj.fitted = bool(s.get("fitted", False)); return obj

# ============================================================================
# 11. Stacker / challenger / meta
# ============================================================================

class ResidualStacker:
    DEFAULT_COLS = ALL_FEATURES
    def __init__(self, n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=15, subsample=0.8, seed=MASTER_SEED):
        self.n_estimators = int(n_estimators); self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate); self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample); self.seed = int(seed)
        self.cols = list(self.DEFAULT_COLS); self.model = None; self.mu = None; self.sd = None
    def _feats(self, df, fill=None):
        x = ensure_features(df, self.cols, fill=fill); return x[self.cols].to_numpy(float)
    def fit(self, df, y):
        X = self._feats(df); self.mu = np.nanmean(X, axis=0); self.sd = np.nanstd(X, axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        X = np.where(np.isfinite(X), X, self.mu[None, :]); X = (X - self.mu) / self.sd
        self.model = GradientBoostingClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                 learning_rate=self.learning_rate, min_samples_leaf=self.min_samples_leaf,
                                                 subsample=self.subsample, random_state=self.seed).fit(X, y)
        return self
    def predict(self, df):
        if self.model is None: raise RuntimeError("stacker not fitted")
        X = self._feats(df, fill=self.mu); X = np.where(np.isfinite(X), X, self.mu[None, :]); X = (X - self.mu) / self.sd
        P = np.asarray(self.model.predict_proba(X), float)
        if P.shape[1] == 3: return clip_matrix(P)
        full = np.zeros((P.shape[0], 3))
        for k, cls in enumerate(self.model.classes_): full[:, int(cls)] = P[:, k]
        return clip_matrix(full)
    def state(self):
        if self.model is None: return {"fitted": False, "cols": self.cols}
        return {"fitted": True, "cols": self.cols, "n_estimators": self.n_estimators,
                "max_depth": self.max_depth, "learning_rate": self.learning_rate,
                "min_samples_leaf": self.min_samples_leaf, "subsample": self.subsample,
                "seed": self.seed, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "pickled": pickle.dumps(self.model)}
    @classmethod
    def restore(cls, s):
        obj = cls(s.get("n_estimators", 150), s.get("max_depth", 3), s.get("learning_rate", 0.05),
                  s.get("min_samples_leaf", 15), s.get("subsample", 0.8), s.get("seed", MASTER_SEED))
        obj.cols = list(s.get("cols", obj.cols))
        if s.get("fitted"):
            obj.mu = np.asarray(s["mu"], float); obj.sd = np.asarray(s["sd"], float)
            try: obj.model = pickle.loads(s["pickled"])
            except Exception: obj.model = None
        return obj

class OutcomeChallenger:
    def __init__(self, C=0.5):
        self.C = float(C); self.model = None; self.cols = list(ALL_FEATURES); self.mu = None; self.sd = None
    def fit(self, df):
        x = CausalContract.normalize(df, True); x = ensure_features(x, self.cols, fill=None)
        X = x[self.cols].to_numpy(float); y = x.y.to_numpy(int)
        self.mu = np.nanmean(X, axis=0); self.sd = np.nanstd(X, axis=0); self.sd[self.sd < 1e-9] = 1.0
        X = np.where(np.isfinite(X), X, self.mu[None, :]); X = (X - self.mu) / self.sd
        self.model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, random_state=MASTER_SEED).fit(X, y)
        return self
    def predict(self, df):
        if self.model is None: raise RuntimeError("challenger not fitted")
        x = ensure_features(df, self.cols, fill=self.mu); X = x[self.cols].to_numpy(float)
        X = np.where(np.isfinite(X), X, self.mu[None, :]); X = (X - self.mu) / self.sd
        return np.asarray(self.model.predict_proba(X), float)
    def state(self):
        if self.model is None: return {"fitted": False, "C": self.C, "cols": self.cols}
        return {"fitted": True, "C": self.C, "cols": list(self.cols),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist(),
                "classes": self.model.classes_.tolist()}
    @classmethod
    def restore(cls, s):
        obj = cls(float(s.get("C", 0.5))); obj.cols = list(s.get("cols", obj.cols))
        if not s.get("fitted", False): return obj
        obj.mu = np.asarray(s["mu"], float); obj.sd = np.asarray(s["sd"], float)
        m = LogisticRegression(C=obj.C, solver="lbfgs", max_iter=1000, random_state=MASTER_SEED)
        m.classes_ = np.asarray(s["classes"], int); m.coef_ = np.asarray(s["coef"], float)
        m.intercept_ = np.asarray(s["intercept"], float); m.n_features_in_ = int(m.coef_.shape[1])
        m.n_iter_ = np.array([1], dtype=int); obj.model = m; return obj

class StackingMetaLearner:
    def __init__(self, C=1.0):
        self.C = float(C); self.model = None; self.mu = None; self.sd = None
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu = np.nanmean(X, axis=0); self.sd = np.nanstd(X, axis=0); self.sd[self.sd < 1e-9] = 1.0
        Xs = np.where(np.isfinite(X), X, self.mu[None, :]); Xs = (Xs - self.mu) / self.sd
        self.model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000,
                                        random_state=MASTER_SEED).fit(Xs, y)
        return self
    def predict(self, X):
        if self.model is None: raise RuntimeError("meta-learner not fitted")
        X = np.asarray(X, float); Xs = np.where(np.isfinite(X), X, self.mu[None, :]); Xs = (Xs - self.mu) / self.sd
        P = np.asarray(self.model.predict_proba(Xs), float)
        if P.shape[1] == 3: return clip_matrix(P)
        full = np.zeros((P.shape[0], 3))
        for k, cls in enumerate(self.model.classes_): full[:, int(cls)] = P[:, k]
        return clip_matrix(full)
    def state(self):
        if self.model is None: return {"fitted": False, "C": self.C}
        return {"fitted": True, "C": self.C, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist(),
                "classes": self.model.classes_.tolist()}
    @classmethod
    def restore(cls, s):
        obj = cls(float(s.get("C", 1.0)))
        if not s.get("fitted"): return obj
        obj.mu = np.asarray(s["mu"], float); obj.sd = np.asarray(s["sd"], float)
        m = LogisticRegression(C=obj.C, solver="lbfgs", max_iter=1000, random_state=MASTER_SEED)
        m.classes_ = np.asarray(s["classes"], int); m.coef_ = np.asarray(s["coef"], float)
        m.intercept_ = np.asarray(s["intercept"], float); m.n_features_in_ = int(m.coef_.shape[1])
        m.n_iter_ = np.array([1], dtype=int); obj.model = m; return obj

# ============================================================================
# 12. Dirichlet calibrator (with chunked transform)
# ============================================================================

class DirichletCalibrator:
    def __init__(self, C=2.0, min_rows=150, max_elems=DIRICHLET_MAX_ELEMS):
        self.C = float(C); self.min_rows = int(min_rows)
        self.max_elems = int(max_elems)
        self.model = None; self.fitted = False
    def fit(self, p_base, y):
        p = clip_matrix(p_base); y = np.asarray(y, int)
        if len(y) < self.min_rows or len(np.unique(y)) < 3:
            self.fitted = False; return {"fitted": False, "n": int(len(y))}
        X = np.log(np.clip(p, EPS, 1.0))
        self.model = LogisticRegression(C=self.C, max_iter=2500).fit(X, y)
        self.fitted = True
        return {"fitted": True, "n": int(len(y))}
    def transform(self, p_base):
        p = clip_matrix(p_base)
        if not self.fitted or self.model is None: return p
        n = p.shape[0]
        if n <= self.max_elems:
            return clip_matrix(self.model.predict_proba(np.log(np.clip(p, EPS, 1.0))))
        out = np.empty_like(p)
        for start in range(0, n, self.max_elems):
            end = min(n, start + self.max_elems)
            out[start:end] = self.model.predict_proba(np.log(np.clip(p[start:end], EPS, 1.0)))
        return clip_matrix(out)
    def state(self):
        if not self.fitted or self.model is None: return {"fitted": False, "C": self.C}
        return {"fitted": True, "C": self.C, "min_rows": self.min_rows, "max_elems": self.max_elems,
                "coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist(),
                "classes": self.model.classes_.tolist()}
    @classmethod
    def restore(cls, s):
        obj = cls(float(s.get("C", 2.0)), int(s.get("min_rows", 150)), int(s.get("max_elems", DIRICHLET_MAX_ELEMS)))
        if not s.get("fitted"): return obj
        m = LogisticRegression(C=obj.C, max_iter=2500)
        m.classes_ = np.asarray(s["classes"], int); m.coef_ = np.asarray(s["coef"], float)
        m.intercept_ = np.asarray(s["intercept"], float); m.n_features_in_ = int(m.coef_.shape[1])
        m.n_iter_ = np.array([1], dtype=int); obj.model = m; obj.fitted = True
        return obj

# ============================================================================
# 13. Evidence gate
# ============================================================================

@dataclass
class CalibrationState:
    T_score: float = 1.0; T_ou: float = 1.0; T_btts: float = 1.0
    copula_theta: float = 0.0; copula_family: str = "frank"; bp_lam3: float = 0.0
    joint_type: str = "copula"; rho: float = 0.0
    challenger_alpha: float = 0.0; stacker_alpha: float = 0.0; meta_alpha: float = 0.0
    meta_enabled: bool = False
    dirichlet_fitted: bool = False
    cal_base_ll: float = float("nan"); cal_eval_ll: float = float("nan")

class EvidenceGate:
    def __init__(self, split_frac=0.5, improvement_threshold=0.0015, use_dc=False,
                 copula_families=("frank", "clayton", "gumbel"), strict_threshold=0.004):
        self.state = CalibrationState()
        self.split_frac = float(np.clip(split_frac, 0.2, 0.8))
        self.improvement_threshold = float(improvement_threshold)
        self.strict_threshold = float(strict_threshold)
        self.use_dc = bool(use_dc); self.copula_families = tuple(copula_families)
        self.copula_theta = 0.0; self.copula_family = "frank"; self.bp_lam3 = 0.0; self.joint_type = "copula"
    def _score_exact(self, model, cal, T_score, family, theta_c, bp_l3, joint, rho, use_dc):
        lh, la, _, _ = model.predict_intensity(cal); cap = int(model.cfg.score_cap); losses = []
        for h, a, gh, ga in zip(lh, la, cal.home_goals, cal.away_goals):
            J = model._joint_matrix(h, a, T_score, family, theta_c, bp_l3, joint, rho, use_dc)
            losses.append(-math.log(np.clip(J[int(min(max(gh, 0), cap)), int(min(max(ga, 0), cap))], EPS, 1.0)))
        return float(np.mean(losses))
    def _score_binary(self, model, cal, T_score, family, theta_c, bp_l3, joint, rho, use_dc, target):
        lh, la, _, _ = model.predict_intensity(cal)
        cap = int(model.cfg.score_cap); p_yes = []; y_yes = []
        for h, a, gh, ga in zip(lh, la, cal.home_goals, cal.away_goals):
            J = model._joint_matrix(h, a, T_score, family, theta_c, bp_l3, joint, rho, use_dc)
            L = J.shape[0]; ii = np.arange(L)[:, None]; jj = np.arange(L)[None, :]
            if target == "ou25":
                p_yes.append(float(J[ii + jj > 2].sum())); y_yes.append(float((gh + ga) > 2.5))
            else:
                p_yes.append(float(J[(ii >= 1) & (jj >= 1)].sum())); y_yes.append(float((gh >= 1) and (ga >= 1)))
        return np.asarray(p_yes), np.asarray(y_yes)
    def _bootstrap(self, deltas, dates, threshold):
        if len(deltas) < 50: return False
        try: ci = chronological_block_bootstrap(deltas, dates)
        except Exception: return False
        return ci["lo"] > threshold
    def fit(self, model, cal, challenger=None, stacker=None):
        cal = CausalContract.normalize(cal, True)
        n = len(cal); n_fit = max(30, int(n * self.split_frac))
        cal_fit = cal.iloc[:n_fit].reset_index(drop=True)
        cal_eval = cal.iloc[n_fit:].reset_index(drop=True) if n - n_fit >= 20 else cal_fit
        y_fit = cal_fit.y.to_numpy(int); y_eval = cal_eval.y.to_numpy(int)
        base = self._score_exact(model, cal_fit, 1.0, "frank", 0.0, 0.0, "none", 0.0, False)
        best_T = (base, 1.0)
        for T in np.linspace(0.80, 1.30, 13):
            ll = self._score_exact(model, cal_fit, float(T), "frank", 0.0, 0.0, "none", 0.0, False)
            if ll < best_T[0]: best_T = (ll, float(T))
        if base - best_T[0] >= self.improvement_threshold: self.state.T_score = best_T[1]
        lh_c, la_c, _, _ = model.predict_intensity(cal_fit)
        candidates = []
        for fam in self.copula_families:
            try:
                th = fit_copula_theta(lh_c, la_c, cal_fit.home_goals.to_numpy(), cal_fit.away_goals.to_numpy(),
                                       model.phi_h, model.phi_a, model.cfg.score_cap, family=fam)
            except Exception: th = 0.0
            ll = self._score_exact(model, cal_fit, self.state.T_score, fam, th, 0.0, "copula", 0.0, False)
            candidates.append(("copula", fam, th, ll))
        try:
            l3 = fit_bp_lambda3(lh_c, la_c, cal_fit.home_goals.to_numpy(), cal_fit.away_goals.to_numpy(), model.cfg.score_cap)
            ll_bp = self._score_exact(model, cal_fit, self.state.T_score, "frank", 0.0, l3, "bp", 0.0, False)
            candidates.append(("bp", "frank", 0.0, ll_bp))
        except Exception:
            l3 = 0.0; candidates.append(("bp", "frank", 0.0, np.inf))
        ll_ind = self._score_exact(model, cal_fit, self.state.T_score, "frank", 0.0, 0.0, "none", 0.0, False)
        candidates.append(("none", "frank", 0.0, ll_ind))
        best = min(candidates, key=lambda c: c[3])
        if ll_ind - best[3] >= self.improvement_threshold:
            self.joint_type = best[0]; self.copula_family = best[1]
            self.copula_theta = float(best[2]) if best[0] == "copula" else 0.0
            self.bp_lam3 = float(l3) if best[0] == "bp" else 0.0
        else:
            self.joint_type = "none"; self.copula_theta = 0.0; self.bp_lam3 = 0.0
        self.state.joint_type = self.joint_type; self.state.copula_family = self.copula_family
        self.state.copula_theta = float(self.copula_theta); self.state.bp_lam3 = float(self.bp_lam3)
        for target, attr in (("ou25", "T_ou"), ("btts", "T_btts")):
            try:
                p_yes, y_yes = self._score_binary(model, cal_fit, self.state.T_score, self.copula_family,
                                                    self.copula_theta, self.bp_lam3, self.joint_type,
                                                    self.state.rho, self.use_dc, target)
                res = minimize_scalar(lambda t: binary_logloss(_temper_binary(p_yes, t), y_yes),
                                       bounds=(0.7, 1.4), method="bounded")
                if np.isfinite(res.fun) and binary_logloss(p_yes, y_yes) - res.fun >= self.improvement_threshold:
                    setattr(self.state, attr, float(res.x))
            except Exception: pass
        p_struct_fit = model.predict_1x2(cal_fit, self.state.T_score, family=self.copula_family,
                                           theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                           joint=self.joint_type, use_dc=self.use_dc, rho=self.state.rho)
        p_struct_eval = model.predict_1x2(cal_eval, self.state.T_score, family=self.copula_family,
                                            theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                            joint=self.joint_type, use_dc=self.use_dc, rho=self.state.rho)
        base_ll = logloss_vec(p_struct_fit, y_fit)
        pc_fit = challenger.predict(cal_fit) if challenger is not None else None
        pc_eval = challenger.predict(cal_eval) if challenger is not None else None
        if pc_fit is not None:
            best = (base_ll, 0.0)
            for a in np.linspace(0.0, 0.25, 26):
                if a == 0.0: continue
                q = clip_matrix((1 - a) * p_struct_fit + a * pc_fit)
                ll = logloss_vec(q, y_fit)
                if ll < best[0]: best = (ll, float(a))
            if base_ll - best[0] >= self.improvement_threshold:
                a = best[1]
                deltas = (-np.log(np.clip(p_struct_eval[np.arange(len(y_eval)), y_eval], EPS, 1.0))
                          - (-np.log(np.clip(clip_matrix((1 - a) * p_struct_eval + a * pc_eval)[
                              np.arange(len(y_eval)), y_eval], EPS, 1.0))))
                self.state.challenger_alpha = a if self._bootstrap(deltas, cal_eval.date.values, 0.0) else 0.0
        ps_fit = stacker.predict(cal_fit) if (stacker is not None and stacker.model is not None) else None
        ps_eval = stacker.predict(cal_eval) if (stacker is not None and stacker.model is not None) else None
        if ps_fit is not None:
            p_cur_fit = p_struct_fit.copy(); p_cur_eval = p_struct_eval.copy()
            if self.state.challenger_alpha > 0 and pc_fit is not None:
                p_cur_fit = clip_matrix((1 - self.state.challenger_alpha) * p_cur_fit + self.state.challenger_alpha * pc_fit)
                p_cur_eval = clip_matrix((1 - self.state.challenger_alpha) * p_cur_eval + self.state.challenger_alpha * pc_eval)
            best = (base_ll, 0.0)
            for a in np.linspace(0.0, 0.35, 36):
                if a == 0.0: continue
                q = clip_matrix((1 - a) * p_cur_fit + a * ps_fit); ll = logloss_vec(q, y_fit)
                if ll < best[0]: best = (ll, float(a))
            if base_ll - best[0] >= self.improvement_threshold:
                a = best[1]
                deltas = (-np.log(np.clip(p_cur_eval[np.arange(len(y_eval)), y_eval], EPS, 1.0))
                          - (-np.log(np.clip(clip_matrix((1 - a) * p_cur_eval + a * ps_eval)[
                              np.arange(len(y_eval)), y_eval], EPS, 1.0))))
                self.state.stacker_alpha = a if self._bootstrap(deltas, cal_eval.date.values, 0.0) else 0.0
        if pc_fit is not None and ps_fit is not None:
            try:
                X_meta_fit = np.hstack([p_struct_fit, pc_fit, ps_fit])
                meta = StackingMetaLearner(C=1.0).fit(X_meta_fit, y_fit)
                p_meta_fit = meta.predict(X_meta_fit)
                if base_ll - logloss_vec(p_meta_fit, y_fit) >= self.improvement_threshold:
                    X_meta_eval = np.hstack([p_struct_eval, pc_eval, ps_eval])
                    p_meta_eval = meta.predict(X_meta_eval)
                    deltas = (-np.log(np.clip(p_struct_eval[np.arange(len(y_eval)), y_eval], EPS, 1.0))
                              - (-np.log(np.clip(p_meta_eval[np.arange(len(y_eval)), y_eval], EPS, 1.0))))
                    self.state.meta_alpha = 1.0 if self._bootstrap(deltas, cal_eval.date.values, self.strict_threshold) else 0.0
            except Exception: self.state.meta_alpha = 0.0
        self.state.meta_enabled = self.state.meta_alpha > 0
        p_final_eval = p_struct_eval.copy()
        if self.state.challenger_alpha > 0 and pc_eval is not None:
            p_final_eval = clip_matrix((1 - self.state.challenger_alpha) * p_final_eval + self.state.challenger_alpha * pc_eval)
        if self.state.stacker_alpha > 0 and ps_eval is not None:
            p_final_eval = clip_matrix((1 - self.state.stacker_alpha) * p_final_eval + self.state.stacker_alpha * ps_eval)
        self.state.cal_base_ll = logloss_vec(model.predict_1x2(cal_eval, 1.0, theta_c=0.0), y_eval)
        self.state.cal_eval_ll = logloss_vec(p_final_eval, y_eval)
        return asdict(self.state)

# ============================================================================
# 14. Market trust / conformal / governance / staking
# ============================================================================

class MarketTrustController:
    def __init__(self, alpha_max=0.25, window=150, ridge=1.0, min_n=75, improvement=0.0005,
                 bootstrap_block_days=14, bootstrap_n=800):
        self.alpha_max = float(np.clip(alpha_max, 0.0, 0.25))
        self.window = int(max(window, 25)); self.ridge = float(max(ridge, 0.0))
        self.min_n = int(max(min_n, 20)); self.improvement = float(improvement)
        self.bootstrap_block_days = int(bootstrap_block_days); self.bootstrap_n = int(bootstrap_n)
        self.alpha = 0.0; self.status = "market_anchor"; self.n = 0
        self.records = []; self.seen = set()
        self.last_bootstrap = None
    @staticmethod
    def pool(pm, ps, a):
        pm = clip_matrix(np.asarray(pm, float)); ps = clip_matrix(np.asarray(ps, float))
        a = float(np.clip(a, 0.0, 0.25))
        return clip_matrix(np.exp((1 - a) * np.log(np.clip(pm, EPS, 1.0)) + a * np.log(np.clip(ps, EPS, 1.0))))
    def _refit(self):
        if len(self.records) < self.min_n:
            self.alpha = 0.0; self.status = "insufficient_prequential_history"; return
        rec = self.records[-self.window:]
        pm = np.vstack([r[1] for r in rec]); ps = np.vstack([r[2] for r in rec])
        y = np.array([r[3] for r in rec], int); n = len(rec)
        dates = [r[4] for r in rec]
        base_losses = -np.log(np.clip(pm[np.arange(n), y], EPS, 1.0))
        best_alpha = 0.0; best_ll = float(np.mean(base_losses))
        for a in np.linspace(0.0, self.alpha_max, 51):
            if a == 0.0: continue
            q = self.pool(pm, ps, float(a))
            ll = logloss_vec(q, y) + self.ridge * float(a * a) / max(n, 1)
            if ll < best_ll: best_ll = ll; best_alpha = float(a)
        if best_alpha <= 0.0 or (float(np.mean(base_losses)) - best_ll) < self.improvement:
            self.alpha = 0.0; self.status = "market_anchor_no_validated_structural_edge"
            self.last_bootstrap = None; return
        pooled = self.pool(pm, ps, best_alpha)
        pooled_losses = -np.log(np.clip(pooled[np.arange(n), y], EPS, 1.0))
        delta = base_losses - pooled_losses
        try:
            ci = chronological_block_bootstrap(delta, dates, block_days=self.bootstrap_block_days,
                                                n_boot=self.bootstrap_n)
        except Exception:
            ci = {"mean": float(np.mean(delta)), "lo": float(np.mean(delta)), "hi": float(np.mean(delta))}
        self.last_bootstrap = ci
        if ci["lo"] > 0.0:
            self.alpha = best_alpha; self.status = "structural_trust_active"
        else:
            self.alpha = 0.0; self.status = "market_anchor_bootstrap_reject"
    def forecast(self, pm, ps):
        if pm is None: return clip_probs(ps), "structural_only"
        if not np.isfinite(pm).all(): return clip_probs(ps), "structural_only_nan_market"
        if not (0.0 <= self.alpha <= self.alpha_max):
            log.warning(f"[market] alpha {self.alpha} out of [0,{self.alpha_max}]; forcing market anchor")
            return clip_probs(pm), "market_anchor_alpha_range"
        if self.alpha <= 0: return clip_probs(pm), "market_anchor"
        return self.pool(pm, ps, self.alpha), "market_structural_log_pool"
    def update(self, key, pm, ps, y, date=None):
        if pm is None or key in self.seen: return
        if not np.isfinite(pm).all(): return
        d = pd.Timestamp(date) if date is not None else pd.Timestamp.utcnow().tz_localize(None)
        self.records.append((str(key), clip_probs(pm), clip_probs(ps), int(y), d))
        if len(self.records) > 4 * self.window: self.records = self.records[-2 * self.window:]
        self.seen.add(str(key)); self.n += 1; self._refit()
    def state(self):
        return {"alpha_max": self.alpha_max, "window": self.window, "ridge": self.ridge,
                "min_n": self.min_n, "improvement": self.improvement,
                "bootstrap_block_days": self.bootstrap_block_days, "bootstrap_n": self.bootstrap_n,
                "alpha": self.alpha, "status": self.status, "n": self.n,
                "last_bootstrap": self.last_bootstrap,
                "records": [(r[0], r[1].tolist(), r[2].tolist(), int(r[3]), r[4].isoformat())
                            for r in self.records[-self.window:]],
                "seen": sorted(self.seen)}
    @classmethod
    def restore(cls, s):
        obj = cls(float(s.get("alpha_max", 0.25)), int(s.get("window", 150)),
                  float(s.get("ridge", 1.0)), int(s.get("min_n", 75)),
                  improvement=float(s.get("improvement", 0.0005)),
                  bootstrap_block_days=int(s.get("bootstrap_block_days", 14)),
                  bootstrap_n=int(s.get("bootstrap_n", 800)))
        obj.alpha = float(s.get("alpha", 0.0)); obj.status = str(s.get("status", "market_anchor"))
        obj.n = int(s.get("n", 0)); obj.last_bootstrap = s.get("last_bootstrap")
        obj.records = []
        for r in s.get("records", []):
            d = pd.Timestamp(r[4]) if len(r) > 4 else pd.Timestamp.utcnow().tz_localize(None)
            obj.records.append((r[0], np.asarray(r[1], float), np.asarray(r[2], float), int(r[3]), d))
        obj.records = obj.records[-obj.window:]
        obj.seen = set(s.get("seen", [])); return obj

def _weighted_quantile(x, w, q, side="right"):
    order = np.argsort(x); cw = np.cumsum(w[order]); target = q * cw[-1]
    idx = int(np.searchsorted(cw, target, side=side))
    return float(x[order][min(max(idx, 0), len(x) - 1)])

class PerLeagueConformalSets:
    def __init__(self, alpha=0.10, kappa=100.0, min_n=30, half_life_days=60.0, mondrian=True):
        self.alpha = alpha; self.kappa = kappa; self.min_n = min_n
        self.half_life_days = half_life_days; self.mondrian = bool(mondrian)
        self.q_global = None; self.q_per_league = {}
        self.q_per_class = {}; self.q_per_league_class = {}
    def _weights(self, n, dates):
        if dates is None or self.half_life_days is None: return np.ones(n)
        dt = pd.to_datetime(dates); ages = (dt.max() - dt).dt.total_seconds().to_numpy() / 86400.0
        return np.exp(-np.log(2.0) * ages / max(self.half_life_days, 1.0))
    def fit(self, p_probs, y_true, leagues, dates=None):
        n = len(y_true)
        if n == 0: self.q_global = 1.0; return self
        scores = 1.0 - p_probs[np.arange(n), y_true]; w = self._weights(n, dates)
        self.q_global = _weighted_quantile(scores, w, 1 - self.alpha, side="right")
        la = np.asarray(leagues)
        for lg in np.unique(la):
            m = la == lg
            if m.sum() < self.min_n: continue
            q_l = _weighted_quantile(scores[m], w[m], 1 - self.alpha, side="right")
            n_l = int(m.sum()); wk = n_l / (n_l + self.kappa); self.q_per_league[lg] = wk * q_l + (1 - wk) * self.q_global
        if self.mondrian:
            pred = np.asarray(p_probs).argmax(axis=1)
            for c in (0, 1, 2):
                m = pred == c
                if m.sum() < self.min_n: continue
                q_c = _weighted_quantile(scores[m], w[m], 1 - self.alpha, side="right")
                n_c = int(m.sum()); wk = n_c / (n_c + self.kappa); self.q_per_class[c] = wk * q_c + (1 - wk) * self.q_global
            for lg in np.unique(la):
                for c in (0, 1, 2):
                    m = (la == lg) & (pred == c)
                    if m.sum() < max(20, self.min_n // 2): continue
                    q_lc = _weighted_quantile(scores[m], w[m], 1 - self.alpha, side="right")
                    n_lc = int(m.sum()); wk = n_lc / (n_lc + self.kappa)
                    base = self.q_per_class.get(c, self.q_per_league.get(lg, self.q_global))
                    self.q_per_league_class[(lg, c)] = wk * q_lc + (1 - wk) * base
        return self
    def q_for(self, leagues, classes=None):
        la = list(leagues)
        if not self.mondrian or classes is None:
            return np.array([self.q_per_league.get(lg, self.q_global) for lg in la])
        out = []
        for lg, c in zip(la, classes):
            key = (lg, int(c))
            if key in self.q_per_league_class: out.append(self.q_per_league_class[key])
            elif int(c) in self.q_per_class: out.append(self.q_per_class[int(c)])
            else: out.append(self.q_per_league.get(lg, self.q_global))
        return np.array(out)
    def mask(self, p, leagues):
        if self.q_global is None: raise RuntimeError("conformal set not fitted")
        p = clip_matrix(p); classes = p.argmax(axis=1) if self.mondrian else None
        q = self.q_for(np.asarray(leagues), classes)
        return (1.0 - p) <= (q[:, None] + 1e-12)
    def coverage(self, p, y, leagues):
        m = self.mask(p, leagues); return float(np.mean(m[np.arange(len(y)), np.asarray(y, int)]))
    def state(self):
        return {"alpha": self.alpha, "kappa": self.kappa, "min_n": self.min_n,
                "half_life_days": self.half_life_days, "mondrian": self.mondrian, "q_global": self.q_global,
                "q_per_league": {str(k): float(v) for k, v in self.q_per_league.items()},
                "q_per_class": {int(k): float(v) for k, v in self.q_per_class.items()},
                "q_per_league_class": {f"{k[0]}|{k[1]}": float(v) for k, v in self.q_per_league_class.items()}}
    @classmethod
    def restore(cls, d):
        obj = cls(alpha=d.get("alpha", 0.10), kappa=d.get("kappa", 100.0),
                  min_n=d.get("min_n", 30), half_life_days=d.get("half_life_days", 60.0),
                  mondrian=d.get("mondrian", True))
        obj.q_global = d.get("q_global"); obj.q_per_league = dict(d.get("q_per_league", {}))
        obj.q_per_class = {int(k): float(v) for k, v in d.get("q_per_class", {}).items()}
        pl = {}
        for k, v in d.get("q_per_league_class", {}).items():
            lg, c = k.split("|"); pl[(lg, int(c))] = float(v)
        obj.q_per_league_class = pl; return obj

class PerLeagueBinaryConformalSets(PerLeagueConformalSets):
    def fit(self, p_yes, y_yes, leagues, dates=None):
        n = len(y_yes)
        if n == 0: self.q_global = 1.0; return self
        scores = np.where(y_yes > 0.5, 1.0 - p_yes, p_yes); w = self._weights(n, dates)
        self.q_global = _weighted_quantile(scores, w, 1 - self.alpha, side="right")
        la = np.asarray(leagues)
        for lg in np.unique(la):
            m = la == lg
            if m.sum() < self.min_n: continue
            q_l = _weighted_quantile(scores[m], w[m], 1 - self.alpha, side="right")
            n_l = int(m.sum()); wk = n_l / (n_l + self.kappa); self.q_per_league[lg] = wk * q_l + (1 - wk) * self.q_global
        return self
    def mask(self, p_yes, leagues):
        if self.q_global is None: raise RuntimeError("conformal set not fitted")
        q = self.q_for(np.asarray(leagues))
        return np.column_stack([1.0 - p_yes <= q + 1e-12, p_yes <= q + 1e-12])

class TrueCumulativeConformal:
    def __init__(self, alpha=0.10, kappa=100.0, min_n=30, window=4, half_life_days=60.0):
        self.alpha = alpha; self.kappa = kappa; self.min_n = min_n; self.window = int(window)
        self.half_life_days = float(half_life_days)
        self.q_lo_global = None; self.q_hi_global = None
        self.q_lo_per_league = {}; self.q_hi_per_league = {}
    def _weights(self, n, dates):
        if dates is None or self.half_life_days is None: return np.ones(n)
        dt = pd.to_datetime(dates); ages = (dt.max() - dt).dt.total_seconds().to_numpy() / 86400.0
        return np.exp(-np.log(2.0) * ages / max(self.half_life_days, 1.0))
    def fit(self, realized_cum, pred_mean, pred_draws, leagues, dates=None):
        r = np.asarray(realized_cum, float) - np.asarray(pred_mean, float)
        pd_ = np.asarray(pred_draws, float)
        if pd_.shape[0] != len(r):
            if pd_.shape[1] == len(r): pd_ = pd_.T
            else: raise ValueError(f"pred_draws shape {pd_.shape} incompatible with n={len(r)}")
        lo_d = np.quantile(pd_, self.alpha / 2, axis=1) - np.asarray(pred_mean)
        hi_d = np.quantile(pd_, 1 - self.alpha / 2, axis=1) - np.asarray(pred_mean)
        n = len(r)
        if n == 0: self.q_lo_global = self.q_hi_global = 0.0; return self
        w = self._weights(n, dates)
        self.q_lo_global = _weighted_quantile(r - lo_d, w, self.alpha / 2)
        self.q_hi_global = _weighted_quantile(r - hi_d, w, 1 - self.alpha / 2)
        la = np.asarray(leagues)
        for lg in np.unique(la):
            m = la == lg
            if m.sum() < self.min_n: continue
            ql = _weighted_quantile((r - lo_d)[m], w[m], self.alpha / 2)
            qh = _weighted_quantile((r - hi_d)[m], w[m], 1 - self.alpha / 2)
            n_l = int(m.sum()); wk = n_l / (n_l + self.kappa)
            self.q_lo_per_league[lg] = wk * ql + (1 - wk) * self.q_lo_global
            self.q_hi_per_league[lg] = wk * qh + (1 - wk) * self.q_hi_global
        return self
    def interval(self, pred_mean, pred_draws, leagues):
        if self.q_lo_global is None: raise RuntimeError("cumulative conformal not fitted")
        pd_ = np.asarray(pred_draws, float)
        if pd_.shape[0] != len(np.asarray(pred_mean)): pd_ = pd_.T
        lo_q = np.quantile(pd_, self.alpha / 2, axis=1); hi_q = np.quantile(pd_, 1 - self.alpha / 2, axis=1)
        la = np.asarray(leagues)
        lo_a = np.array([self.q_lo_per_league.get(lg, self.q_lo_global) for lg in la])
        hi_a = np.array([self.q_hi_per_league.get(lg, self.q_hi_global) for lg in la])
        return lo_q + lo_a, hi_q + hi_a
    def state(self):
        return {"alpha": self.alpha, "kappa": self.kappa, "min_n": self.min_n, "window": self.window,
                "half_life_days": self.half_life_days,
                "q_lo_global": self.q_lo_global, "q_hi_global": self.q_hi_global,
                "q_lo_per_league": {str(k): float(v) for k, v in self.q_lo_per_league.items()},
                "q_hi_per_league": {str(k): float(v) for k, v in self.q_hi_per_league.items()}}
    @classmethod
    def restore(cls, d):
        obj = cls(alpha=d.get("alpha", 0.10), kappa=d.get("kappa", 100.0),
                  min_n=d.get("min_n", 30), window=d.get("window", 4),
                  half_life_days=d.get("half_life_days", 60.0))
        obj.q_lo_global = d.get("q_lo_global"); obj.q_hi_global = d.get("q_hi_global")
        obj.q_lo_per_league = dict(d.get("q_lo_per_league", {}))
        obj.q_hi_per_league = dict(d.get("q_hi_per_league", {})); return obj

class SelectiveGovernor:
    def __init__(self, window=600, min_n=120, target_risk=0.45, defer_refit=False):
        self.window = window; self.min_n = min_n; self.target_risk = target_risk
        self.records = []; self.threshold = 1 / 3; self.active = False
        self.defer_refit = bool(defer_refit)
    def score(self, p, disagreement=0.0):
        q = clip_probs(p); conf = float(q.max())
        ent = float(-np.sum(q * np.log(np.clip(q, EPS, 1))) / math.log(3))
        return conf - 0.25 * max(disagreement, 0.0) - 0.08 * ent, conf, ent
    def observe(self, p, y, dis=0.0, ts=None):
        s, _, _ = self.score(p, dis)
        self.records.append((s, int(np.argmax(p) == int(y)), pd.Timestamp(ts) if ts is not None else None))
        self.records = self.records[-self.window:]
        if not self.defer_refit: self._refit()
    def observe_batch(self, rows):
        """rows: list of (p, y, dis, ts). Appends all then refits once."""
        for p, y, dis, ts in rows:
            s, _, _ = self.score(p, dis)
            self.records.append((s, int(np.argmax(p) == int(y)),
                                  pd.Timestamp(ts) if ts is not None else None))
        self.records = self.records[-self.window:]
        self._refit()
    def _refit(self):
        if len(self.records) < self.min_n: self.active = False; return
        s = np.array([r[0] for r in self.records]); h = np.array([r[1] for r in self.records]); best = None
        for t in np.linspace(np.quantile(s, 0.1), np.quantile(s, 0.9), 41):
            m = s >= t; cov = m.mean()
            if cov < 0.35 or not m.any(): continue
            risk = 1 - h[m].mean()
            if risk <= self.target_risk and (best is None or cov > best[0]): best = (cov, float(t), risk)
        if best: self.active = True; self.threshold = best[1]
    def evaluate(self, p, dis=0.0):
        s, c, e = self.score(p, dis)
        return {"publish": bool((not self.active) or (s >= self.threshold)), "score": float(s),
                "confidence": float(c), "entropy": float(e), "threshold": float(self.threshold), "active": bool(self.active)}
    def state(self):
        return {"window": self.window, "min_n": self.min_n, "target_risk": self.target_risk,
                "defer_refit": self.defer_refit,
                "records": [(a, b, str(c) if c is not None else None) for a, b, c in self.records],
                "threshold": self.threshold, "active": self.active}
    @classmethod
    def restore(cls, s):
        obj = cls(int(s.get("window", 600)), int(s.get("min_n", 120)), float(s.get("target_risk", 0.45)),
                  bool(s.get("defer_refit", False)))
        obj.records = [(float(a), int(b), pd.Timestamp(c) if c else None) for a, b, c in s.get("records", [])]
        obj.threshold = float(s.get("threshold", 1 / 3)); obj.active = bool(s.get("active", False)); return obj

class PageHinkley:
    def __init__(self, baseline=1.0, delta=0.01, threshold=5.0, warmup=50, baseline_half_life=500.0):
        self.baseline = baseline; self.delta = delta; self.threshold = threshold; self.warmup = warmup
        self.n = 0; self.cum = 0.0; self.min_cum = 0.0; self.alerts = 0
        self.baseline_half_life = float(baseline_half_life)
        self._decay = math.exp(-math.log(2.0) / max(self.baseline_half_life, 1.0))
    @classmethod
    def warm_started(cls, losses, delta=0.01, threshold=5.0, warmup=50, baseline_half_life=500.0):
        obj = cls(baseline=None, delta=delta, threshold=threshold, warmup=warmup, baseline_half_life=baseline_half_life)
        if losses is None or len(losses) == 0: return obj
        arr = np.asarray(losses, float); obj.baseline = float(arr.mean()); obj.n = len(arr); return obj
    def update(self, loss):
        self.n += 1
        if self.baseline is None: self.baseline = float(loss); return False
        if self.n <= self.warmup:
            self.baseline = self._decay * self.baseline + (1 - self._decay) * float(loss); return False
        self.cum += float(loss) - self.baseline - self.delta; self.min_cum = min(self.min_cum, self.cum)
        alert = False
        if self.cum - self.min_cum > self.threshold:
            self.alerts += 1; self.cum = 0.0; self.min_cum = 0.0; alert = True
        self.baseline = self._decay * self.baseline + (1 - self._decay) * float(loss)
        return alert
    def state(self):
        return {"baseline": self.baseline, "delta": self.delta, "threshold": self.threshold,
                "warmup": self.warmup, "n": self.n, "cum": self.cum, "min_cum": self.min_cum,
                "alerts": self.alerts, "baseline_half_life": self.baseline_half_life}
    @classmethod
    def restore(cls, s):
        obj = cls(float(s.get("baseline", 1.0)), float(s.get("delta", 0.01)),
                  float(s.get("threshold", 5.0)), int(s.get("warmup", 50)),
                  float(s.get("baseline_half_life", 500.0)))
        obj.n = int(s.get("n", 0)); obj.cum = float(s.get("cum", 0.0))
        obj.min_cum = float(s.get("min_cum", 0.0)); obj.alerts = int(s.get("alerts", 0)); return obj

@dataclass
class GateConfig:
    min_matches: int = 200; min_folds: int = 5; min_mean_ll_gain: float = 0.002
    min_ci_lower: float = 0.0; max_abs_ece_increase: float = 0.015; max_fold_degradation: float = 0.03

class ModelRegistry:
    def __init__(self, min_improvement=0.005, min_obs=200, block=200, gate_cfg=None):
        self.champion = None; self.challengers = {}
        self.min_improvement = float(min_improvement); self.min_obs = int(min_obs); self.block = int(block)
        self.gate_cfg = gate_cfg or GateConfig()
        self.history = []; self._losses = {}; self._dates = {}
    def _ensure(self, name):
        if name not in self._losses:
            self._losses[name] = deque(maxlen=5000); self._dates[name] = deque(maxlen=5000)
    def register_champion(self, name, metric, n_obs):
        ts = datetime.now(timezone.utc).isoformat()
        self.champion = {"name": str(name), "metric": float(metric), "n_obs": int(n_obs), "registered_at": ts}
        self.history.append((ts, "INIT", str(name), float(metric), float(metric))); self._ensure(str(name))
    def record(self, name, loss, date=None):
        self._ensure(str(name)); self._losses[str(name)].append(float(loss))
        self._dates[str(name)].append(pd.Timestamp(date) if date is not None else pd.Timestamp.utcnow().tz_localize(None))
    def rolling_metric(self, name):
        buf = self._losses.get(str(name))
        if buf is None or len(buf) < self.block: return None
        return float(np.mean(list(buf)[-self.block:]))
    def try_promote(self, name):
        if self.champion is None or str(name) not in self.challengers: return False, "unknown"
        c = self.challengers[str(name)]
        if c["n_obs"] < self.min_obs: return False, "insufficient_obs"
        improvement = self.champion["metric"] - c["metric"]
        if improvement < self.min_improvement: return False, "no_improvement"
        buf = self._losses.get(str(name))
        if buf is not None and len(buf) >= self.block:
            recent = np.asarray(list(buf)[-self.block:], float)
            arr = self.champion["metric"] - recent
            try: ci = chronological_block_bootstrap(arr, list(self._dates[str(name)])[-self.block:])
            except Exception: ci = {"lo": improvement, "hi": improvement}
            if ci["lo"] <= self.gate_cfg.min_ci_lower: return False, "not_significant"
        ts = datetime.now(timezone.utc).isoformat(); prev = self.champion["name"]
        self.history.append((ts, "PROMOTE", str(name), float(c["metric"]), float(self.champion["metric"])))
        self.champion = c; del self.challengers[str(name)]
        return True, {"promoted_from": prev, "to": str(name)}
    def register_challenger(self, name, metric, n_obs):
        ts = datetime.now(timezone.utc).isoformat()
        self.challengers[str(name)] = {"name": str(name), "metric": float(metric), "n_obs": int(n_obs), "registered_at": ts}
        self._ensure(str(name))
    def state(self):
        return {"champion": self.champion, "challengers": self.challengers,
                "history": self.history, "min_improvement": self.min_improvement,
                "min_obs": self.min_obs, "block": self.block,
                "rolling_metrics": {k: self.rolling_metric(k) for k in self._losses}}
    @classmethod
    def restore(cls, d):
        obj = cls(d.get("min_improvement", 0.005), d.get("min_obs", 200), d.get("block", 200))
        obj.champion = d.get("champion"); obj.challengers = d.get("challengers", {})
        obj.history = [tuple(h) for h in d.get("history", [])]
        if obj.champion: obj._ensure(obj.champion["name"])
        return obj

class AutoRefitScheduler:
    def __init__(self, max_recent_alerts=2):
        self.max_recent = max_recent_alerts; self.requests = []; self._last = 0
    def should_refit(self, ph):
        if ph is None: return False
        if ph.alerts > self._last: self._last = ph.alerts; return True
        return False
    def request(self, reason):
        self.requests.append({"ts": datetime.now(timezone.utc).isoformat(), "reason": reason})
        if len(self.requests) > 200: self.requests = self.requests[-100:]
    def clear(self): self.requests = []; self._last = 0
    def state(self): return {"max_recent": self.max_recent, "requests": self.requests, "last_alert_count": self._last}
    @classmethod
    def restore(cls, s):
        obj = cls(int(s.get("max_recent", 2))); obj.requests = list(s.get("requests", []))
        obj._last = int(s.get("last_alert_count", 0)); return obj

class CLVTracker:
    def __init__(self, min_n=50): self.records = []; self.min_n = int(min_n)
    def record(self, key, po, pc, market):
        if po is None or pc is None: return
        self.records.append((str(key), np.asarray(po, float), np.asarray(pc, float), str(market)))
        if len(self.records) > 20000: self.records = self.records[-10000:]
    def summary(self):
        if not self.records: return {"n": 0, "clv": 0.0, "beat_close_pct": 0.0}
        clvs = np.asarray([float(pc.mean() - po.mean()) for _, po, pc, _ in self.records], float)
        return {"n": len(clvs), "clv": float(clvs.mean()), "beat_close_pct": float(np.mean(clvs > 0))}
    def state(self):
        return {"records": [(r[0], r[1].tolist(), r[2].tolist(), r[3]) for r in self.records[-500:]], "min_n": self.min_n}
    @classmethod
    def restore(cls, s):
        obj = cls(int(s.get("min_n", 50)))
        obj.records = [(r[0], np.asarray(r[1], float), np.asarray(r[2], float), r[3]) for r in s.get("records", [])]
        return obj

class PortfolioStaking:
    def __init__(self, bankroll=1.0, risk_aversion=1.0, max_leverage=2.0, min_edge=0.02,
                 min_stake=0.001, edge_quantile=0.25, max_per_match_pct=0.05,
                 min_effective_samples=30):
        self.bankroll = bankroll; self.lam = risk_aversion; self.leverage = max_leverage
        self.min_edge = min_edge; self.min_stake = min_stake; self.q = edge_quantile
        self.max_per_match_pct = float(max_per_match_pct)
        self.min_effective_samples = int(min_effective_samples)
    def optimize(self, ids, p_mean, p_draws, odds, conformal_mask=None):
        edges = p_draws * odds[None, :, :] - 1.0
        edge_q = np.quantile(edges, self.q, axis=0); cands = []
        for i in range(len(ids)):
            for j, mk in enumerate(("H", "D", "A")):
                o = odds[i, j]
                if not np.isfinite(o) or o < 1.01: continue
                p = float(p_mean[i, j])
                if not np.isfinite(p) or p <= 0: continue
                if conformal_mask is not None and not bool(conformal_mask[i, j]): continue
                e = float(edge_q[i, j])
                if e < self.min_edge: continue
                cands.append({"i": i, "j": j, "id": ids[i], "mkt": mk, "p": p, "o": float(o), "e": e})
        n = len(cands)
        if n == 0: return []
        S = p_draws.shape[0]
        ex = np.zeros((S, n))
        for k, c in enumerate(cands): ex[:, k] = p_draws[:, c["i"], c["j"]] * c["o"] - 1.0
        # Degenerate covariance guard
        if n == 1 or S < self.min_effective_samples:
            Sigma = np.diag(np.maximum(ex.var(axis=0), 1e-6))
        else:
            try: Sigma = LedoitWolf().fit(ex).covariance_
            except Exception: Sigma = np.diag(np.maximum(ex.var(axis=0), 1e-6))
        Sigma = 0.5 * (Sigma + Sigma.T) + np.eye(n) * 1e-6
        # Ensure positive definiteness
        eigs = np.linalg.eigvalsh(Sigma)
        if eigs.min() <= 0: Sigma += (abs(eigs.min()) + 1e-8) * np.eye(n)
        e_v = np.array([c["e"] for c in cands]); o_v = np.array([c["o"] for c in cands])
        per_match = self.max_per_match_pct * self.bankroll; match_ids = np.array([c["i"] for c in cands])
        def nobj(s): return -(s @ e_v - 0.5 * self.lam * s @ Sigma @ s)
        def nobj_g(s): return -(e_v - self.lam * (Sigma @ s))
        def lev(s): return self.leverage * self.bankroll - np.sum(np.abs(s) * o_v)
        def lev_g(s): return -np.sign(s) * o_v
        cons = [{"type": "ineq", "fun": lev, "jac": lev_g}]
        for i in np.unique(match_ids):
            mask = match_ids == i
            def mcon(s, mask=mask): return per_match - float(np.sum(s[mask] * o_v[mask]))
            cons.append({"type": "ineq", "fun": mcon})
        r = minimize(nobj, np.full(n, self.bankroll * 0.005), jac=nobj_g, method="SLSQP",
                     bounds=[(0, self.bankroll * 0.05)] * n, constraints=cons,
                     options={"maxiter": 400, "ftol": 1e-9})
        if not r.success: return []
        out = []
        for k, c in enumerate(cands):
            if r.x[k] < self.min_stake: continue
            pl = float(np.quantile(p_draws[:, c["i"], c["j"]], 0.10))
            w = (c["p"] - pl) / max(c["p"], 1e-6)
            conf = "high" if w < 0.30 else ("medium" if w < 0.55 else "low")
            out.append({"fixture_id": c["id"], "market": c["mkt"], "p_mean": c["p"],
                        "p_lower": pl, "odds": c["o"], "edge": c["e"],
                        "stake_pct": float(r.x[k]) / self.bankroll, "confidence": conf})
        return out

# ============================================================================
# 15. Engine
# ============================================================================

@dataclass
class EngineConfig:
    maher: MaherConfig = field(default_factory=MaherConfig)
    hier: HierStateConfig = field(default_factory=HierStateConfig)
    evidence_cfg: EvidenceCoverageConfig = field(default_factory=EvidenceCoverageConfig)
    enable_challenger: bool = True
    challenger_C: float = 0.5
    enable_stacker: bool = True
    stacker_n_estimators: int = 150
    stacker_max_depth: int = 3
    stacker_learning_rate: float = 0.05
    market_alpha_max: float = 0.25
    market_window: int = 150
    market_ridge: float = 1.0
    market_min_n: int = 75
    conformal_alpha: float = 0.10
    cumulative_windows: Tuple[int, ...] = (2, 3, 4, 5, 6)
    n_posterior_samples: int = 300
    top_k_cs: int = 8
    registry_min_improvement: float = 0.005
    registry_min_obs: int = 200
    registry_block: int = 200
    calibration_split_frac: float = 0.5
    calibration_improvement: float = 0.0015
    calibration_strict_improvement: float = 0.004
    compress_save: bool = True
    use_joint_coherence: bool = True
    use_dc: bool = False
    copula_families: Tuple[str, ...] = ("frank", "clayton", "gumbel")
    use_mondrian_conformal: bool = True
    market_half_life_days: float = 180.0
    market_refit_every: int = 200
    enable_evidence_features: bool = True
    enable_evidence_residual: bool = True
    evidence_min_rows: int = 200
    evidence_folds: int = 5
    evidence_purge_days: int = 1
    evidence_max_logit_cap: float = 0.15
    evidence_min_out_of_sample_gain: float = 0.0005
    forecast_cutoff_minutes_before_kickoff: float = 5.0
    use_dirichlet: bool = True
    strict_team_boundary: bool = True
    allow_team_expansion: bool = False
    market_bootstrap_block_days: int = 14
    market_bootstrap_n: int = 800
    per_league_drift: bool = True
    joint_batch_max_elems: int = BATCH_JOINT_MAX_ELEMS
    dirichlet_max_elems: int = DIRICHLET_MAX_ELEMS
    selective_defer_refit: bool = True
    ensemble_evidence_buckets: int = 2

class FAPMEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg or EngineConfig()
        self.feature_builder = CausalFeatureBuilder(market_half_life_days=self.cfg.market_half_life_days,
                                                     market_refit_every=self.cfg.market_refit_every)
        self.model = MaherPoissonModel(self.cfg.maher)
        self.hier = HierarchicalDynamicGoalState(self.cfg.hier, self.cfg.maher)
        self.challenger = OutcomeChallenger(self.cfg.challenger_C) if self.cfg.enable_challenger else None
        self.stacker = ResidualStacker(n_estimators=self.cfg.stacker_n_estimators,
                                         max_depth=self.cfg.stacker_max_depth,
                                         learning_rate=self.cfg.stacker_learning_rate) \
            if self.cfg.enable_stacker else None
        self.meta = StackingMetaLearner(C=1.0)
        self.ensemble = RegimeConditionalEnsemble(evidence_buckets=self.cfg.ensemble_evidence_buckets)
        self.dirichlet = DirichletCalibrator(max_elems=self.cfg.dirichlet_max_elems) if self.cfg.use_dirichlet else None
        self.gate = EvidenceGate(self.cfg.calibration_split_frac, self.cfg.calibration_improvement,
                                  use_dc=self.cfg.use_dc, copula_families=self.cfg.copula_families,
                                  strict_threshold=self.cfg.calibration_strict_improvement)
        self.market = MarketTrustController(
            self.cfg.market_alpha_max, self.cfg.market_window, self.cfg.market_ridge, self.cfg.market_min_n,
            bootstrap_block_days=self.cfg.market_bootstrap_block_days,
            bootstrap_n=self.cfg.market_bootstrap_n)
        self.selective = SelectiveGovernor(defer_refit=self.cfg.selective_defer_refit)
        self.drift = PageHinkley()
        self.drift_per_league: Dict[str, PageHinkley] = {}
        self.registry = ModelRegistry(self.cfg.registry_min_improvement, self.cfg.registry_min_obs,
                                       self.cfg.registry_block)
        self.refit_scheduler = AutoRefitScheduler()
        self.clv = CLVTracker()
        self.conformal = None; self.conformal_ou = None; self.conformal_btts = None
        self.cumulative_conformal = {}
        self.evidence_ledger = None; self.evidence_coverage = None; self.evidence_residual = None
        self.fitted = False
        self.journal = []; self.assimilated = set()
        self.copula_theta = 0.0; self.copula_family = "frank"; self.bp_lam3 = 0.0; self.joint_type = "copula"
        self._frozen_market_alpha = 0.0
        self._conformal_dirty = False
        self._assimilation_count = 0
        self._evidence_gain = None

    def _warmup_market_controller(self, warmup_frame):
        if warmup_frame is None or len(warmup_frame) == 0: return
        frame = warmup_frame.sort_values("date").reset_index(drop=True)
        try:
            p_struct = self.model.predict_1x2(
                frame, self.gate.state.T_score, family=self.copula_family,
                theta_c=self.copula_theta, bp_l3=self.bp_lam3, joint=self.joint_type,
                use_dc=self.cfg.use_dc, rho=self.gate.state.rho)
        except Exception as e:
            log.warning(f"[market-warmup] structural batch failed: {e}")
            return
        for i, r in frame.iterrows():
            pm = self._market_prob_from_row(r)
            if pm is None: continue
            self.market.update(str(r.match_key), pm, p_struct[i], int(r.y), date=r.date)

    def _blend(self, p, df):
        if self.challenger is not None and self.gate.state.challenger_alpha > 0:
            try: p = clip_matrix((1 - self.gate.state.challenger_alpha) * p
                                  + self.gate.state.challenger_alpha * self.challenger.predict(df))
            except Exception: pass
        if self.stacker is not None and self.stacker.model is not None and self.gate.state.stacker_alpha > 0:
            try: p = clip_matrix((1 - self.gate.state.stacker_alpha) * p
                                  + self.gate.state.stacker_alpha * self.stacker.predict(df))
            except Exception: pass
        return p

    def _apply_meta(self, p, df):
        if self.gate.state.meta_alpha <= 0 or self.meta.model is None: return p
        if self.challenger is None or self.stacker is None or self.stacker.model is None: return p
        try:
            X = np.hstack([p, self.challenger.predict(df), self.stacker.predict(df)])
            pm = self.meta.predict(X); a = float(self.gate.state.meta_alpha)
            return clip_matrix((1 - a) * p + a * pm)
        except Exception: return p

    def _apply_dirichlet(self, p):
        if self.dirichlet is not None and self.dirichlet.fitted:
            try: return self.dirichlet.transform(p)
            except Exception: return p
        return p

    @staticmethod
    def _market_prob_from_row(r):
        vals = [r.get("market_p_home"), r.get("market_p_draw"), r.get("market_p_away")]
        if all(pd.notna(v) for v in vals):
            try: return clip_probs(vals)
            except ValueError: pass
        vals = [r.get("b365h"), r.get("b365d"), r.get("b365a")]
        if all(pd.notna(v) for v in vals):
            try:
                o = _odds_to_prob_vec(np.array([float(v) for v in vals], float))
                if np.isfinite(o).all(): return clip_probs(o)
            except (TypeError, ValueError): pass
        return None

    def _compute_posterior_draws(self, x, S=None):
        S = int(S or max(20, int(self.cfg.n_posterior_samples)))
        rng = np.random.default_rng(MASTER_SEED)
        theta_draws = self.model.posterior_draws(S, rng)
        phi_h_draws, phi_a_draws = self.model.posterior_phi_draws(S, rng)
        gh_draws, ga_draws = self.model.posterior_gamma_draws(S, rng)
        lg, hi, ai, neutral, X = self.model._indices(x)
        lam_h, lam_a, _, _ = self.model.predict_intensity_batch(lg, hi, ai, neutral, X, theta_draws)
        log_lh = np.log(np.maximum(lam_h, EPS)); log_la = np.log(np.maximum(lam_a, EPS))
        adj_h = np.where(np.abs(gh_draws)[:, None] > 1e-9,
                         np.exp(gh_draws[:, None] * (log_lh - self.model.log_lambda_total_mean)), 1.0)
        adj_a = np.where(np.abs(ga_draws)[:, None] > 1e-9,
                         np.exp(ga_draws[:, None] * (log_la - self.model.log_lambda_total_mean)), 1.0)
        phi_h_eff = np.clip(phi_h_draws[:, None] * adj_h, self.model.cfg.nb_phi_min, self.model.cfg.nb_phi_max)
        phi_a_eff = np.clip(phi_a_draws[:, None] * adj_a, self.model.cfg.nb_phi_min, self.model.cfg.nb_phi_max)
        p_draws_struct = _batch_joint_1x2(lam_h, lam_a, self.gate.state.T_score, self.copula_family,
                                            self.copula_theta, self.bp_lam3, self.joint_type,
                                            self.gate.state.rho, self.cfg.use_dc,
                                            phi_h_eff, phi_a_eff, self.model.cfg.score_cap,
                                            max_elems=self.cfg.joint_batch_max_elems)
        return p_draws_struct, lam_h, lam_a

    def _apply_dirichlet_to_draws(self, p_draws):
        """Applies the multinomial-logistic calibrator to each posterior draw, chunked."""
        if self.dirichlet is None or not self.dirichlet.fitted: return p_draws
        S, n, _ = p_draws.shape
        total = S * n
        flat = p_draws.reshape(-1, 3)
        cap = int(self.cfg.dirichlet_max_elems)
        if total <= cap:
            cal = self.dirichlet.transform(flat)
        else:
            cal = np.empty_like(flat)
            for start in range(0, total, cap):
                end = min(total, start + cap)
                cal[start:end] = self.dirichlet.transform(flat[start:end])
        return cal.reshape(S, n, 3)

    def _apply_market_pool_batch(self, p_draws, market_probs, alpha):
        """p_draws: (S, n, 3); market_probs: list of length n of (3,) or None.
        Returns pooled draws. If alpha <= 0 returns p_draws unchanged."""
        if alpha <= 0.0: return p_draws
        S, n, _ = p_draws.shape
        log_pd = np.log(np.clip(p_draws, EPS, 1.0))
        log_pm = np.zeros((n, 3))
        present = np.zeros(n, dtype=bool)
        for i, pm in enumerate(market_probs):
            if pm is not None:
                log_pm[i] = np.log(np.clip(pm, EPS, 1.0)); present[i] = True
        out = p_draws.copy()
        if not present.any(): return out
        idx = np.where(present)[0]
        lp = log_pm[idx][None, :, :]
        lq = log_pd[:, idx, :]
        mixed = np.exp((1 - alpha) * lp + alpha * lq)
        s = mixed.sum(axis=2, keepdims=True)
        mixed = mixed / np.maximum(s, EPS)
        out[:, idx, :] = mixed
        return out

    def _build_evidence_features(self, train_n, cal_n, evidence):
        if evidence is None or not self.cfg.enable_evidence_features: return None, None, None
        self.evidence_ledger = ProspectiveEvidenceLedger(strict=True)
        ingest = self.evidence_ledger.ingest(evidence)
        log.info(f"[evidence] ingest {ingest}")
        cutoff_min = float(self.cfg.forecast_cutoff_minutes_before_kickoff)
        evs = {}
        for name, fr in (("train", train_n), ("cal", cal_n)):
            fx = fr[["match_key", "date", "home", "away"]].copy()
            fx["match_id"] = fr["match_key"].astype(str)
            fx["home_team"] = fr["home"].astype(str); fx["away_team"] = fr["away"].astype(str)
            fx["kickoff_utc"] = fr["date"].astype(str)
            fx["forecast_cutoff_utc"] = (pd.to_datetime(fr["date"]) - pd.Timedelta(minutes=cutoff_min)).astype(str)
            ff = self.evidence_ledger.feature_frame(
                fx[["match_id", "home_team", "away_team", "kickoff_utc", "forecast_cutoff_utc"]])
            evs[name] = ff
        combined = pd.concat([evs["train"], evs["cal"]], ignore_index=True)
        self.evidence_coverage = EvidenceCoverageController(self.cfg.evidence_cfg)
        cov = self.evidence_coverage.assess(pd.concat([train_n, cal_n], ignore_index=True), combined)
        log.info(f"[evidence] coverage: {cov['channel_counts']}")
        return evs["train"], evs["cal"], combined

    def fit(self, train, cal, evidence=None):
        train_n = CausalContract.normalize(train, True)
        cal_n = CausalContract.normalize(cal, True)
        overlap = set(train_n.match_key) & set(cal_n.match_key)
        if overlap: cal_n = cal_n[~cal_n.match_key.isin(overlap)].reset_index(drop=True)
        ev_train, ev_cal, _ = self._build_evidence_features(train_n, cal_n, evidence)
        full = pd.concat([train_n, cal_n], ignore_index=True)
        fb = CausalFeatureBuilder(market_half_life_days=self.cfg.market_half_life_days,
                                   market_refit_every=self.cfg.market_refit_every)
        ev_full = pd.concat([ev_train, ev_cal], ignore_index=True) if ev_train is not None else None
        full_feat = fb.build(full, update_state=True, evidence_features=ev_full)
        self.feature_builder = fb
        keys_first = set(train_n.match_key); mask = full_feat.match_key.isin(keys_first).to_numpy()
        train_feat = full_feat.loc[mask].reset_index(drop=True)
        cal_all = full_feat.loc[~mask].reset_index(drop=True)
        n_cal = len(cal_all)
        if n_cal < 60:
            cal_gate = cal_all; cal_conf = cal_all
        else:
            n_gate = int(n_cal * 0.5)
            cal_gate = cal_all.iloc[:n_gate].reset_index(drop=True)
            cal_conf = cal_all.iloc[n_gate:].reset_index(drop=True)
        self.model.fit(train_feat)
        if self.cfg.hier.enabled:
            self.hier.fit(train_feat, cutoff=pd.Timestamp(train_feat["date"].max()) + pd.Timedelta(seconds=1),
                          phi_h=self.model.phi_h, phi_a=self.model.phi_a)
        if self.challenger is not None: self.challenger.fit(train_feat)
        if self.stacker is not None:
            try: self.stacker.fit(train_feat, train_feat.y.to_numpy(int))
            except Exception as e: log.warning(f"[stacker] {e}"); self.stacker = None
        cal_state = self.gate.fit(self.model, cal_gate, self.challenger, self.stacker)
        self.copula_theta = float(self.gate.copula_theta); self.copula_family = self.gate.copula_family
        self.bp_lam3 = float(self.gate.bp_lam3); self.joint_type = self.gate.joint_type
        if self.gate.state.meta_alpha > 0 and self.challenger is not None and self.stacker is not None \
                and self.stacker.model is not None:
            try:
                ps = self.model.predict_1x2(cal_gate, self.gate.state.T_score, family=self.copula_family,
                                              theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                              joint=self.joint_type, use_dc=self.cfg.use_dc, rho=self.gate.state.rho)
                pb = self._blend(ps, cal_gate)
                X = np.hstack([pb, self.challenger.predict(cal_gate), self.stacker.predict(cal_gate)])
                self.meta.fit(X, cal_gate.y.to_numpy(int))
            except Exception as e: log.warning(f"[meta] {e}"); self.gate.state.meta_alpha = 0.0
        if self.dirichlet is not None:
            try:
                p_struct = self.model.predict_1x2(cal_gate, self.gate.state.T_score, family=self.copula_family,
                                                    theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                                    joint=self.joint_type, use_dc=self.cfg.use_dc, rho=self.gate.state.rho)
                p_blend = self._apply_meta(self._blend(p_struct, cal_gate), cal_gate)
                rep = self.dirichlet.fit(p_blend, cal_gate.y.to_numpy(int))
                log.info(f"[dirichlet] {rep}")
                self.gate.state.dirichlet_fitted = bool(self.dirichlet.fitted)
            except Exception as e:
                log.warning(f"[dirichlet] {e}"); self.gate.state.dirichlet_fitted = False
        try:
            p_struct = self.model.predict_1x2(cal_gate, self.gate.state.T_score, family=self.copula_family,
                                                theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                                joint=self.joint_type, use_dc=self.cfg.use_dc, rho=self.gate.state.rho)
            p_blend = self._apply_dirichlet(self._apply_meta(self._blend(p_struct, cal_gate), cal_gate))
            experts = {"structural": p_blend}
            if self.challenger is not None: experts["challenger"] = self.challenger.predict(cal_gate)
            if self.stacker is not None and self.stacker.model is not None: experts["stacker"] = self.stacker.predict(cal_gate)
            if self.hier.fitted:
                try: experts["hierarchical"] = self.hier.probabilities(cal_gate, phi_h=self.model.phi_h,
                                                                        phi_a=self.model.phi_a, cap=self.model.cfg.score_cap)
                except Exception: pass
            p_draws_cal, lam_h_cal, lam_a_cal = self._compute_posterior_draws(cal_gate, S=40)
            cv = (lam_h_cal.std(axis=0) / np.maximum(lam_h_cal.mean(axis=0), EPS)
                  + lam_a_cal.std(axis=0) / np.maximum(lam_a_cal.mean(axis=0), EPS)) / 2.0
            ep = cal_gate.get("evidence_present", pd.Series(0.0, index=cal_gate.index)).fillna(0.0).to_numpy()
            rk = [self.ensemble._regime(e, u) for e, u in zip(ep, cv)]
            rep = self.ensemble.fit(experts, cal_gate.y.to_numpy(int), rk,
                                     cv_for_edges=cv, evidence_for_edges=ep)
            log.info(f"[ensemble] {rep.get('n', 0)} rows, {rep.get('n_regimes', 0)} regimes")
        except Exception as e:
            log.warning(f"[ensemble] {e}"); self.ensemble.fitted = False
        self._evidence_gain = None
        if self.cfg.enable_evidence_residual and ev_cal is not None and not ev_cal.empty and len(cal_conf) >= 60:
            try:
                p_blend = self._apply_dirichlet(self._apply_meta(self._blend(
                    self.model.predict_1x2(cal_gate, self.gate.state.T_score, family=self.copula_family,
                                            theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                            joint=self.joint_type, use_dc=self.cfg.use_dc, rho=self.gate.state.rho),
                    cal_gate), cal_gate))
                ids_gate = cal_gate["match_id"].astype(str).to_numpy() if "match_id" in cal_gate.columns \
                    else cal_gate["match_key"].astype(str).to_numpy()
                ev_g = ev_cal.set_index("match_id").reindex(ids_gate).reset_index(drop=True)
                self.evidence_residual = CrossFittedEvidenceResidualExpert(self.cfg.evidence_cfg)
                rep = self.evidence_residual.fit_cross_fitted(
                    ev_g, p_blend, cal_gate.y.to_numpy(int), cal_gate["date"].to_numpy())
                log.info(f"[evidence_residual] {rep}")
                if not self.evidence_residual.fitted:
                    self.evidence_residual = None
                else:
                    p_base_conf = self._apply_dirichlet(self._apply_meta(self._blend(
                        self.model.predict_1x2(cal_conf, self.gate.state.T_score, family=self.copula_family,
                                                theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                                joint=self.joint_type, use_dc=self.cfg.use_dc,
                                                rho=self.gate.state.rho), cal_conf), cal_conf))
                    ids_conf = cal_conf["match_id"].astype(str).to_numpy() if "match_id" in cal_conf.columns \
                        else cal_conf["match_key"].astype(str).to_numpy()
                    ev_c = ev_cal.set_index("match_id").reindex(ids_conf).reset_index(drop=True)
                    p_ev_conf = self.evidence_residual.predict(ev_c, p_base_conf)
                    ll_base = logloss_vec(p_base_conf, cal_conf.y.values)
                    ll_ev = logloss_vec(p_ev_conf, cal_conf.y.values)
                    gain = ll_base - ll_ev
                    self._evidence_gain = float(gain)
                    if gain < self.cfg.evidence_min_out_of_sample_gain:
                        log.info(f"[evidence_residual] disabled: oos gain {gain:.5f} < {self.cfg.evidence_min_out_of_sample_gain}")
                        self.evidence_residual = None
                    else:
                        log.info(f"[evidence_residual] kept: oos gain {gain:.5f}")
            except Exception as e:
                log.warning(f"[evidence_residual] {e}"); self.evidence_residual = None
        try:
            self._warmup_market_controller(train_feat)
        except Exception as e:
            log.warning(f"[market-warmup] {e}")
        self._frozen_market_alpha = float(self.market.alpha)
        log.info(f"[market] frozen alpha after warmup: {self._frozen_market_alpha:.4f} status={self.market.status}")
        p_cal = self._cal_pool_replay(cal_conf)
        self.conformal = PerLeagueConformalSets(alpha=self.cfg.conformal_alpha,
                                                  mondrian=self.cfg.use_mondrian_conformal).fit(
            p_cal, cal_conf.y.values, cal_conf.league.values, dates=cal_conf.date.values)
        mats, _, _ = self._score_mats(cal_conf)
        L = mats[0].shape[0]; ii = np.arange(L)[:, None]; jj = np.arange(L)[None, :]
        ou = np.array([float(J[ii + jj > 2].sum()) for J in mats])
        bt = np.array([float(J[(ii >= 1) & (jj >= 1)].sum()) for J in mats])
        if abs(self.gate.state.T_ou - 1.0) > 1e-9: ou = _temper_binary(ou, self.gate.state.T_ou)
        if abs(self.gate.state.T_btts - 1.0) > 1e-9: bt = _temper_binary(bt, self.gate.state.T_btts)
        y_ou = ((cal_conf.home_goals + cal_conf.away_goals) > 2.5).astype(float).values
        y_bt = ((cal_conf.home_goals >= 1) & (cal_conf.away_goals >= 1)).astype(float).values
        self.conformal_ou = PerLeagueBinaryConformalSets(alpha=self.cfg.conformal_alpha).fit(
            ou, y_ou, cal_conf.league.values, dates=cal_conf.date.values)
        self.conformal_btts = PerLeagueBinaryConformalSets(alpha=self.cfg.conformal_alpha).fit(
            bt, y_bt, cal_conf.league.values, dates=cal_conf.date.values)
        self._fit_cumulative_conformal(cal_conf)
        losses = -np.log(np.clip(p_cal[np.arange(len(cal_conf)), cal_conf.y.values], EPS, 1.0))
        self.drift = PageHinkley.warm_started(losses)
        self.drift_per_league = {}
        if self.cfg.per_league_drift:
            lg_arr = cal_conf.league.values
            for lg in np.unique(lg_arr):
                mask_lg = lg_arr == lg
                self.drift_per_league[str(lg)] = PageHinkley.warm_started(losses[mask_lg])
        champ = f"fapm_v67_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.registry.register_champion(champ, float(np.mean(losses)), len(cal_conf))
        for i, r in cal_conf.iterrows():
            self.registry.record(champ, float(losses[i]), date=r.get("date"))
        self._assimilation_count = 0
        self._conformal_dirty = False
        self.fitted = True
        return {"model": "fitted", "n_train": len(train_feat), "n_cal_gate": len(cal_gate),
                "n_cal_conf": len(cal_conf),
                "gate": cal_state, "phi_h": self.model.phi_h, "phi_a": self.model.phi_a,
                "copula_theta": self.copula_theta, "copula_family": self.copula_family,
                "bp_lam3": self.bp_lam3, "joint_type": self.joint_type,
                "meta_alpha": float(self.gate.state.meta_alpha),
                "ensemble_fitted": bool(self.ensemble.fitted),
                "dirichlet_fitted": bool(self.dirichlet and self.dirichlet.fitted),
                "evidence_residual_fitted": bool(self.evidence_residual and self.evidence_residual.fitted),
                "evidence_residual_gain": self._evidence_gain,
                "hier_fitted": bool(self.hier.fitted),
                "hier_phi_h_used": float(self.hier._phi_h),
                "hier_phi_a_used": float(self.hier._phi_a),
                "n_active_features": len(self.model.active_features),
                "n_evidence_features": len([f for f in self.model.active_features if f in EVIDENCE_FEATURES]),
                "frozen_market_alpha": self._frozen_market_alpha,
                "frozen_market_status": self.market.status,
                "n_pinned_regime_edges": 0 if self.ensemble.edges is None else len(self.ensemble.edges),
                "n_pinned_evidence_edges": 0 if self.ensemble.evidence_edges is None else len(self.ensemble.evidence_edges),
                "n_per_league_drift": len(self.drift_per_league)}

    def _score_mats(self, cal_feat):
        lh, la, _, _ = self.model.predict_intensity(cal_feat)
        mats = [self.model._joint_matrix(h, a, self.gate.state.T_score, self.copula_family,
                                          self.copula_theta, self.bp_lam3, self.joint_type,
                                          self.gate.state.rho, self.cfg.use_dc)
                for h, a in zip(lh, la)]
        return mats, lh, la

    def _cal_pool_replay(self, cal_feat):
        p = self.model.predict_1x2(cal_feat, self.gate.state.T_score, family=self.copula_family,
                                    theta_c=self.copula_theta, bp_l3=self.bp_lam3,
                                    joint=self.joint_type, use_dc=self.cfg.use_dc, rho=self.gate.state.rho)
        p = self._blend(p, cal_feat)
        p = self._apply_meta(p, cal_feat)
        p = self._apply_dirichlet(p)
        if self.evidence_residual is not None and self.evidence_residual.fitted:
            try: p = self.evidence_residual.predict(cal_feat, p)
            except Exception: pass
        alpha_used = float(self._frozen_market_alpha)
        if alpha_used <= 0:
            out = np.empty_like(p)
            for i, r in cal_feat.iterrows():
                pm = self._market_prob_from_row(r)
                out[i] = pm if pm is not None else p[i]
            return out
        # Vectorised pooling over market-present rows
        market_probs = [self._market_prob_from_row(r) for _, r in cal_feat.iterrows()]
        # Broadcast p to (1, n, 3) so we can reuse the batch helper
        p_draws_single = p[None, :, :]
        mixed = self._apply_market_pool_batch(p_draws_single, market_probs, alpha_used)[0]
        return mixed

    def _fit_cumulative_conformal(self, cal_feat):
        self.cumulative_conformal = {}
        if len(cal_feat) < 10: return
        team_idx = {}
        for i, r in cal_feat.iterrows():
            team_idx.setdefault(r.home, []).append((i, 0)); team_idx.setdefault(r.away, []).append((i, 1))
        lg, hi, ai, neutral, X = self.model._indices(cal_feat)
        rng = np.random.default_rng(MASTER_SEED + 13)
        S_cal = 40; S_sim = 50
        theta_draws = self.model.posterior_draws(S_cal, rng)
        phi_h_draws, phi_a_draws = self.model.posterior_phi_draws(S_cal, rng)
        n = len(cal_feat)
        lam_h, lam_a, _, _ = self.model.predict_intensity_batch(lg, hi, ai, neutral, X, theta_draws)
        hg = cal_feat.home_goals.to_numpy(int); ag = cal_feat.away_goals.to_numpy(int)
        for W in self.cfg.cumulative_windows:
            if n < W + 5: continue
            realized = []; pred_mean = []; draws_list = []; leagues = []; window_dates = []
            for uid, entries in team_idx.items():
                if len(entries) < W: continue
                entries_sorted = sorted(entries, key=lambda e: cal_feat.iloc[e[0]].date)
                n_win = len(entries_sorted) - W + 1
                idx_mat = np.zeros((n_win, W), dtype=int); side_mat = np.zeros((n_win, W), dtype=int)
                for s_i, start in enumerate(range(n_win)):
                    for k in range(W):
                        ii_, side = entries_sorted[start + k]
                        idx_mat[s_i, k] = ii_; side_mat[s_i, k] = side
                gf = np.where(side_mat == 0, hg[idx_mat], ag[idx_mat])
                obs_tot = gf.sum(axis=1)
                sim_totals = np.zeros((S_cal, n_win, S_sim))
                for s in range(S_cal):
                    lm = np.where(side_mat == 0, lam_h[s][idx_mat], lam_a[s][idx_mat])
                    phi_mat = np.where(side_mat == 0, phi_h_draws[s], phi_a_draws[s])
                    for w_i in range(n_win):
                        for k in range(W):
                            lam_k = max(float(lm[w_i, k]), EPS)
                            ph_k = max(float(phi_mat[w_i, k]), 0.5)
                            sim_totals[s, w_i] += rng.negative_binomial(ph_k, ph_k / (ph_k + lam_k), size=S_sim)
                for w_i in range(n_win):
                    realized.append(float(obs_tot[w_i]))
                    pm = float(np.mean([np.where(side_mat[w_i] == 0, lam_h[s][idx_mat[w_i]], lam_a[s][idx_mat[w_i]]).sum()
                                        for s in range(S_cal)]))
                    pred_mean.append(pm)
                    draws_list.append(sim_totals[:, w_i, :].ravel())
                    leagues.append(str(cal_feat.iloc[idx_mat[w_i, 0]].league))
                    window_dates.append(cal_feat.iloc[idx_mat[w_i, -1]].date)
            if len(realized) < 30: continue
            max_len = max(len(d) for d in draws_list)
            padded = np.zeros((len(draws_list), max_len))
            for i, d in enumerate(draws_list):
                padded[i, :len(d)] = d
                if len(d) < max_len: padded[i, len(d):] = d[-1]
            cc = TrueCumulativeConformal(alpha=self.cfg.conformal_alpha, window=W)
            cc.fit(np.asarray(realized), np.asarray(pred_mean), padded, np.asarray(leagues),
                   dates=np.asarray(window_dates))
            self.cumulative_conformal[W] = cc

    def _predict_raw(self, fixtures, light=False):
        x = fixtures.reset_index(drop=True)
        leaked = CausalContract.leak_check(x)
        if leaked: raise ValueError(f"forecast payload contains realized fields: {leaked}")
        S = max(20, int(self.cfg.n_posterior_samples))
        n = len(x)
        p_draws_struct, lam_h_draws, lam_a_draws = self._compute_posterior_draws(x, S)
        if self.challenger is not None and self.gate.state.challenger_alpha > 0:
            try:
                pc = self.challenger.predict(x); a = float(self.gate.state.challenger_alpha)
                p_draws_struct = np.stack(
                    [clip_matrix((1 - a) * p_draws_struct[s] + a * pc) for s in range(S)], axis=0)
            except Exception: pass
        if self.stacker is not None and self.stacker.model is not None and self.gate.state.stacker_alpha > 0:
            try:
                ps = self.stacker.predict(x); a = float(self.gate.state.stacker_alpha)
                p_draws_struct = np.stack(
                    [clip_matrix((1 - a) * p_draws_struct[s] + a * ps) for s in range(S)], axis=0)
            except Exception: pass
        p_stage1 = clip_matrix(p_draws_struct.mean(axis=0))
        if self.ensemble.fitted:
            try:
                experts = {"structural": p_stage1}
                if self.challenger is not None: experts["challenger"] = self.challenger.predict(x)
                if self.stacker is not None and self.stacker.model is not None: experts["stacker"] = self.stacker.predict(x)
                if self.hier.fitted:
                    try: experts["hierarchical"] = self.hier.probabilities(x, phi_h=self.model.phi_h,
                                                                            phi_a=self.model.phi_a, cap=self.model.cfg.score_cap)
                    except Exception: pass
                cv = (lam_h_draws.std(axis=0) / np.maximum(lam_h_draws.mean(axis=0), EPS)
                      + lam_a_draws.std(axis=0) / np.maximum(lam_a_draws.mean(axis=0), EPS)) / 2.0
                ep = x.get("evidence_present", pd.Series(0.0, index=x.index)).fillna(0.0).to_numpy()
                rk = [self.ensemble._regime(e, u) for e, u in zip(ep, cv)]
                p_ens = self.ensemble.predict(experts, rk)
                delta = p_ens - p_stage1
                for s in range(S): p_draws_struct[s] = clip_matrix(p_draws_struct[s] + delta)
                p_stage1 = p_ens
            except Exception: pass
        if self.gate.state.meta_alpha > 0 and self.meta.model is not None \
                and self.challenger is not None and self.stacker is not None:
            try:
                X_meta = np.hstack([p_stage1, self.challenger.predict(x), self.stacker.predict(x)])
                p_meta = self.meta.predict(X_meta); a = float(self.gate.state.meta_alpha)
                p_new = clip_matrix((1 - a) * p_stage1 + a * p_meta)
                delta = p_new - p_stage1
                for s in range(S): p_draws_struct[s] = clip_matrix(p_draws_struct[s] + delta)
                p_stage1 = p_new
            except Exception: pass
        p_draws_struct = self._apply_dirichlet_to_draws(p_draws_struct)
        p_stage1 = clip_matrix(p_draws_struct.mean(axis=0))
        if self.evidence_residual is not None and self.evidence_residual.fitted and self.cfg.enable_evidence_residual:
            try:
                p_ev = self.evidence_residual.predict(x, p_stage1)
                dlog = np.log(np.clip(p_ev, EPS, 1.0)) - np.log(np.clip(p_stage1, EPS, 1.0))
                for s in range(S):
                    lp = np.log(np.clip(p_draws_struct[s], EPS, 1.0)) + dlog
                    lp -= logsumexp(lp, axis=1, keepdims=True)
                    p_draws_struct[s] = np.exp(lp)
                p_stage1 = p_ev
            except Exception: pass
        market_statuses = []; pool_modes = []; market_probs = []
        for i, r in x.iterrows():
            pm = self._market_prob_from_row(r); status = "not_available"
            if pm is not None: status = "explicit" if pd.notna(r.get("market_p_home")) else "b365"
            _, mode = self.market.forecast(pm, p_stage1[i])
            pool_modes.append(mode); market_statuses.append(status); market_probs.append(pm)
        # Vectorised market pooling on draws
        p_draws_mixed = self._apply_market_pool_batch(p_draws_struct, market_probs, float(self.market.alpha))
        p_final = clip_matrix(p_draws_mixed.mean(axis=0))
        return self._assemble(x, p_stage1, p_final, p_draws_mixed, lam_h_draws, lam_a_draws,
                                market_statuses, pool_modes, light)

    def _assemble(self, x, p_pre, p_final, p_draws, lam_h_draws, lam_a_draws,
                   market_statuses, pool_modes, light):
        rows = []; L = self.model.cfg.score_cap + 1
        ii = np.arange(L)[:, None]; jj = np.arange(L)[None, :]
        for i, r in x.iterrows():
            p_lo = np.quantile(p_draws[:, i, :], 0.10, axis=0)
            p_hi = np.quantile(p_draws[:, i, :], 0.90, axis=0)
            lh_m = float(lam_h_draws[:, i].mean()); la_m = float(lam_a_draws[:, i].mean())
            J = self.model._joint_matrix(lh_m, la_m, self.gate.state.T_score, self.copula_family,
                                          self.copula_theta, self.bp_lam3, self.joint_type,
                                          self.gate.state.rho, self.cfg.use_dc)
            ipf_ok = True; ipf_dev = 0.0
            if self.cfg.use_joint_coherence:
                J2, conv, dev = project_to_joint(J, target_1x2=p_final[i], iters=1, tol=1e-9)
                ipf_ok = bool(conv); ipf_dev = float(dev)
                if conv: J = J2
            flat = J.ravel(); top_idx = np.argsort(flat)[::-1][:self.cfg.top_k_cs]
            top = [{"home_goals": int(q // L), "away_goals": int(q % L), "probability": float(flat[q])} for q in top_idx]
            over25 = float(J[ii + jj > 2].sum()); btts = float(J[(ii >= 1) & (jj >= 1)].sum())
            if abs(self.gate.state.T_ou - 1.0) > 1e-9: over25 = float(_temper_binary(np.array([over25]), self.gate.state.T_ou)[0])
            if abs(self.gate.state.T_btts - 1.0) > 1e-9: btts = float(_temper_binary(np.array([btts]), self.gate.state.T_btts)[0])
            cold = (str(r.home) not in self.model.teams) or (str(r.away) not in self.model.teams)
            dis = float(np.linalg.norm(p_pre[i] - p_final[i], ord=1) / 2)
            sel = self.selective.evaluate(p_final[i], dis)
            conf_in = None
            if self.conformal is not None:
                try:
                    m = self.conformal.mask(p_final[i][None, :], [r.league])
                    conf_in = bool(m[0, int(p_final[i].argmax())])
                except Exception: pass
            ou_in = bt_in = None
            if self.conformal_ou is not None:
                try:
                    m = self.conformal_ou.mask(np.array([over25]), [r.league])
                    ou_in = bool(m[0, 0 if over25 > 0.5 else 1])
                except Exception: pass
            if self.conformal_btts is not None:
                try:
                    m = self.conformal_btts.mask(np.array([btts]), [r.league])
                    bt_in = bool(m[0, 0 if btts > 0.5 else 1])
                except Exception: pass
            rows.append({
                "match_key": r.get("match_key"), "date": r.date, "league": r.league,
                "home": r.home, "away": r.away,
                "p_home": float(p_final[i, 0]), "p_draw": float(p_final[i, 1]), "p_away": float(p_final[i, 2]),
                "p_home_lo": float(p_lo[0]), "p_home_hi": float(p_hi[0]),
                "p_draw_lo": float(p_lo[1]), "p_draw_hi": float(p_hi[1]),
                "p_away_lo": float(p_lo[2]), "p_away_hi": float(p_hi[2]),
                "p_premarket_home": float(p_pre[i, 0]), "p_premarket_draw": float(p_pre[i, 1]),
                "p_premarket_away": float(p_pre[i, 2]),
                "market_status": market_statuses[i], "pool_mode": pool_modes[i],
                "market_alpha": float(self.market.alpha),
                "copula_theta": float(self.copula_theta), "copula_family": self.copula_family,
                "bp_lam3": float(self.bp_lam3), "joint_type": self.joint_type,
                "T_score": float(self.gate.state.T_score), "T_ou": float(self.gate.state.T_ou),
                "T_btts": float(self.gate.state.T_btts),
                "stacker_alpha": float(self.gate.state.stacker_alpha),
                "challenger_alpha": float(self.gate.state.challenger_alpha),
                "meta_alpha": float(self.gate.state.meta_alpha),
                "dirichlet_active": bool(self.dirichlet and self.dirichlet.fitted),
                "ensemble_active": bool(self.ensemble.fitted),
                "evidence_residual_active": bool(self.evidence_residual and self.evidence_residual.fitted),
                "hier_active": bool(self.hier.fitted),
                "lambda_home": lh_m, "lambda_away": la_m,
                "lambda_home_lo": float(np.quantile(lam_h_draws[:, i], 0.10)),
                "lambda_home_hi": float(np.quantile(lam_h_draws[:, i], 0.90)),
                "lambda_away_lo": float(np.quantile(lam_a_draws[:, i], 0.10)),
                "lambda_away_hi": float(np.quantile(lam_a_draws[:, i], 0.90)),
                "over25": over25, "btts": btts,
                "p_home_or_draw": float(p_final[i, 0] + p_final[i, 1]),
                "p_away_or_draw": float(p_final[i, 1] + p_final[i, 2]),
                "entropy": float(-np.sum(p_final[i] * np.log(np.maximum(p_final[i], EPS)))),
                "argmax": OUTCOME_INV[int(p_final[i].argmax())],
                "disagreement": dis, "selective_publish": sel["publish"], "selective_score": sel["score"],
                "cold_start": cold, "conformal_in_set": conf_in,
                "conformal_ou_in_set": ou_in, "conformal_btts_in_set": bt_in,
                "ipf_converged": ipf_ok, "ipf_max_dev": ipf_dev,
                "conformal_dirty": bool(self._conformal_dirty),
                "top_scores": top, "score_surface": (J.tolist() if not light else None),
            })
        return pd.DataFrame(rows), ([r["score_surface"] for r in rows] if not light else None)

    def predict(self, fixtures, light=False):
        if not self.fitted: raise RuntimeError("engine not fitted")
        x_norm = CausalContract.normalize(fixtures, False)
        feats = self.feature_builder.build(x_norm, update_state=False)
        return self._predict_raw(feats, light=light)

    def refit_hierarchical(self, historical_frame, cutoff=None):
        if not self.cfg.hier.enabled: return {"fitted": False, "reason": "disabled"}
        return self.hier.fit(historical_frame, cutoff=cutoff, phi_h=self.model.phi_h, phi_a=self.model.phi_a)

    def assimilate(self, results, cutoff=None, strict_teams=None):
        if results is None or len(results) == 0: return {"updated": False, "n": 0}
        strict = self.cfg.strict_team_boundary if strict_teams is None else bool(strict_teams)
        rr = CausalContract.normalize(results, True)
        try: _ = self.feature_builder.build(rr, update_state=True)
        except Exception as e: log.warning(f"[assimilate] feature update failed: {e}")
        by_key = {}
        for j in reversed(self.journal): by_key.setdefault(str(j.get("match_key")), j)
        cutoff_ts = None
        if cutoff is not None:
            cutoff_ts = pd.Timestamp(cutoff)
            if cutoff_ts.tzinfo is not None: cutoff_ts = cutoff_ts.tz_convert("UTC").tz_localize(None)
        champ = self.registry.champion["name"] if self.registry.champion else None
        n = 0; skipped_unseen = 0; expanded = 0
        old_alpha = float(self.market.alpha)
        batch_selective = []
        for _, r in rr.iterrows():
            key = str(r.match_key)
            if key in self.assimilated: continue
            r_date = pd.Timestamp(r.date)
            if r_date.tzinfo is not None: r_date = r_date.tz_convert("UTC").tz_localize(None)
            if cutoff_ts is not None and r_date > cutoff_ts: continue
            j = by_key.get(key)
            if j is None: continue
            unseen = [t for t in (str(r.home), str(r.away)) if t not in self.model.teams]
            if unseen and strict:
                log.warning(f"[assimilate] skipping {key}: unseen teams {unseen}")
                skipped_unseen += 1; continue
            elif unseen and self.cfg.allow_team_expansion:
                lg_home = str(r.league)
                league_of = {str(t): lg_home for t in unseen}
                rep = self.model.expand_teams(unseen, league_of)
                expanded += int(rep.get("added", 0))
                self._conformal_dirty = True
                log.warning(f"[assimilate] expanded teams by {rep.get('added', 0)}; refit required")
            elif unseen:
                log.warning(f"[assimilate] skipping {key}: unseen teams {unseen} (expansion disabled)")
                skipped_unseen += 1; continue
            pm = None
            mcols = [j.get("market_p_home"), j.get("market_p_draw"), j.get("market_p_away")]
            if all(v is not None and np.isfinite(v) for v in mcols):
                try: pm = clip_probs(mcols)
                except Exception: pm = None
            ps = np.array([j["p_premarket_home"], j["p_premarket_draw"], j["p_premarket_away"]])
            self.market.update(key, pm, ps, int(r.y), date=r_date)
            ll = -math.log(np.clip(float(j["p_home"] if int(r.y) == 0 else
                                           j["p_draw"] if int(r.y) == 1 else j["p_away"]), EPS, 1.0))
            self.drift.update(ll)
            if self.cfg.per_league_drift:
                lg = str(r.league)
                if lg not in self.drift_per_league:
                    self.drift_per_league[lg] = PageHinkley.warm_started([])
                self.drift_per_league[lg].update(ll)
            if champ: self.registry.record(champ, ll, date=r_date)
            batch_selective.append((np.array([j["p_home"], j["p_draw"], j["p_away"]]),
                                     int(r.y), float(j.get("disagreement", 0.0)), r_date))
            try:
                pre = np.array([j["p_premarket_home"], j["p_premarket_draw"], j["p_premarket_away"]])
                pv = [r.get("market_p_home"), r.get("market_p_draw"), r.get("market_p_away")]
                if all(pd.notna(v) for v in pv): self.clv.record(key, pre, clip_probs(pv), "1x2")
            except Exception: pass
            for t in (str(r.home), str(r.away)):
                ti = self.model.teams.get(t, -1)
                if ti >= 0 and ti < self.model.team_games.size:
                    self.model.team_games[ti] += 1.0
            self.assimilated.add(key); n += 1
        if batch_selective:
            self.selective.observe_batch(batch_selective)
        if n > 0:
            _refresh_pooled(self.model)
            self._assimilation_count += n
            if abs(self.market.alpha - old_alpha) > 0.02:
                self._conformal_dirty = True
                log.warning(f"[assimilate] market alpha drifted {old_alpha:.3f} → {self.market.alpha:.3f}; conformal sets stale")
        if self.refit_scheduler.should_refit(self.drift):
            self.refit_scheduler.request("PH alert (global)")
        if self.cfg.per_league_drift:
            for lg, ph in self.drift_per_league.items():
                if ph.n > ph.warmup and ph.alerts > 0:
                    self.refit_scheduler.request(f"PH alert ({lg})")
                    break
        return {"updated": True, "n": n, "skipped_unseen": skipped_unseen, "expanded_teams": expanded,
                "market_alpha": self.market.alpha, "market_status": self.market.status,
                "drift_alerts": self.drift.alerts,
                "per_league_alerts": {k: v.alerts for k, v in self.drift_per_league.items()},
                "needs_refit": bool(self.model.needs_refit),
                "conformal_dirty": bool(self._conformal_dirty),
                "refit_requested": bool(self.refit_scheduler.requests),
                "clv": self.clv.summary(),
                "assimilation_count_since_refresh": self._assimilation_count}

    def cumulative_event_probability(self, fixtures, n_matches, target_ga, team_name, n_draws=500):
        tgt = str(team_name).strip().upper()
        sub = fixtures.iloc[:n_matches].copy()
        x_norm = CausalContract.normalize(sub, False)
        feats = self.feature_builder.build(x_norm, update_state=False)
        S = int(max(20, n_draws)); rng = np.random.default_rng(MASTER_SEED)
        theta_draws = self.model.posterior_draws(S, rng)
        phi_h_draws, phi_a_draws = self.model.posterior_phi_draws(S, rng)
        gh_draws, ga_draws = self.model.posterior_gamma_draws(S, rng)
        lg, hi, ai, neutral, X = self.model._indices(feats)
        sides = []
        for i in range(n_matches):
            h = str(sub.iloc[i].get("home", "")).strip().upper(); a = str(sub.iloc[i].get("away", "")).strip().upper()
            if h == tgt: sides.append(0)
            elif a == tgt: sides.append(1)
            else: raise ValueError(f"team {tgt} not in fixture {i}")
        cap = self.model.cfg.score_cap
        # Precompute joint matrices for all (s, k) in a batched fashion
        lam_h_s, lam_a_s, _, _ = self.model.predict_intensity_batch(lg, hi, ai, neutral, X, theta_draws)
        log_lh = np.log(np.maximum(lam_h_s, EPS)); log_la = np.log(np.maximum(lam_a_s, EPS))
        adj_h = np.where(np.abs(gh_draws)[:, None] > 1e-9,
                         np.exp(gh_draws[:, None] * (log_lh - self.model.log_lambda_total_mean)), 1.0)
        adj_a = np.where(np.abs(ga_draws)[:, None] > 1e-9,
                         np.exp(ga_draws[:, None] * (log_la - self.model.log_lambda_total_mean)), 1.0)
        phi_h_eff = np.clip(phi_h_draws[:, None] * adj_h, self.model.cfg.nb_phi_min, self.model.cfg.nb_phi_max)
        phi_a_eff = np.clip(phi_a_draws[:, None] * adj_a, self.model.cfg.nb_phi_min, self.model.cfg.nb_phi_max)
        # Slice to the first n_matches
        lam_h_s = lam_h_s[:, :n_matches]; lam_a_s = lam_a_s[:, :n_matches]
        phi_h_eff = phi_h_eff[:, :n_matches]; phi_a_eff = phi_a_eff[:, :n_matches]
        J_all = _batch_joint_1x2(lam_h_s, lam_a_s, self.gate.state.T_score, self.copula_family,
                                   self.copula_theta, self.bp_lam3, self.joint_type,
                                   self.gate.state.rho, self.cfg.use_dc,
                                   phi_h_eff, phi_a_eff, cap, max_elems=self.cfg.joint_batch_max_elems)
        # J_all: (S, n_matches, 3) — but we need full score distributions, not just 1X2.
        # Reconstruct the flat joint by re-running with score output if needed. Fall back
        # to per-sample batching but with inverse-CDF sampling for speed.
        totals = np.zeros(S, dtype=int)
        side_arr = np.asarray(sides, dtype=int)
        for s in range(S):
            for k in range(n_matches):
                lh = float(lam_h_s[s, k]); la = float(lam_a_s[s, k])
                J = self.model._joint_matrix(lh, la, self.gate.state.T_score, self.copula_family,
                                              self.copula_theta, self.bp_lam3, self.joint_type,
                                              self.gate.state.rho, self.cfg.use_dc,
                                              phi_h_override=float(phi_h_eff[s, k]),
                                              phi_a_override=float(phi_a_eff[s, k]))
                flat = J.ravel()
                cdf = np.cumsum(flat)
                cdf /= cdf[-1]
                u = float(rng.random())
                idx = int(np.searchsorted(cdf, u))
                gh_sample = idx // (cap + 1); ga_sample = idx % (cap + 1)
                totals[s] += int(gh_sample if side_arr[k] == 0 else ga_sample)
        p_target = float(np.mean(totals == target_ga)) if target_ga >= 0 else 0.0
        out = {"p_target": p_target, "n_matches": n_matches, "target_ga": target_ga,
                "team": tgt, "n_draws_used": int(S), "joint_type_used": self.joint_type}
        if n_matches in self.cumulative_conformal:
            try:
                lg_arr = np.array([str(sub.iloc[0].get("league", "GLOBAL"))])
                lo, hi_ = self.cumulative_conformal[n_matches].interval(
                    np.array([float(totals.mean())]), totals[None, :], lg_arr)
                out["interval"] = (float(lo[0]), float(hi_[0]))
            except Exception as e: out["interval_error"] = str(e)
        return out

    def conformal_coverage_report(self, cal_frame):
        if self.conformal is None: return {"error": "no_conformal"}
        x = CausalContract.normalize(cal_frame, True)
        feats = self.feature_builder.build(x, update_state=False)
        p = self._cal_pool_replay(feats)
        y = feats.y.to_numpy(int); lg = feats.league.values
        out = {}
        try:
            out["split_1x2_coverage"] = self.conformal.coverage(p, y, lg)
            out["target_1x2_coverage"] = 1.0 - self.cfg.conformal_alpha
        except Exception as e: out["split_1x2_error"] = str(e)
        if self.conformal_ou is not None:
            try:
                mats, _, _ = self._score_mats(feats)
                L = mats[0].shape[0]; ii = np.arange(L)[:, None]; jj = np.arange(L)[None, :]
                ou = np.array([float(J[ii + jj > 2].sum()) for J in mats])
                if abs(self.gate.state.T_ou - 1.0) > 1e-9: ou = _temper_binary(ou, self.gate.state.T_ou)
                y_ou = ((feats.home_goals + feats.away_goals) > 2.5).astype(float).values
                out["split_ou_coverage"] = self.conformal_ou.coverage(ou, y_ou, lg)
                out["target_ou_coverage"] = 1.0 - self.cfg.conformal_alpha
            except Exception as e: out["split_ou_error"] = str(e)
        if self.conformal_btts is not None:
            try:
                mats, _, _ = self._score_mats(feats)
                L = mats[0].shape[0]; ii = np.arange(L)[:, None]; jj = np.arange(L)[None, :]
                bt = np.array([float(J[(ii >= 1) & (jj >= 1)].sum()) for J in mats])
                if abs(self.gate.state.T_btts - 1.0) > 1e-9: bt = _temper_binary(bt, self.gate.state.T_btts)
                y_bt = ((feats.home_goals >= 1) & (feats.away_goals >= 1)).astype(float).values
                out["split_btts_coverage"] = self.conformal_btts.coverage(bt, y_bt, lg)
                out["target_btts_coverage"] = 1.0 - self.cfg.conformal_alpha
            except Exception as e: out["split_btts_error"] = str(e)
        return out

    def stake(self, fixtures, odds, bankroll=1.0):
        pred_df, _ = self.predict(fixtures, light=False)
        odds = np.asarray(odds, float)
        staker = PortfolioStaking(bankroll=bankroll)
        x_norm = CausalContract.normalize(fixtures, False)
        feats = self.feature_builder.build(x_norm, update_state=False)
        S = max(20, int(self.cfg.n_posterior_samples))
        p_draws_raw, _, _ = self._compute_posterior_draws(feats, S)
        if self.challenger is not None and self.gate.state.challenger_alpha > 0:
            try:
                pc = self.challenger.predict(feats); a = float(self.gate.state.challenger_alpha)
                p_draws_raw = np.stack([clip_matrix((1 - a) * p_draws_raw[s] + a * pc) for s in range(S)], axis=0)
            except Exception: pass
        if self.stacker is not None and self.stacker.model is not None and self.gate.state.stacker_alpha > 0:
            try:
                ps = self.stacker.predict(feats); a = float(self.gate.state.stacker_alpha)
                p_draws_raw = np.stack([clip_matrix((1 - a) * p_draws_raw[s] + a * ps) for s in range(S)], axis=0)
            except Exception: pass
        p_draws_raw = self._apply_dirichlet_to_draws(p_draws_raw)
        p_mean_final = pred_df[["p_home", "p_draw", "p_away"]].to_numpy(float)
        mask = None
        if self.conformal is not None:
            try: mask = self.conformal.mask(p_mean_final, pred_df.league.values)
            except Exception: mask = None
        ids = [str(i) for i in range(len(fixtures))]
        recs = staker.optimize(ids, p_mean_final, p_draws_raw, odds, conformal_mask=mask)
        return pd.DataFrame(recs)

    def report(self, fixtures, light=False):
        df, _ = self.predict(fixtures, light=light)
        for _, row in df.iterrows():
            entry = {}
            for k, v in row.items():
                if isinstance(v, (list, np.ndarray, tuple, dict)): entry[k] = None
                else: entry[k] = v.item() if isinstance(v, (np.floating, np.integer)) else v
            self.journal.append(entry)
        if len(self.journal) > 20000: self.journal = self.journal[-10000:]
        return df

    def save(self, path, compress=None):
        compress = self.cfg.compress_save if compress is None else bool(compress)
        payload = {"version": VERSION, "schema": SCHEMA_VERSION, "cfg": asdict(self.cfg),
                   "model": self.model.state(), "hier": self.hier.state(),
                   "challenger": self.challenger.state() if self.challenger else None,
                   "stacker": self.stacker.state() if self.stacker else None,
                   "meta": self.meta.state(), "ensemble": self.ensemble.state(),
                   "dirichlet": self.dirichlet.state() if self.dirichlet else None,
                   "gate": asdict(self.gate.state), "market": self.market.state(),
                   "selective": self.selective.state(), "drift": self.drift.state(),
                   "drift_per_league": {k: v.state() for k, v in self.drift_per_league.items()},
                   "registry": self.registry.state(), "refit": self.refit_scheduler.state(),
                   "clv": self.clv.state(),
                   "conformal": self.conformal.state() if self.conformal else None,
                   "conformal_ou": self.conformal_ou.state() if self.conformal_ou else None,
                   "conformal_btts": self.conformal_btts.state() if self.conformal_btts else None,
                   "cumulative_conformal": {int(k): v.state() for k, v in self.cumulative_conformal.items()},
                   "journal": self.journal[-5000:], "assimilated": sorted(self.assimilated),
                   "fitted": self.fitted, "copula_theta": float(self.copula_theta),
                   "copula_family": self.copula_family, "bp_lam3": float(self.bp_lam3),
                   "joint_type": self.joint_type,
                   "feature_builder_state": self.feature_builder.state(),
                   "evidence_residual": self.evidence_residual.state() if self.evidence_residual else None,
                   "evidence_gain": self._evidence_gain,
                   "frozen_market_alpha": float(self._frozen_market_alpha),
                   "conformal_dirty": bool(self._conformal_dirty),
                   "assimilation_count": int(self._assimilation_count)}
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write(path, blob, compress=compress)
        return {"path": str(path), "bytes": len(blob), "compressed": compress}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f: raw = f.read()
        if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
        payload = pickle.loads(raw)
        if int(payload.get("schema", 0)) != SCHEMA_VERSION:
            raise ValueError(f"schema mismatch: {payload.get('schema')} != {SCHEMA_VERSION}")
        cfg_d = payload.get("cfg", {}); maher = cfg_d.get("maher", {})
        cfg = EngineConfig(maher=MaherConfig(**maher))
        for k, v in cfg_d.items():
            if k == "maher": continue
            if hasattr(cfg, k):
                try: setattr(cfg, k, v)
                except Exception: pass
        e = cls(cfg)
        e.model = MaherPoissonModel.restore(payload["model"])
        e.hier = HierarchicalDynamicGoalState.restore(payload.get("hier", {}))
        if e.challenger is not None and payload.get("challenger") is not None:
            e.challenger = OutcomeChallenger.restore(payload["challenger"])
        if payload.get("stacker") is not None: e.stacker = ResidualStacker.restore(payload["stacker"])
        if payload.get("meta") is not None: e.meta = StackingMetaLearner.restore(payload["meta"])
        if payload.get("ensemble") is not None: e.ensemble = RegimeConditionalEnsemble.restore(payload["ensemble"])
        if payload.get("dirichlet") is not None and e.dirichlet is not None:
            e.dirichlet = DirichletCalibrator.restore(payload["dirichlet"])
        gs = payload["gate"]; st = CalibrationState()
        for k, v in gs.items():
            if hasattr(st, k): setattr(st, k, v)
        e.gate.state = st
        e.copula_theta = float(payload.get("copula_theta", 0.0))
        e.copula_family = str(payload.get("copula_family", "frank"))
        e.bp_lam3 = float(payload.get("bp_lam3", 0.0)); e.joint_type = str(payload.get("joint_type", "copula"))
        e.market = MarketTrustController.restore(payload["market"])
        e.selective = SelectiveGovernor.restore(payload["selective"])
        e.drift = PageHinkley.restore(payload["drift"])
        e.drift_per_league = {str(k): PageHinkley.restore(v) for k, v in payload.get("drift_per_league", {}).items()}
        e.registry = ModelRegistry.restore(payload.get("registry", {}))
        e.refit_scheduler = AutoRefitScheduler.restore(payload.get("refit", {}))
        if payload.get("clv"): e.clv = CLVTracker.restore(payload["clv"])
        if payload.get("conformal") is not None: e.conformal = PerLeagueConformalSets.restore(payload["conformal"])
        if payload.get("conformal_ou") is not None: e.conformal_ou = PerLeagueBinaryConformalSets.restore(payload["conformal_ou"])
        if payload.get("conformal_btts") is not None: e.conformal_btts = PerLeagueBinaryConformalSets.restore(payload["conformal_btts"])
        e.cumulative_conformal = {int(k): TrueCumulativeConformal.restore(v) for k, v in payload.get("cumulative_conformal", {}).items()}
        e.journal = list(payload.get("journal", [])); e.assimilated = set(payload.get("assimilated", []))
        e.fitted = bool(payload.get("fitted", False))
        e._frozen_market_alpha = float(payload.get("frozen_market_alpha", 0.0))
        e._conformal_dirty = bool(payload.get("conformal_dirty", False))
        e._assimilation_count = int(payload.get("assimilation_count", 0))
        e._evidence_gain = payload.get("evidence_gain")
        if payload.get("feature_builder_state"):
            e.feature_builder = CausalFeatureBuilder.from_state(payload["feature_builder_state"])
        if payload.get("evidence_residual") is not None:
            e.evidence_residual = CrossFittedEvidenceResidualExpert.restore(payload["evidence_residual"])
        return e


def _refresh_pooled(model):
    n_t = model.n_t
    if n_t == 0 or model.team_prior_attack.size != n_t: return
    model.team_prior_attack = np.full(n_t, model.attack_mean)
    model.team_prior_defence = np.full(n_t, model.defence_mean)
    for ti, li in model.team_league.items():
        if li in model.league_attack_mean: model.team_prior_attack[ti] = model.league_attack_mean[li]
        if li in model.league_defence_mean: model.team_prior_defence[ti] = model.league_defence_mean[li]
    if model.n_l > 0:
        model.league_prior_attack = np.array(
            [model.league_attack_mean.get(li, model.attack_mean) for li in range(model.n_l)])
        model.league_prior_defence = np.array(
            [model.league_defence_mean.get(li, model.defence_mean) for li in range(model.n_l)])

# ============================================================================
# 16. Tests
# ============================================================================

def make_synthetic(n=900, seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    leagues = ["E0", "SP1", "I1", "N1"]; teams = [f"{l}_{i}" for l in leagues for i in range(8)]
    rows = []; start = pd.Timestamp("2021-08-01")
    atk = {t: rng.normal(0, 0.30) for t in teams}; dfn = {t: rng.normal(0, 0.25) for t in teams}
    for m in range(n):
        lg = leagues[m % len(leagues)]; pool = [t for t in teams if t.startswith(lg)]
        h, a = rng.choice(pool, 2, replace=False)
        d = start + pd.Timedelta(days=int(m * 1.3))
        neutral = 1.0 if rng.random() < 0.03 else 0.0
        lh = 0.30 + atk[h] - dfn[a] + (0.10 * (1 - neutral)); la = 0.15 + atk[a] - dfn[h]
        lam = np.exp(lh); mu = np.exp(la); hg = int(rng.poisson(lam)); ag = int(rng.poisson(mu))
        rows.append({"date": d, "league": lg, "home": h, "away": a, "home_goals": hg, "away_goals": ag,
                     "is_neutral": neutral,
                     "market_p_home": np.clip(lam / (lam + mu + 0.6), 0.05, 0.9),
                     "market_p_draw": np.clip(0.6 / (lam + mu + 0.6), 0.05, 0.9),
                     "market_p_away": np.clip(mu / (lam + mu + 0.6), 0.05, 0.9)})
    return pd.DataFrame(rows)

def _test_invariants():
    issues = []
    # Copula independence / positive control
    rng = np.random.default_rng(0)
    lam_h = np.full(600, 1.4); lam_a = np.full(600, 1.1); phi_c = 8.0
    obs_h = rng.negative_binomial(phi_c, phi_c / (phi_c + 1.4), size=600)
    obs_a = rng.negative_binomial(phi_c, phi_c / (phi_c + 1.1), size=600)
    th = fit_copula_theta(lam_h, lam_a, obs_h, obs_a, phi_c, phi_c, cap=20)
    if not (abs(th) < 0.5): issues.append(f"copula independence: |theta|={abs(th):.3f}")
    mh = nb_pmf(1.4, phi_c, 20); ma = nb_pmf(1.1, phi_c, 20)
    J_true = copula_joint(mh, ma, 2.0, "frank"); flat = J_true.ravel() / J_true.ravel().sum()
    if abs(flat.sum() - 1) > 1e-8: issues.append("copula positive-control pmf not normalised")
    idx = rng.choice(len(flat), size=600, p=flat); gi = idx // 21; gj = idx % 21
    th_pos = fit_copula_theta(np.full(600, 1.4), np.full(600, 1.1), gi, gj, phi_c, phi_c, cap=20)
    if not (abs(th_pos - 2.0) < 0.6): issues.append(f"copula positive control: {th_pos:.3f}")
    # BP recovery
    rng = np.random.default_rng(1); l3t = 0.25
    h_obs = []; a_obs = []
    for _ in range(500):
        k3 = rng.poisson(l3t); k1 = rng.poisson(1.4 - l3t); k2 = rng.poisson(1.1 - l3t)
        h_obs.append(k1 + k3); a_obs.append(k2 + k3)
    l3h = fit_bp_lambda3(np.full(500, 1.4), np.full(500, 1.1), np.array(h_obs), np.array(a_obs), cap=20)
    if not (abs(l3h - l3t) < 0.15): issues.append(f"BP recovery: {l3h:.3f}")
    # Leak linter
    for name in ("goal_form_diff", "expected_goals", "team_xg_goals", "form_points_diff10", "attack_defence_mismatch"):
        df_test = pd.DataFrame({name: [0.0]})
        if CausalContract.leak_check(df_test):
            issues.append(f"leak linter false-positive: {name}")
    leaky = pd.DataFrame({"home_goals": [1], "away_goals": [0], "y": [0], "outcome": [0], "result": [0]})
    hits2 = CausalContract.leak_check(leaky)
    if not {"home_goals", "away_goals", "y", "outcome", "result"}.issubset(set(hits2)):
        issues.append(f"leak linter missed: {hits2}")
    # Market EWMA recovery
    rng2 = np.random.default_rng(7)
    n = 500; teams = [f"T{i}" for i in range(6)]
    rows = []; start = pd.Timestamp("2020-01-01")
    for i in range(n):
        h = teams[i % 6]; a = teams[(i + 1) % 6]
        rows.append({"date": start + pd.Timedelta(days=int(i * 3)),
                     "home": h, "away": a, "league": "L",
                     "market_p_home": 0.4 + 0.01 * ((i % 5) - 2),
                     "market_p_draw": 0.25, "market_p_away": 0.35 - 0.01 * ((i % 5) - 2)})
    df_m = pd.DataFrame(rows)
    _, _, _, snap, _ = _compute_rolling_market_strength(df_m, return_final=True)
    if not any(abs(v) > 1e-9 for v in snap.values()):
        issues.append("market strength EWMA identically zero")
    # Kalman observation variance coherence with fitted phi
    hcfg = HierStateConfig(prior_dispersion=8.0, use_fitted_dispersion=True)
    hier = HierarchicalDynamicGoalState(hcfg)
    hier._phi_h = 8.0; hier._phi_a = 8.0
    oh_small = hier._observation_var(0.5, side="home"); oh_large = hier._observation_var(3.0, side="home")
    if not (oh_small > oh_large):
        issues.append(f"Kalman obs var not decreasing in lambda: {oh_small:.4f} vs {oh_large:.4f}")
    expected_small = 1.0 / 0.5 + 1.0 / 8.0
    if abs(oh_small - expected_small) > 1e-9:
        issues.append(f"Kalman obs var delta-method mismatch: {oh_small:.4f} vs {expected_small:.4f}")
    # Fitted phi flows into hier
    raw = make_synthetic(250, seed=99)
    m_test = MaherPoissonModel()
    m_test.fit(CausalContract.normalize(raw.iloc[:150], True))
    hier2 = HierarchicalDynamicGoalState()
    rep_h = hier2.fit(raw, cutoff=raw["date"].max() + pd.Timedelta(days=1),
                       phi_h=m_test.phi_h, phi_a=m_test.phi_a)
    if rep_h.get("fitted"):
        if abs(hier2._phi_h - m_test.phi_h) > 1e-9:
            issues.append(f"hier phi_h not wired: {hier2._phi_h} vs {m_test.phi_h}")
    # MC-vs-point mean
    hier3 = HierarchicalDynamicGoalState()
    rep = hier3.fit(raw, cutoff=raw["date"].max() + pd.Timedelta(days=1))
    if rep.get("fitted"):
        test = raw.tail(20).copy()
        lh_mc, la_mc = hier3.predict_mc(test, n_samples=500)
        lh_pt = np.zeros(len(test)); la_pt = np.zeros(len(test))
        for i, (_, r) in enumerate(test.iterrows()):
            comp = str(r.get("competition", r.get("league", "UNKNOWN")))
            home = str(r.get("home_team", r.get("home"))); away = str(r.get("away_team", r.get("away")))
            hs = hier3._state(comp, home); aas = hier3._state(comp, away)
            lh_pt[i] = math.exp(np.clip(hier3.global_log_goal + hier3.home_adv[comp] + hs["attack"] - aas["defence"],
                                          math.log(0.05), math.log(6.0)))
            la_pt[i] = math.exp(np.clip(hier3.global_log_goal + aas["attack"] - hs["defence"],
                                          math.log(0.05), math.log(6.0)))
        rel_h = np.abs(lh_mc.mean(axis=0) - lh_pt) / np.maximum(lh_pt, 0.1)
        rel_a = np.abs(la_mc.mean(axis=0) - la_pt) / np.maximum(la_pt, 0.1)
        if rel_h.max() > 0.05 or rel_a.max() > 0.05:
            issues.append(f"MC-vs-point mean > 5%: {rel_h.max():.3f}/{rel_a.max():.3f}")
    # Ensemble with evidence quantile edges
    ens = RegimeConditionalEnsemble(n_buckets=3, evidence_buckets=2)
    experts = {"a": clip_matrix(np.random.dirichlet([2, 2, 2], size=40)),
               "b": clip_matrix(np.random.dirichlet([2, 2, 2], size=40))}
    y = np.random.randint(0, 3, 40)
    cv = np.random.rand(40); ev = np.random.rand(40)
    rep2 = ens.fit(experts, y, [ens._regime(e, u) for e, u in zip(ev, cv)],
                    cv_for_edges=cv, evidence_for_edges=ev, min_rows_per_regime=5)
    if rep2.get("fitted"):
        if ens.evidence_edges is None or len(ens.evidence_edges) < 3:
            issues.append("ensemble evidence_edges not built")
        p = ens.predict(experts, [ens._regime(e, u) for e, u in zip(ev, cv)])
        if np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-6: issues.append("ensemble rows not normalized")
    st = ens.state(); ens2 = RegimeConditionalEnsemble.restore(st)
    if ens2.fitted:
        p2 = ens2.predict(experts, [ens2._regime(e, u) for e, u in zip(ev, cv)])
        p1 = ens.predict(experts, [ens._regime(e, u) for e, u in zip(ev, cv)])
        if np.max(np.abs(p1 - p2)) > 1e-9: issues.append("ensemble restore mismatch")
        if not np.allclose(ens.evidence_edges, ens2.evidence_edges): issues.append("ensemble evidence_edges not preserved")
    # clip_matrix NaN safety and strict mode
    test_nan = clip_matrix(np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]]))
    if not np.allclose(test_nan[0], 1.0 / 3.0): issues.append("clip_matrix NaN row not uniform")
    try:
        clip_matrix(np.array([[np.nan, np.nan, np.nan]]), strict=True)
        issues.append("clip_matrix strict did not raise on all-NaN rows")
    except ValueError: pass
    # Copula/DC mutual exclusion
    m = MaherPoissonModel()
    J1 = m._joint_matrix(1.4, 1.1, 1.0, "frank", 2.0, 0.0, "copula", 0.15, True)
    J2 = m._joint_matrix(1.4, 1.1, 1.0, "frank", 2.0, 0.0, "copula", 0.0, False)
    if not np.allclose(J1, J2): issues.append("copula+DC mutual exclusion violated")
    # Market α bootstrap under pure noise
    mtc = MarketTrustController(min_n=30, window=80, improvement=0.001,
                                 bootstrap_block_days=7, bootstrap_n=400)
    rng3 = np.random.default_rng(101)
    for k in range(120):
        pm = np.array([0.4, 0.3, 0.3]); ps = np.clip(pm + rng3.normal(0, 0.15, 3), 0.05, None); ps /= ps.sum()
        y_true = int(rng3.choice(3, p=pm))
        mtc.update(f"k{k}", pm, ps, y_true, date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=k))
    if mtc.alpha > 0:
        issues.append(f"market α adopted on pure noise: {mtc.alpha:.4f} status={mtc.status}")
    mtc2 = MarketTrustController.restore(mtc.state())
    if abs(mtc2.improvement - mtc.improvement) > 1e-12:
        issues.append(f"MarketTrustController.improvement not persisted")
    # MarketTrustController alpha range guard
    mtc3 = MarketTrustController()
    mtc3.alpha = 0.50  # out of range
    _, mode = mtc3.forecast(np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.3, 0.3]))
    if mode != "market_anchor_alpha_range":
        issues.append(f"MarketTrustController alpha range guard failed: {mode}")
    # expand_teams arithmetic + predictive continuity
    m2 = MaherPoissonModel(); m2.fit(make_synthetic(300, seed=17))
    old_n_t = m2.n_t
    old_theta = m2.theta.copy(); old_blocks = {k: (s.start, s.stop) for k, s in m2.design_blocks.items()}
    old_teams = dict(m2.teams)
    rep_ex = m2.expand_teams(["BRAND_NEW_TEAM"], {"BRAND_NEW_TEAM": "E0"})
    if rep_ex.get("added") != 1: issues.append(f"expand_teams added {rep_ex.get('added')} teams")
    if m2.n_t != old_n_t + 1: issues.append(f"expand_teams n_t mismatch")
    b = m2.design_blocks
    if b["attack"].stop != b["defence"].start: issues.append("expand_teams: attack.stop != defence.start")
    if b["defence"].stop != b["beta_h"].start: issues.append("expand_teams: defence.stop != beta_h.start")
    if b["beta_h"].stop != b["beta_a"].start: issues.append("expand_teams: beta_h.stop != beta_a.start")
    if b["beta_a"].stop != m2.n_params: issues.append(f"expand_teams: beta_a.stop != n_params")
    if m2.theta.shape[0] != m2.n_params: issues.append("expand_teams: theta length != n_params")
    for t, ti in old_teams.items():
        att_old = old_theta[old_blocks["attack"][0] + ti]; att_new = m2.theta[b["attack"].start + ti]
        if abs(att_old - att_new) > 1e-12: issues.append(f"expand_teams: attack for {t} corrupted")
        dfn_old = old_theta[old_blocks["defence"][0] + ti]; dfn_new = m2.theta[b["defence"].start + ti]
        if abs(dfn_old - dfn_new) > 1e-12: issues.append(f"expand_teams: defence for {t} corrupted")
    # Predictive path after expansion
    test_pred_frame = make_synthetic(20, seed=101).iloc[:20]
    try:
        p_after = m2.predict_1x2(CausalContract.normalize(test_pred_frame, False))
        if not (np.isfinite(p_after).all() and np.allclose(p_after.sum(axis=1), 1.0, atol=1e-6)):
            issues.append("expand_teams: predict_1x2 after expansion returns invalid probabilities")
    except Exception as e:
        issues.append(f"expand_teams: predict_1x2 raised after expansion: {e}")
    # Vectorised intensity batch equivalence
    synth = make_synthetic(60, seed=23)
    xs = CausalContract.normalize(synth, False)
    m3 = MaherPoissonModel(); m3.fit(make_synthetic(200, seed=31))
    lg, hi, ai, neutral, X = m3._indices(xs)
    theta_batch = m3.posterior_draws(3, np.random.default_rng(11))
    lam_h_b, lam_a_b, _, _ = m3.predict_intensity_batch(lg, hi, ai, neutral, X, theta_batch)
    for s in range(3):
        lh_s, la_s, _, _ = m3.predict_intensity_precomp(lg, hi, ai, neutral, X, theta_batch[s])
        if not np.allclose(lh_s, lam_h_b[s], rtol=1e-9, atol=1e-9):
            issues.append(f"predict_intensity_batch home mismatch at s={s}")
        if not np.allclose(la_s, lam_a_b[s], rtol=1e-9, atol=1e-9):
            issues.append(f"predict_intensity_batch away mismatch at s={s}")
    # Chunked joint equivalence
    ph_c = m3.phi_h_vec(lam_h_b); pa_c = m3.phi_a_vec(lam_a_b)
    p_full = _batch_joint_1x2(lam_h_b, lam_a_b, 1.0, "frank", 0.0, 0.0, "none", 0.0, False,
                                ph_c, pa_c, m3.cfg.score_cap, max_elems=10**12)
    p_chunk = _batch_joint_1x2(lam_h_b, lam_a_b, 1.0, "frank", 0.0, 0.0, "none", 0.0, False,
                                 ph_c, pa_c, m3.cfg.score_cap, max_elems=1000)
    if not np.allclose(p_full, p_chunk, rtol=1e-9, atol=1e-9):
        issues.append("chunked batch joint != unchunked batch joint")
    # Vectorised market pooling equivalence
    eng_test = FAPMEngine(EngineConfig(enable_challenger=False, enable_stacker=False,
                                        enable_evidence_features=False, enable_evidence_residual=False))
    eng_test.market.alpha = 0.15
    S = 20; n_m = 30
    p_draws = np.stack([clip_matrix(np.random.dirichlet([2, 2, 2], size=n_m)) for _ in range(S)], axis=0)
    market_probs = []
    for i in range(n_m):
        if i % 3 == 0: market_probs.append(None)
        else: market_probs.append(clip_probs(np.random.dirichlet([2, 2, 2])))
    # Reference: python loop
    ref = p_draws.copy()
    for i in range(n_m):
        pm = market_probs[i]
        if pm is None or eng_test.market.alpha <= 0: continue
        a = float(eng_test.market.alpha)
        for s in range(S):
            q = np.exp((1 - a) * np.log(np.clip(pm, EPS, 1.0)) + a * np.log(np.clip(ref[s, i], EPS, 1.0)))
            ref[s, i] = q / max(q.sum(), EPS)
    vec = eng_test._apply_market_pool_batch(p_draws, market_probs, 0.15)
    if not np.allclose(ref, vec, rtol=1e-9, atol=1e-9):
        issues.append(f"vectorised market pooling != python loop (max abs diff {np.max(np.abs(ref - vec)):.2e})")
    # stable_match_key == _vectorized_match_keys
    k1 = stable_match_key(pd.Timestamp("2024-01-01 20:00:00"), "E0", "A", "B")
    k2 = _vectorized_match_keys([pd.Timestamp("2024-01-01 20:00:00")], ["E0"], ["A"], ["B"])[0]
    if k1 != k2: issues.append("stable_match_key != _vectorized_match_keys")
    # sha256_json strictness
    try:
        sha256_json({"x": np.float64(1.5)})
        issues.append("sha256_json accepted np.float64 without raising")
    except TypeError: pass
    except Exception: pass  # numpy scalar may or may not raise; tolerate
    # TrueCumulativeConformal interval guard
    tcc = TrueCumulativeConformal()
    try:
        tcc.interval(np.array([0.0]), np.zeros((1, 10)), np.array(["L"]))
        issues.append("TrueCumulativeConformal.interval did not raise before fit")
    except RuntimeError: pass
    return issues

def _test_pipeline_consistency(eng, cal_feat, train_feat):
    issues = []
    pre_df, _ = eng.predict(cal_feat.head(30), light=True)
    try:
        _ = eng.report(cal_feat.head(30))
        _ = eng.assimilate(CausalContract.normalize(cal_feat.head(30), True))
    except Exception as e:
        issues.append(f"assimilation raised: {e}")
        return issues
    post_df, _ = eng.predict(cal_feat.head(30), light=True)
    diff = float(np.max(np.abs(pre_df[["p_home", "p_draw", "p_away"]].to_numpy() -
                                post_df[["p_home", "p_draw", "p_away"]].to_numpy())))
    if diff > 0.20:
        issues.append(f"post-assimilation prediction drift unbounded: {diff:.3f}")
    try:
        odds_dummy = np.full((5, 3), 3.0)
        _ = eng.stake(cal_feat.head(5), odds_dummy)
    except Exception as e:
        issues.append(f"stake raised: {e}")
    try:
        cov = eng.conformal_coverage_report(cal_feat.head(200))
        target = 1.0 - eng.cfg.conformal_alpha
        observed = cov.get("split_1x2_coverage", None)
        if observed is not None and abs(observed - target) > 0.20:
            issues.append(f"conformal coverage far from nominal: {observed:.3f} vs {target:.3f}")
    except Exception as e:
        issues.append(f"coverage report raised: {e}")
    if eng._frozen_market_alpha > 0:
        try:
            p_cal = eng._cal_pool_replay(cal_feat.head(20))
            if p_cal.shape[0] != 20:
                issues.append("cal_pool_replay returned wrong shape")
        except Exception as e:
            issues.append(f"cal_pool_replay raised: {e}")
    if eng.dirichlet is not None and eng.dirichlet.fitted:
        try:
            test_df = cal_feat.head(15)
            S = 20
            p_draws, _, _ = eng._compute_posterior_draws(test_df, S)
            mean_raw = clip_matrix(p_draws.mean(axis=0))
            p_cal_draws = eng._apply_dirichlet_to_draws(p_draws)
            mean_cal = clip_matrix(p_cal_draws.mean(axis=0))
            p_cal_of_mean = eng.dirichlet.transform(mean_raw)
            spread = np.max(np.abs(mean_cal - p_cal_of_mean))
            if spread > 0.15:
                issues.append(f"Dirichlet draw-vs-mean spread large: {spread:.3f}")
        except Exception as e:
            issues.append(f"Dirichlet coherence test raised: {e}")
    return issues

def _golden_regression(path=None):
    raw = make_synthetic(n=600, seed=42)
    n = len(raw); n_cal = int(n * 0.20)
    tr = raw.iloc[:n - n_cal]; cal = raw.iloc[n - n_cal:]
    eng = FAPMEngine(EngineConfig(n_posterior_samples=40, cumulative_windows=(2, 4)))
    eng.fit(tr, cal)
    test_rows = cal.iloc[:5]
    pred_df, _ = eng.predict(test_rows, light=True)
    pinned = []
    for _, r in pred_df.iterrows():
        pinned.append({"home": str(r["home"]), "away": str(r["away"]),
                        "p_home": round(float(r["p_home"]), 4),
                        "p_draw": round(float(r["p_draw"]), 4),
                        "p_away": round(float(r["p_away"]), 4)})
    fingerprint = {
        "pinned_rows": pinned,
        "phi_h": float(eng.model.phi_h), "phi_a": float(eng.model.phi_a),
        "copula_theta_abs": float(abs(eng.copula_theta)),
        "T_score": float(eng.gate.state.T_score),
        "n_teams": int(eng.model.n_t), "n_leagues": int(eng.model.n_l),
        "ensemble_fitted": bool(eng.ensemble.fitted),
        "dirichlet_fitted": bool(eng.dirichlet and eng.dirichlet.fitted),
        "frozen_market_alpha": float(eng._frozen_market_alpha),
    }
    issues = []
    if int(fingerprint["n_teams"]) != 32: issues.append(f"n_teams: {fingerprint['n_teams']}")
    if path is not None and Path(path).exists():
        try:
            golden = json.loads(Path(path).read_text())
            for k, v in golden.items():
                if k == "pinned_rows":
                    for i, (row, grow) in enumerate(zip(pinned, v)):
                        if row["home"] != grow["home"] or row["away"] != grow["away"]:
                            issues.append(f"golden row {i} team mismatch"); continue
                        for pk in ("p_home", "p_draw", "p_away"):
                            if abs(row[pk] - grow[pk]) > 2e-3:
                                issues.append(f"golden row {i} {pk}: {row[pk]} vs {grow[pk]}")
                elif k in fingerprint and isinstance(v, (int, float)) and isinstance(fingerprint[k], (int, float)):
                    if abs(fingerprint[k] - v) > max(5e-3, abs(v) * 5e-3):
                        issues.append(f"golden mismatch {k}: {fingerprint[k]} vs {v}")
        except Exception as e: issues.append(f"golden load failed: {e}")
    elif path is not None:
        Path(path).write_text(json.dumps(fingerprint, indent=2))
    return {"status": "PASS" if not issues else "FAIL", "fingerprint": fingerprint, "issues": issues}

def self_test(golden_path=None):
    raw = make_synthetic(900)
    n = len(raw); n_test = 150; n_cal = 180
    tr = raw.iloc[:n - n_test - n_cal]; cal = raw.iloc[n - n_test - n_cal:n - n_test]; te = raw.iloc[n - n_test:]
    eng = FAPMEngine(EngineConfig(n_posterior_samples=40, cumulative_windows=(2, 4)))
    fit_res = eng.fit(tr, cal)
    assert eng.fitted
    issues = _test_invariants()
    try:
        train_feat = eng.feature_builder.build(CausalContract.normalize(tr, True), update_state=False)
        cal_feat = eng.feature_builder.build(CausalContract.normalize(cal, True), update_state=False)
        issues += _test_pipeline_consistency(eng, cal_feat, train_feat)
    except Exception as e:
        issues.append(f"pipeline consistency raised: {e}")
    pred_df, _ = eng.predict(te, light=True)
    assert np.allclose(pred_df[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-6)
    assert (pred_df["p_home_hi"] >= pred_df["p_home_lo"]).all()
    cum = eng.cumulative_event_probability(te, n_matches=4, target_ga=6,
                                             team_name=te.iloc[0].home, n_draws=200)
    assert cum.get("n_draws_used") == 200
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name
    ok = False
    try:
        eng.save(tmp, compress=True)
        e2 = FAPMEngine.load(tmp)
        assert e2.fitted
        assert abs(e2.model.phi_h - eng.model.phi_h) < 1e-9
        assert abs(e2.copula_theta - eng.copula_theta) < 1e-9
        assert abs(e2._frozen_market_alpha - eng._frozen_market_alpha) < 1e-9
        assert set(e2.drift_per_league.keys()) == set(eng.drift_per_league.keys())
        assert abs(e2.market.improvement - eng.market.improvement) < 1e-12
        assert abs(e2.hier._phi_h - eng.hier._phi_h) < 1e-9
        assert e2.ensemble.evidence_edges is not None if eng.ensemble.evidence_edges is not None else True
        ok = True
    finally:
        try: os.remove(tmp)
        except Exception: pass
    gold = _golden_regression(golden_path)
    return {"status": "PASS" if not issues else "INVARIANT_ISSUES", "version": VERSION,
            "n_params": int(eng.model.n_params), "n_teams": int(eng.model.n_t),
            "n_leagues": int(eng.model.n_l),
            "phi_h": float(eng.model.phi_h), "phi_a": float(eng.model.phi_a),
            "hier_phi_h_used": float(eng.hier._phi_h), "hier_phi_a_used": float(eng.hier._phi_a),
            "copula_theta": float(eng.copula_theta), "joint_type": eng.joint_type,
            "T_score": float(eng.gate.state.T_score), "T_ou": float(eng.gate.state.T_ou),
            "T_btts": float(eng.gate.state.T_btts),
            "meta_alpha": float(eng.gate.state.meta_alpha),
            "ensemble_fitted": bool(eng.ensemble.fitted),
            "ensemble_evidence_edges": None if eng.ensemble.evidence_edges is None else list(eng.ensemble.evidence_edges),
            "dirichlet_fitted": bool(eng.dirichlet and eng.dirichlet.fitted),
            "hier_fitted": bool(eng.hier.fitted),
            "frozen_market_alpha": float(eng._frozen_market_alpha),
            "frozen_market_status": eng.market.status,
            "market_bootstrap": eng.market.last_bootstrap,
            "evidence_residual_active": bool(eng.evidence_residual and eng.evidence_residual.fitted),
            "evidence_residual_gain": eng._evidence_gain,
            "n_per_league_drift": len(eng.drift_per_league),
            "persistence_roundtrip": bool(ok),
            "cumulative_p": cum.get("p_target"),
            "invariant_issues": issues, "golden": gold}

# ============================================================================
# 17. CLI
# ============================================================================

def _resolve_files(data):
    p = Path(data)
    if p.is_dir(): return [str(x) for x in p.glob("*.csv")]
    return [str(p)]

def cmd_fit(args):
    files = _resolve_files(args.data)
    frames = [pd.read_csv(fp, low_memory=False) for fp in files if fp.endswith(".csv")]
    raw = pd.concat(frames, ignore_index=True)
    raw = CausalContract.normalize(raw, True).sort_values("date").reset_index(drop=True)
    n = len(raw); n_cal = int(n * args.cal_frac)
    tr = raw.iloc[:n - n_cal]; cal = raw.iloc[n - n_cal:]
    evidence = None
    if args.evidence:
        ev_frames = []
        for p in args.evidence:
            path = Path(p)
            ev_frames.append(Top15EvidenceAdapter.load_jsonl([path]) if path.suffix.lower() == ".jsonl"
                              else Top15EvidenceAdapter.load_csv([path]))
        evidence = pd.concat(ev_frames, ignore_index=True) if ev_frames else pd.DataFrame()
    eng = FAPMEngine(EngineConfig(n_posterior_samples=args.samples,
                                    enable_evidence_features=evidence is not None,
                                    enable_evidence_residual=evidence is not None))
    res = eng.fit(tr, cal, evidence=evidence)
    print(json.dumps(res, indent=2, default=str))
    eng.save(args.model_out); return 0

def cmd_predict(args):
    eng = FAPMEngine.load(args.model)
    files = _resolve_files(args.fixture)
    frames = [pd.read_csv(fp, low_memory=False) for fp in files if fp.endswith(".csv")]
    raw = pd.concat(frames, ignore_index=True)
    df = eng.report(raw)
    cols = ["date", "home", "away", "league", "p_home", "p_draw", "p_away", "over25", "btts",
            "lambda_home", "lambda_away", "market_status", "pool_mode", "argmax",
            "cold_start", "conformal_in_set", "selective_publish",
            "copula_family", "joint_type", "T_score", "T_ou", "T_btts", "meta_alpha",
            "dirichlet_active", "ensemble_active", "evidence_residual_active", "hier_active"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    if args.report:
        df.drop(columns=[c for c in ("score_surface", "top_scores") if c in df.columns]) \
          .to_json(args.report, orient="records", date_format="iso", indent=2)
    return 0

def cmd_self_test(args):
    print(json.dumps(self_test(golden_path=args.golden), indent=2, default=str)); return 0

def cmd_version(args):
    print(json.dumps({"model": VERSION, "schema": SCHEMA_VERSION}, indent=2)); return 0

def main(argv=None):
    argv = argv or os.sys.argv[1:]
    ap = argparse.ArgumentParser(description=f"FAPM-AθΩ · XwZi Fusion {VERSION}")
    sub = ap.add_subparsers(dest="cmd")
    p_fit = sub.add_parser("fit"); p_fit.add_argument("data")
    p_fit.add_argument("--model-out", default="fapm_v67.pkl")
    p_fit.add_argument("--cal-frac", type=float, default=0.20)
    p_fit.add_argument("--samples", type=int, default=300)
    p_fit.add_argument("--evidence", nargs="+", default=None)
    p_pred = sub.add_parser("predict"); p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--fixture", required=True); p_pred.add_argument("--report", default=None)
    p_st = sub.add_parser("self-test"); p_st.add_argument("--golden", default=None)
    sub.add_parser("version")
    args = ap.parse_args(argv)
    if args.cmd is None or args.cmd == "self-test": return cmd_self_test(args)
    if args.cmd == "version": return cmd_version(args)
    if args.cmd == "fit": return cmd_fit(args)
    if args.cmd == "predict": return cmd_predict(args)
    ap.print_help(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
