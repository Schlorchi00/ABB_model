"""
Anisotropic Bergström–Boyce (ABB) Model + Mullins Damage — Full Calibration
=============================================================================
Test inputs:
  - Tensile tests:    0, 30, 45, 60, 90 degrees × 3 rates (6, 60, 600 mm/min)
  - Bulk tests:       0 and 90 degrees (hydrostatic)
  - Relaxation test:  5 strain levels (5,10,25,50,100%), 2mm/min ramp + 1hr hold
  - Cyclic test:      5 cycles × 4 strain levels (5,10,25,50%), 10mm/min

Key addition vs v1:
  - Multi-rate tensile data now used to calibrate BB flow rule: A, C, m_flow
  - Anisotropy fit restricted to slowest rate (6 mm/min) as quasi-static reference
  - Two viscoelastic networks identified: short-time (from rate tensile) +
    long-time (from relaxation holds)
  - Global optimiser now includes flow parameters in joint refinement

Bergström–Boyce flow rule (network B):
  dλ_B/dt ≈ A · (λ_chain_B − 1 + ε)^C · |τ_B / μ_ref|^m_flow · sign(τ_B)
  where τ_B is the driving stress on the viscous network.

Units: MPa, mm, seconds throughout.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json
import warnings
from math import erf as math_erf
warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONSTANTS & HELPERS
# =============================================================================

# Specimen geometry — adjust to match your test setup
GAUGE_LENGTH_MM = 25.0   # mm, used to convert crosshead speed → strain rate

def crosshead_to_strain_rate(speed_mm_per_min, gauge_mm=GAUGE_LENGTH_MM):
    """Convert crosshead speed (mm/min) to nominal strain rate (1/s)."""
    return (speed_mm_per_min / 60.0) / gauge_mm

RATES_MM_MIN  = [6.0, 60.0, 600.0]
STRAIN_RATES  = {v: crosshead_to_strain_rate(v) for v in RATES_MM_MIN}

# Approx characteristic strain rate of relaxation ramp (2mm/min)
RELAX_RATE = crosshead_to_strain_rate(2.0)
CYCLIC_RATE = crosshead_to_strain_rate(10.0)


# =============================================================================
# 2. CONSTITUTIVE PRIMITIVES
# =============================================================================

# TODO: Use different Langevin approximations like Padme or Bergstrom
def langevin_inv(beta):
    """Approximation to inverse Langevin: L^{-1}(x) ≈ x(3-x²)/(1-x²)."""
    beta = np.clip(beta, 1e-9, 0.9999)
    return beta * (3.0 - beta**2) / (1.0 - beta**2)


def ab_stress_uniaxial(lam, mu, N):
    """
    Arruda–Boyce nominal stress (uniaxial, incompressible).
    lam: stretch array, mu: shear modulus (MPa), N: chain length parameter.
    """
    lam = np.asarray(lam, dtype=float)
    lam_chain = np.sqrt((lam**2 + 2.0 / lam) / 3.0)
    beta = np.clip(lam_chain / np.sqrt(N), 1e-9, 0.9999)
    inv_L = langevin_inv(beta)
    return mu * np.sqrt(N) / (3.0 * lam_chain) * inv_L * (lam**2 - 1.0 / lam)


def ab_energy_uniaxial(lam, mu, N):
    """Arruda–Boyce strain energy density (uniaxial, incompressible)."""
    lam = np.asarray(lam, dtype=float)
    I1 = lam**2 + 2.0 / lam
    return mu * (
        0.5 * (I1 - 3)
        + (1.0 / (20.0 * N)) * (I1**2 - 9)
        + (11.0 / (1050.0 * N**2)) * (I1**3 - 27)
        + (19.0 / (7050.0 * N**3)) * (I1**4 - 81)
    )


def anisotropy_kappa(angle_deg, delta, phi0_deg):
    """
    Scalar directional stiffness factor.
    delta: anisotropy amplitude, phi0_deg: preferred direction (degrees).
    For FEA: build H = (1−delta)·I + delta·(a⊗a), a = unit vector at phi0_deg.
    """
    theta = np.radians(angle_deg - phi0_deg)
    return 1.0 + delta * np.cos(2.0 * theta)**2


def mullins_eta(W, W_max, r, m_M, beta_m):
    """Ogden–Roxburgh Mullins damage: eta in [0,1]."""
    arg = np.clip((W_max - W) / (m_M + beta_m * W_max + 1e-12), -8.0, 8.0)
    return 1.0 - (1.0 / r) * np.vectorize(math_erf)(arg)


# =============================================================================
# 3. BERGSTRÖM–BOYCE VISCOUS FLOW INTEGRATION
# =============================================================================

def simulate_bb_uniaxial(lam_history, t_history, mu0, N0,
                          mu_ve, N_ve, A_flow, C_flow, m_flow,
                          mu_ve2=0.0, N_ve2=4.0, A_flow2=0.0, C_flow2=0.0, m_flow2=1.0):
    """
    Simulate the Bergström–Boyce model under a prescribed uniaxial stretch history.

    Network A (equilibrium): Arruda–Boyce, mu0, N0
    Network B1 (viscous, short-time): mu_ve, N_ve, A_flow, C_flow, m_flow
    Network B2 (viscous, long-time, optional): mu_ve2, N_ve2, A_flow2, C_flow2, m_flow2

    The viscous flow rule (Bergström–Boyce 1998):
      d(lam_ve)/dt = A · (lam_chain_B − 1 + 0.001)^C · |sigma_B_dev / mu_ve|^m · sign(sigma_B_dev)
    where lam_chain_B is the chain stretch in the viscous network.

    lam_history: prescribed total stretch (array)
    t_history:   corresponding time points (array)
    Returns: total nominal stress P_total (array)
    """
    from scipy.interpolate import interp1d
    lam_interp = interp1d(t_history, lam_history, kind='linear', fill_value='extrapolate')

    EPS_FLOW = 0.001   # offset to avoid singularity at lam_ve=1

    def ode_rhs(t, state):
        lam_ve1, lam_ve2_ = state
        lam = float(lam_interp(t))
        lam_ve1 = max(lam_ve1, 1e-6)
        lam_ve2_ = max(lam_ve2_, 1e-6)

        # Elastic stretch in each viscous network
        lam_e1 = lam / lam_ve1
        lam_e2 = lam / lam_ve2_

        # Driving stress on each network (nominal stress in elastic sub-chain)
        sigma1 = ab_stress_uniaxial(np.array([lam_e1]), mu_ve, N_ve)[0]
        sigma2 = ab_stress_uniaxial(np.array([lam_e2]), mu_ve2, N_ve2)[0] if mu_ve2 > 0 else 0.0

        # Chain stretch in viscous network
        lam_chain_ve1 = np.sqrt((lam_ve1**2 + 2.0 / lam_ve1) / 3.0)
        lam_chain_ve2 = np.sqrt((lam_ve2_**2 + 2.0 / lam_ve2_) / 3.0)

        # Flow rate
        flow_amp1 = A_flow * (max(lam_chain_ve1 - 1.0, 0.0) + EPS_FLOW)**C_flow
        dlam_ve1_dt = flow_amp1 * abs(sigma1 / (mu_ve + 1e-12))**m_flow * np.sign(sigma1)

        dlam_ve2_dt = 0.0
        if mu_ve2 > 0 and A_flow2 > 0:
            flow_amp2 = A_flow2 * (max(lam_chain_ve2 - 1.0, 0.0) + EPS_FLOW)**C_flow2
            dlam_ve2_dt = flow_amp2 * abs(sigma2 / (mu_ve2 + 1e-12))**m_flow2 * np.sign(sigma2)

        return [dlam_ve1_dt, dlam_ve2_dt]

    # Integrate
    sol = solve_ivp(
        ode_rhs,
        t_span=(t_history[0], t_history[-1]),
        y0=[1.0, 1.0],
        t_eval=t_history,
        method='RK45',
        rtol=1e-4, atol=1e-6,
        max_step=(t_history[-1] - t_history[0]) / 200.0
    )

    lam_ve1_sol = np.clip(sol.y[0], 1e-6, None)
    lam_ve2_sol = np.clip(sol.y[1], 1e-6, None) if mu_ve2 > 0 else np.ones_like(lam_ve1_sol)

    lam_arr = lam_interp(sol.t)
    lam_e1 = lam_arr / lam_ve1_sol
    lam_e2 = lam_arr / lam_ve2_sol

    P_A  = ab_stress_uniaxial(lam_arr, mu0, N0)
    P_B1 = ab_stress_uniaxial(lam_e1, mu_ve, N_ve)
    P_B2 = ab_stress_uniaxial(lam_e2, mu_ve2, N_ve2) if mu_ve2 > 0 else 0.0

    return sol.t, P_A + P_B1 + P_B2


def make_ramp_history(lam_max, strain_rate, n_pts=80):
    """Build (lam, t) arrays for a monotonic ramp at constant strain rate."""
    eps_max = lam_max - 1.0
    t_end   = eps_max / strain_rate
    t = np.linspace(0.0, t_end, n_pts)
    lam = 1.0 + strain_rate * t
    return lam, t


def make_relaxation_history(lam_hold, ramp_rate, hold_time_s, n_ramp=40, n_hold=150):
    """Build (lam, t) arrays for ramp + hold relaxation."""
    eps_hold = lam_hold - 1.0
    t_ramp   = eps_hold / ramp_rate
    t_ramp_arr = np.linspace(0.0, t_ramp, n_ramp)
    lam_ramp   = 1.0 + ramp_rate * t_ramp_arr
    t_hold_arr = np.linspace(t_ramp, t_ramp + hold_time_s, n_hold)
    lam_hold_arr = np.full(n_hold, lam_hold)
    t   = np.concatenate([t_ramp_arr, t_hold_arr[1:]])
    lam = np.concatenate([lam_ramp, lam_hold_arr[1:]])
    return lam, t


def make_cyclic_history(lam_pk, strain_rate, n_cycles=5, n_pts_half=60):
    """Build (lam, t) arrays for triangular cyclic loading."""
    eps_pk = lam_pk - 1.0
    t_half = eps_pk / strain_rate
    lam_list, t_list = [], []
    t_cur = 0.0
    for _ in range(n_cycles):
        t_up = np.linspace(t_cur, t_cur + t_half, n_pts_half)
        lam_up = 1.0 + strain_rate * (t_up - t_cur)
        lam_list.append(lam_up); t_list.append(t_up)
        t_cur += t_half
        t_dn = np.linspace(t_cur, t_cur + t_half, n_pts_half)
        lam_dn = lam_pk - strain_rate * (t_dn - t_cur)
        lam_list.append(lam_dn); t_list.append(t_dn)
        t_cur += t_half
    return np.concatenate(lam_list), np.concatenate(t_list)


# =============================================================================
# 4. DATA LOADING  ← REPLACE THESE WITH YOUR REAL DATA
# =============================================================================

def load_data():
    """
    Populate with your real experimental data.

    tensile_rate[rate_mm_min][angle_deg] = {"lam": array, "P": array (MPa)}
    bulk[angle_deg]                       = {"J": array,   "p": array (MPa)}
    relaxation                            = list of dicts (see below)
    cyclic                                = list of dicts (see below)
    """
    np.random.seed(0)

    # True parameters for synthetic data generation
    TRUE = dict(mu0=0.5, N0=4.0, mu_ve=0.3, N_ve=3.5, A=0.05, C=1.5, m_flow=2.5,
                mu_ve2=0.1, N_ve2=5.0, A2=0.002, C2=1.0, m_flow2=2.0,
                delta=0.12, phi0=5.0, K=500.0, r=1.4, m_M=0.05, beta_m=0.01)

    # --- Multi-rate tensile ---
    tensile_rate = {}
    for speed in RATES_MM_MIN:
        tensile_rate[speed] = {}
        sr = STRAIN_RATES[speed]
        for angle in [0, 30, 45, 60, 90]:
            lam_max = 2.2
            lam_arr, t_arr = make_ramp_history(lam_max, sr, n_pts=60)
            _, P_arr = simulate_bb_uniaxial(
                lam_arr, t_arr,
                TRUE["mu0"], TRUE["N0"],
                TRUE["mu_ve"], TRUE["N_ve"], TRUE["A"], TRUE["C"], TRUE["m_flow"],
                TRUE["mu_ve2"], TRUE["N_ve2"], TRUE["A2"], TRUE["C2"], TRUE["m_flow2"]
            )
            kap = anisotropy_kappa(angle, TRUE["delta"], TRUE["phi0"])
            noise = np.random.normal(0, 0.008, P_arr.shape)
            tensile_rate[speed][angle] = {"lam": lam_arr, "P": kap * P_arr + noise}

    # --- Bulk tests ---
    bulk = {}
    for angle in [0, 90]:
        J = np.linspace(0.97, 1.0, 30)
        bulk[angle] = {"J": J, "p": TRUE["K"] * (J - 1.0) + np.random.normal(0, 0.5, 30)}

    # --- Relaxation tests ---
    relaxation = []
    for eps_pct in [5, 10, 25, 50, 100]:
        lam_hold = 1.0 + eps_pct / 100.0
        lam_arr, t_arr = make_relaxation_history(lam_hold, RELAX_RATE, 3600.0)
        _, P_arr = simulate_bb_uniaxial(
            lam_arr, t_arr,
            TRUE["mu0"], TRUE["N0"],
            TRUE["mu_ve"], TRUE["N_ve"], TRUE["A"], TRUE["C"], TRUE["m_flow"],
            TRUE["mu_ve2"], TRUE["N_ve2"], TRUE["A2"], TRUE["C2"], TRUE["m_flow2"]
        )
        noise = np.random.normal(0, 0.005, P_arr.shape)
        relaxation.append({"strain_pct": eps_pct, "lam_hold": lam_hold,
                           "t": t_arr, "P": P_arr + noise})

    # --- Cyclic tests ---
    cyclic = []
    for eps_pct in [5, 10, 25, 50]:
        lam_pk = 1.0 + eps_pct / 100.0
        lam_arr, t_arr = make_cyclic_history(lam_pk, CYCLIC_RATE, n_cycles=5)
        _, P_no_dmg = simulate_bb_uniaxial(
            lam_arr, t_arr,
            TRUE["mu0"], TRUE["N0"],
            TRUE["mu_ve"], TRUE["N_ve"], TRUE["A"], TRUE["C"], TRUE["m_flow"]
        )
        # Apply Mullins damage
        P_dmg = np.zeros_like(P_no_dmg)
        W_max = 0.0
        lam_prev = lam_arr[0]
        for i, (lam, P_raw) in enumerate(zip(lam_arr, P_no_dmg)):
            W = ab_energy_uniaxial(np.array([lam]), TRUE["mu0"], TRUE["N0"])[0]
            if lam >= lam_prev:
                W_max = max(W_max, W)
            eta = mullins_eta(W, W_max, TRUE["r"], TRUE["m_M"], TRUE["beta_m"])
            P_dmg[i] = eta * P_raw
            lam_prev = lam
        noise = np.random.normal(0, 0.005, P_dmg.shape)
        cyclic.append({"strain_pct": eps_pct, "lam": lam_arr, "t": t_arr, "P": P_dmg + noise})

    return tensile_rate, bulk, relaxation, cyclic


# =============================================================================
# 5. CALIBRATION STAGES
# =============================================================================

def calibrate_bulk(bulk_data):
    print("\n[3a] Bulk modulus K ...")
    J_all = np.concatenate([d["J"] for d in bulk_data.values()])
    p_all = np.concatenate([d["p"] for d in bulk_data.values()])
    popt, _ = curve_fit(lambda J, K: K * (J - 1.0), J_all, p_all, p0=[300.0])
    K = popt[0]
    print(f"      K = {K:.1f} MPa")
    return K


def calibrate_equilibrium(relaxation_data, tensile_slow):
    """
    mu0, N0 from long-time (equilibrium) plateau of relaxation + slow tensile (6mm/min, 0°).
    The slow tensile at 6mm/min is quasi-static enough to approximate equilibrium
    at small-to-moderate strains; we weight it less to avoid viscous contamination.
    """
    print("\n[3b] Equilibrium network (mu0, N0) ...")
    lam_eq = np.array([d["lam_hold"] for d in relaxation_data])
    P_eq   = np.array([np.mean(d["P"][-30:]) for d in relaxation_data])

    # Also use slow tensile (0°) up to moderate stretch
    lam_t = tensile_slow["lam"]
    P_t   = tensile_slow["P"]
    mask  = lam_t < 1.5   # avoid large-stretch rate effects
    lam_all = np.concatenate([lam_eq, lam_t[mask]])
    P_all   = np.concatenate([P_eq,   P_t[mask]])

    popt, _ = curve_fit(ab_stress_uniaxial, lam_all, P_all,
                        p0=[0.4, 4.0], bounds=([0.01, 1.1], [5.0, 20.0]), maxfev=5000)
    mu0, N0 = popt
    print(f"      mu0 = {mu0:.4f} MPa,  N0 = {N0:.3f}")
    return mu0, N0


def calibrate_anisotropy(tensile_slow_all_angles, mu0, N0):
    """
    Fit delta, phi0 from the 6mm/min (quasi-static) tensile data across all 5 directions.
    """
    print("\n[3c] Anisotropy (delta, phi0) using slowest rate (6mm/min) ...")
    angles = sorted(tensile_slow_all_angles.keys())
    sig_meas, sig_iso = [], []
    for a in angles:
        d   = tensile_slow_all_angles[a]
        lam = d["lam"]; P = d["P"]
        mask = (lam >= 1.15) & (lam <= 1.6)
        sig_meas.append(np.mean(P[mask]) if mask.any() else P[len(P)//2])
        sig_iso.append(np.mean(ab_stress_uniaxial(lam[mask], mu0, N0)) if mask.any()
                       else ab_stress_uniaxial(np.array([lam[len(lam)//2]]), mu0, N0)[0])
    sig_meas = np.array(sig_meas)
    sig_iso  = np.array(sig_iso)
    ratios   = sig_meas / (sig_iso + 1e-12)

    def kappa_model(angles_arr, delta, phi0):
        return np.array([anisotropy_kappa(a, delta, phi0) for a in angles_arr])

    try:
        popt, _ = curve_fit(kappa_model, np.array(angles, dtype=float), ratios,
                            p0=[0.1, 0.0], bounds=([-0.5, -45], [1.0, 45]))
        delta, phi0 = popt
    except Exception:
        delta, phi0 = 0.0, 0.0
        print("      Warning: anisotropy did not converge, using isotropic.")
    print(f"      delta = {delta:.4f},  phi0 = {phi0:.2f} deg")
    return delta, phi0


def calibrate_flow_params(tensile_rate_data, mu0, N0, mu_ve_guess=0.3, N_ve_guess=3.5):
    """
    Calibrate BB flow rule: mu_ve, N_ve, A_flow, C_flow, m_flow from multi-rate tensile.

    Strategy: compare peak stress at each rate for 0° direction.
    The rate-stiffening encodes A and m_flow; N_ve shapes the non-linearity.
    We use the 0° direction to avoid anisotropy confound.
    """
    print("\n[3d] Viscoelastic network + flow rule (mu_ve, N_ve, A, C, m) ...")
    print("     (using 0° direction across all 3 rates — slowest to fastest)")

    # Build reference curves: lam + measured P for each rate at 0°
    datasets = []
    for speed in sorted(RATES_MM_MIN):
        d  = tensile_rate_data[speed][0]   # 0° direction
        sr = STRAIN_RATES[speed]
        lam_arr, t_arr = make_ramp_history(d["lam"].max(), sr, n_pts=len(d["lam"]))
        datasets.append({"lam": d["lam"], "P": d["P"],
                         "lam_sim": lam_arr, "t_sim": t_arr, "sr": sr})

    def objective(x):
        mu_ve, log_Nve, log_A, C, m_f = x
        N_ve  = np.exp(log_Nve)
        A     = np.exp(log_A)
        if mu_ve <= 0 or N_ve <= 1.0 or A <= 0 or C <= 0 or m_f <= 0:
            return 1e9
        total = 0.0
        for ds in datasets:
            try:
                _, P_pred = simulate_bb_uniaxial(
                    ds["lam_sim"], ds["t_sim"],
                    mu0, N0, mu_ve, N_ve, A, C, m_f
                )
                # Interpolate prediction to measured stretch points
                P_interp = np.interp(ds["lam"], ds["lam_sim"], P_pred)
                total += np.mean((P_interp - ds["P"])**2)
            except Exception:
                return 1e9
        return total

    result = differential_evolution(
        objective,
        bounds=[
            (0.01, 3.0),                   # mu_ve
            (np.log(1.1), np.log(12.0)),   # log(N_ve)
            (np.log(1e-4), np.log(10.0)),  # log(A)
            (0.1, 5.0),                    # C
            (0.5, 6.0),                    # m_flow
        ],
        seed=42, maxiter=400, tol=1e-7, workers=1, polish=True,
        x0=[mu_ve_guess, np.log(N_ve_guess), np.log(0.05), 1.5, 2.5]
    )
    mu_ve, log_Nve, log_A, C_flow, m_flow = result.x
    N_ve   = np.exp(log_Nve)
    A_flow = np.exp(log_A)
    print(f"      mu_ve  = {mu_ve:.4f} MPa")
    print(f"      N_ve   = {N_ve:.3f}")
    print(f"      A_flow = {A_flow:.5f}")
    print(f"      C_flow = {C_flow:.3f}")
    print(f"      m_flow = {m_flow:.3f}")
    return mu_ve, N_ve, A_flow, C_flow, m_flow


def calibrate_second_network(relaxation_data, mu0, N0, mu_ve, N_ve, A, C, m_f):
    """
    Calibrate a second (long-time) viscoelastic network from the relaxation curves.
    The first network handles the fast rate-stiffening; the second captures the slow
    1hr creep/relaxation that can't be seen in short tensile ramps.
    """
    print("\n[3e] Second viscoelastic network (long-time relaxation) ...")

    def objective(x):
        mu2, log_N2, log_A2, C2, m2 = x
        N2 = np.exp(log_N2); A2 = np.exp(log_A2)
        if mu2 <= 0 or N2 <= 1.0:
            return 1e9
        total = 0.0
        for d in relaxation_data:
            lam_arr, t_arr = make_relaxation_history(d["lam_hold"], RELAX_RATE, 3600.0)
            try:
                _, P_pred = simulate_bb_uniaxial(
                    lam_arr, t_arr,
                    mu0, N0, mu_ve, N_ve, A, C, m_f,
                    mu2, N2, A2, C2, m2
                )
                # Focus on hold phase (last 75% of time points)
                n = len(t_arr)
                start = n // 4
                P_hold_pred = np.interp(d["t"][start:], t_arr, P_pred)
                total += np.mean((P_hold_pred - d["P"][start:])**2)
            except Exception:
                return 1e9
        return total

    result = differential_evolution(
        objective,
        bounds=[
            (0.001, 1.0),                  # mu_ve2
            (np.log(1.1), np.log(12.0)),   # log(N_ve2)
            (np.log(1e-6), np.log(1.0)),   # log(A2)
            (0.1, 3.0),                    # C2
            (0.5, 5.0),                    # m_flow2
        ],
        seed=7, maxiter=300, tol=1e-7, workers=1, polish=True,
    )
    mu2, log_N2, log_A2, C2, m2 = result.x
    N2 = np.exp(log_N2); A2 = np.exp(log_A2)
    print(f"      mu_ve2  = {mu2:.4f} MPa")
    print(f"      N_ve2   = {N2:.3f}")
    print(f"      A_flow2 = {A2:.6f}")
    print(f"      C_flow2 = {C2:.3f}")
    print(f"      m_flow2 = {m2:.3f}")
    return mu2, N2, A2, C2, m2


def calibrate_mullins(cyclic_data, mu0, N0):
    """Mullins damage: r, m_M, beta_m from cyclic loading/unloading."""
    print("\n[3f] Mullins damage (r, m_M, beta_m) ...")
    lam_all = np.concatenate([d["lam"] for d in cyclic_data])
    P_all   = np.concatenate([d["P"]   for d in cyclic_data])

    def objective(params):
        r, m_M, bm = params
        if r <= 1.0 or m_M <= 0 or bm < 0:
            return 1e9
        P_pred = np.zeros_like(lam_all)
        W_max = 0.0; lam_prev = lam_all[0]
        for i, lam in enumerate(lam_all):
            W = ab_energy_uniaxial(np.array([lam]), mu0, N0)[0]
            if lam >= lam_prev:
                W_max = max(W_max, W)
            eta = mullins_eta(W, W_max, r, m_M, bm)
            P_pred[i] = eta * ab_stress_uniaxial(np.array([lam]), mu0, N0)[0]
            lam_prev = lam
        return np.mean((P_pred - P_all)**2)

    result = differential_evolution(
        objective, bounds=[(1.01, 5.0), (1e-5, 1.0), (0.0, 0.5)],
        seed=42, maxiter=400, tol=1e-7, workers=1, polish=True
    )
    r, m_M, bm = result.x
    print(f"      r      = {r:.4f}")
    print(f"      m_M    = {m_M:.6f}")
    print(f"      beta_m = {bm:.6f}")
    return r, m_M, bm


# =============================================================================
# 6. GLOBAL JOINT OPTIMISATION
# =============================================================================

def global_optimise(tensile_rate_data, relaxation_data, cyclic_data, p0):
    """
    Step 4: refine all parameters simultaneously.
    Uses a weighted sum over all test types.
    """
    print("\n[4] Global joint optimisation ...")

    mu0    = p0["mu0"];    N0     = p0["N0"]
    mu_ve  = p0["mu_ve"]; N_ve   = p0["N_ve"]
    A      = p0["A"];      C      = p0["C"];      m_f   = p0["m_flow"]
    mu2    = p0["mu_ve2"]; N2     = p0["N_ve2"]
    A2     = p0["A2"];     C2     = p0["C2"];     m2    = p0["m_flow2"]
    delta  = p0["delta"]; phi0   = p0["phi0"]
    r      = p0["r"];      m_M    = p0["m_M"];    bm    = p0["beta_m"]

    # Pre-build time histories (expensive, do once)
    relax_hists = []
    for d in relaxation_data:
        lam_h, t_h = make_relaxation_history(d["lam_hold"], RELAX_RATE, 3600.0)
        relax_hists.append({"lam": lam_h, "t": t_h, "P_meas": d["P"], "t_meas": d["t"]})

    cyc_hists = []
    for d in cyclic_data:
        lam_h, t_h = make_cyclic_history(1.0 + d["strain_pct"]/100.0, CYCLIC_RATE, 5)
        cyc_hists.append({"lam": lam_h, "t": t_h, "P_meas": d["P"]})

    ten_hists = []
    for speed in RATES_MM_MIN:
        sr = STRAIN_RATES[speed]
        for angle in [0, 30, 45, 60, 90]:
            d = tensile_rate_data[speed][angle]
            lam_h, t_h = make_ramp_history(d["lam"].max(), sr, n_pts=len(d["lam"]))
            kap = anisotropy_kappa(angle, delta, phi0)  # kept fixed in joint opt
            ten_hists.append({"lam": lam_h, "t": t_h,
                               "lam_meas": d["lam"], "P_meas": d["P"], "kap": kap})

    def objective(x):
        mu0_, N0_, mu_v_, log_Nv_, log_A_, C_, m_f_, \
            mu2_, log_N2_, log_A2_, C2_, m2_, r_, m_M_, bm_ = x

        N_v_ = np.exp(log_Nv_); A_ = np.exp(log_A_)
        N2_  = np.exp(log_N2_); A2_= np.exp(log_A2_)

        if any(v <= 0 for v in [mu0_, N0_-1.1, mu_v_, N_v_-1.1, A_, C_, m_f_,
                                  mu2_, N2_-1.1, A2_, C2_, m2_, r_-1.01, m_M_, ]):
            return 1e9

        total = 0.0
        w_ten = 1.0; w_rel = 1.5; w_cyc = 2.0

        # Tensile
        for th in ten_hists:
            try:
                _, P_pred = simulate_bb_uniaxial(
                    th["lam"], th["t"],
                    mu0_, N0_, mu_v_, N_v_, A_, C_, m_f_,
                    mu2_, N2_, A2_, C2_, m2_
                )
                P_int = np.interp(th["lam_meas"], th["lam"], P_pred) * th["kap"]
                total += w_ten * np.mean((P_int - th["P_meas"])**2)
            except Exception:
                return 1e9

        # Relaxation
        for rh in relax_hists:
            try:
                _, P_pred = simulate_bb_uniaxial(
                    rh["lam"], rh["t"],
                    mu0_, N0_, mu_v_, N_v_, A_, C_, m_f_,
                    mu2_, N2_, A2_, C2_, m2_
                )
                P_int = np.interp(rh["t_meas"], rh["t"], P_pred)
                total += w_rel * np.mean((P_int - rh["P_meas"])**2)
            except Exception:
                return 1e9

        # Cyclic (Mullins applied on top of AB equilibrium only for speed)
        for ch in cyc_hists:
            lam_arr = ch["lam"]
            W_max = 0.0; lam_prev = lam_arr[0]
            P_pred_cyc = np.zeros_like(lam_arr)
            for i, lam in enumerate(lam_arr):
                W = ab_energy_uniaxial(np.array([lam]), mu0_, N0_)[0]
                if lam >= lam_prev:
                    W_max = max(W_max, W)
                eta = mullins_eta(W, W_max, r_, m_M_, bm_)
                P_pred_cyc[i] = eta * ab_stress_uniaxial(np.array([lam]), mu0_, N0_)[0]
                lam_prev = lam
            total += w_cyc * np.mean((P_pred_cyc - ch["P_meas"])**2)

        return total

    x0 = [mu0, N0, mu_ve, np.log(N_ve), np.log(A), C, m_f,
           mu2, np.log(N2), np.log(A2), C2, m2, r, m_M, bm]

    bounds = [
        (0.01, 5.0), (1.1, 20.0),           # mu0, N0
        (0.01, 3.0), (np.log(1.1), np.log(12.0)), (np.log(1e-5), np.log(10.0)),
        (0.1, 5.0), (0.5, 7.0),             # mu_ve, N_ve, A, C, m_flow
        (0.001, 2.0), (np.log(1.1), np.log(12.0)), (np.log(1e-8), np.log(1.0)),
        (0.1, 3.0), (0.5, 5.0),             # mu_ve2, N_ve2, A2, C2, m_flow2
        (1.01, 5.0), (1e-5, 2.0), (0.0, 0.5),  # r, m_M, beta_m
    ]

    result = differential_evolution(
        objective, bounds, seed=42, maxiter=600,
        init='latinhypercube', tol=1e-8, workers=1, polish=True, x0=x0
    )

    mu0_, N0_, mu_v_, log_Nv_, log_A_, C_, m_f_, \
        mu2_, log_N2_, log_A2_, C2_, m2_, r_, m_M_, bm_ = result.x

    out = dict(
        mu0=mu0_, N0=N0_,
        mu_ve=mu_v_, N_ve=np.exp(log_Nv_), A=np.exp(log_A_), C=C_, m_flow=m_f_,
        mu_ve2=mu2_, N_ve2=np.exp(log_N2_), A2=np.exp(log_A2_), C2=C2_, m_flow2=m2_,
        r=r_, m_M=m_M_, beta_m=bm_,
        delta=p0["delta"], phi0=p0["phi0"], K=p0["K"]
    )
    for k, v in out.items():
        print(f"      {k:<12} = {v:.5f}")
    print(f"      Objective  = {result.fun:.6f}")
    return out


# =============================================================================
# 7. VALIDATION
# =============================================================================

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def compute_metrics(p, tensile_rate_data, relaxation_data, cyclic_data):
    print("\n[5] Validation metrics")
    header = f"  {'Test':<38} {'RMSE (MPa)':<14} {'R²'}"
    print(header); print("  " + "-"*(len(header)-2))

    mu0=p["mu0"]; N0=p["N0"]
    mu_ve=p["mu_ve"]; N_ve=p["N_ve"]; A=p["A"]; C=p["C"]; m_f=p["m_flow"]
    mu2=p["mu_ve2"]; N2=p["N_ve2"]; A2=p["A2"]; C2=p["C2"]; m2=p["m_flow2"]
    delta=p["delta"]; phi0=p["phi0"]

    for speed in RATES_MM_MIN:
        for angle in [0, 30, 45, 60, 90]:
            d  = tensile_rate_data[speed][angle]
            sr = STRAIN_RATES[speed]
            lam_h, t_h = make_ramp_history(d["lam"].max(), sr, n_pts=len(d["lam"]))
            _, P_pred = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
            kap = anisotropy_kappa(angle, delta, phi0)
            P_int = np.interp(d["lam"], lam_h, P_pred) * kap
            rmse = np.sqrt(np.mean((P_int - d["P"])**2))
            print(f"  Tensile {speed:>5.0f}mm/min {angle:>2}°{'':<12} {rmse:.5f}        {r2_score(d['P'], P_int):.4f}")

    for d in relaxation_data:
        lam_h, t_h = make_relaxation_history(d["lam_hold"], RELAX_RATE, 3600.0)
        _, P_pred = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
        P_int = np.interp(d["t"], t_h, P_pred)
        rmse  = np.sqrt(np.mean((P_int - d["P"])**2))
        print(f"  Relaxation {d['strain_pct']:>3}%{'':<23} {rmse:.5f}        {r2_score(d['P'], P_int):.4f}")

    for d in cyclic_data:
        lam_arr = d["lam"]
        P_pred = np.zeros_like(lam_arr); W_max=0.0; lam_prev=lam_arr[0]
        for i, lam in enumerate(lam_arr):
            W = ab_energy_uniaxial(np.array([lam]), mu0, N0)[0]
            if lam >= lam_prev: W_max = max(W_max, W)
            eta = mullins_eta(W, W_max, p["r"], p["m_M"], p["beta_m"])
            P_pred[i] = eta * ab_stress_uniaxial(np.array([lam]), mu0, N0)[0]
            lam_prev = lam
        rmse = np.sqrt(np.mean((P_pred - d["P"])**2))
        print(f"  Cyclic {d['strain_pct']:>3}%{'':<28} {rmse:.5f}        {r2_score(d['P'], P_pred):.4f}")


def plot_validation(p, tensile_rate_data, relaxation_data, cyclic_data):
    mu0=p["mu0"]; N0=p["N0"]
    mu_ve=p["mu_ve"]; N_ve=p["N_ve"]; A=p["A"]; C=p["C"]; m_f=p["m_flow"]
    mu2=p["mu_ve2"]; N2=p["N_ve2"]; A2=p["A2"]; C2=p["C2"]; m2=p["m_flow2"]
    delta=p["delta"]; phi0=p["phi0"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ABB + Mullins — Full Validation", fontsize=13)

    # Row 1: Rate-dependent tensile (0° at all 3 rates)
    ax = axes[0, 0]
    for speed, col in zip(RATES_MM_MIN, ["#1f77b4", "#ff7f0e", "#d62728"]):
        d  = tensile_rate_data[speed][0]
        sr = STRAIN_RATES[speed]
        lam_h, t_h = make_ramp_history(d["lam"].max(), sr, n_pts=100)
        _, P_pred = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
        ax.plot(d["lam"]-1, d["P"], "o", ms=2, color=col, label=f"{speed:.0f}mm/min data")
        ax.plot(lam_h-1, P_pred, "-", lw=1.5, color=col, label=f"{speed:.0f}mm/min fit")
    ax.set_xlabel("Engineering strain"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("0° tensile — rate dependence"); ax.legend(fontsize=7)

    # Row 1: Anisotropy (6mm/min, all directions)
    ax = axes[0, 1]
    cols_ang = plt.cm.tab10(np.linspace(0, 0.5, 5))
    for (angle, d), c in zip(sorted(tensile_rate_data[RATES_MM_MIN[0]].items()), cols_ang):
        sr  = STRAIN_RATES[RATES_MM_MIN[0]]
        lam_h, t_h = make_ramp_history(d["lam"].max(), sr, n_pts=80)
        _, P_pred = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
        kap = anisotropy_kappa(angle, delta, phi0)
        ax.plot(d["lam"]-1, d["P"], "o", ms=2, color=c, label=f"{angle}° data")
        ax.plot(lam_h-1, P_pred*kap, "-", lw=1.5, color=c, label=f"{angle}° fit")
    ax.set_xlabel("Engineering strain"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("Tensile 6mm/min — 5 directions"); ax.legend(fontsize=7, ncol=2)

    # Row 1: Relaxation
    ax = axes[0, 2]
    cols_rel = plt.cm.viridis(np.linspace(0.1, 0.9, len(relaxation_data)))
    for d, c in zip(relaxation_data, cols_rel):
        lam_h, t_h = make_relaxation_history(d["lam_hold"], RELAX_RATE, 3600.0)
        _, P_pred = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
        ax.plot(d["t"], d["P"], "o", ms=1.5, color=c, label=f"{d['strain_pct']}% data")
        ax.plot(t_h, P_pred, "-", lw=1.5, color=c, label=f"{d['strain_pct']}% fit")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("Relaxation (all strains)"); ax.legend(fontsize=7)

    # Row 2: Cyclic
    ax = axes[1, 0]
    cols_cyc = plt.cm.copper(np.linspace(0.1, 0.9, len(cyclic_data)))
    for d, c in zip(cyclic_data, cols_cyc):
        lam_arr = d["lam"]
        P_pred = np.zeros_like(lam_arr); W_max=0.0; lam_prev=lam_arr[0]
        for i, lam in enumerate(lam_arr):
            W = ab_energy_uniaxial(np.array([lam]), mu0, N0)[0]
            if lam >= lam_prev: W_max = max(W_max, W)
            eta = mullins_eta(W, W_max, p["r"], p["m_M"], p["beta_m"])
            P_pred[i] = eta * ab_stress_uniaxial(np.array([lam]), mu0, N0)[0]
            lam_prev = lam
        ax.plot(lam_arr, d["P"], "-", lw=0.8, color=c, alpha=0.5, label=f"{d['strain_pct']}% data")
        ax.plot(lam_arr, P_pred, "--", lw=1.2, color=c, label=f"{d['strain_pct']}% fit")
    ax.set_xlabel("Stretch λ"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("Cyclic + Mullins"); ax.legend(fontsize=7)

    # Row 2: Rate stiffening summary (peak stress vs log strain rate)
    ax = axes[1, 1]
    angles_plot = [0, 45, 90]
    marker_styles = ["o", "s", "^"]
    for angle, mk in zip(angles_plot, marker_styles):
        P_meas_pk, P_pred_pk, srs = [], [], []
        for speed in RATES_MM_MIN:
            d  = tensile_rate_data[speed][angle]
            sr = STRAIN_RATES[speed]
            srs.append(sr)
            P_meas_pk.append(d["P"].max())
            lam_h, t_h = make_ramp_history(d["lam"].max(), sr, n_pts=80)
            _, P_pr = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
            kap = anisotropy_kappa(angle, delta, phi0)
            P_pred_pk.append(P_pr.max() * kap)
        ax.semilogx(srs, P_meas_pk, mk+"--", label=f"{angle}° data")
        ax.semilogx(srs, P_pred_pk, mk+"-",  label=f"{angle}° fit")
    ax.set_xlabel("Strain rate (1/s)"); ax.set_ylabel("Peak stress (MPa)")
    ax.set_title("Rate stiffening summary"); ax.legend(fontsize=7)

    # Row 2: Residuals distribution
    ax = axes[1, 2]
    all_resid = []
    for speed in RATES_MM_MIN:
        for angle in [0, 30, 45, 60, 90]:
            d  = tensile_rate_data[speed][angle]
            sr = STRAIN_RATES[speed]
            lam_h, t_h = make_ramp_history(d["lam"].max(), sr, n_pts=len(d["lam"]))
            _, P_pred = simulate_bb_uniaxial(lam_h, t_h, mu0, N0, mu_ve, N_ve, A, C, m_f, mu2, N2, A2, C2, m2)
            kap = anisotropy_kappa(angle, delta, phi0)
            P_int = np.interp(d["lam"], lam_h, P_pred) * kap
            all_resid.extend((P_int - d["P"]).tolist())
    ax.hist(all_resid, bins=40, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual (MPa)"); ax.set_ylabel("Count")
    ax.set_title("Residual distribution (tensile, all rates+directions)")

    plt.tight_layout()
    plt.savefig("abb_mullins_validation_v2.png", dpi=150)
    print("\nValidation plot saved → abb_mullins_validation_v2.png")
    plt.show()


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  ABB + Mullins — Full Calibration (with multi-rate tensile)")
    print("=" * 65)
    print(f"\n  Strain rates: {', '.join(f'{v:.4f} s⁻¹' for v in STRAIN_RATES.values())}")
    print(f"  Gauge length: {GAUGE_LENGTH_MM} mm")
    print(f"  Adjust GAUGE_LENGTH_MM at top of file to match your specimen.\n")

    tensile_rate, bulk_data, relaxation_data, cyclic_data = load_data()

    K                         = calibrate_bulk(bulk_data)
    mu0, N0                   = calibrate_equilibrium(relaxation_data, tensile_rate[RATES_MM_MIN[0]][0])
    delta, phi0               = calibrate_anisotropy(tensile_rate[RATES_MM_MIN[0]], mu0, N0)
    mu_ve, N_ve, A, C, m_f   = calibrate_flow_params(tensile_rate, mu0, N0)
    mu2, N2, A2, C2, m2       = calibrate_second_network(relaxation_data, mu0, N0, mu_ve, N_ve, A, C, m_f)
    r, m_M, bm                = calibrate_mullins(cyclic_data, mu0, N0)

    p0 = dict(K=K, mu0=mu0, N0=N0,
              mu_ve=mu_ve, N_ve=N_ve, A=A, C=C, m_flow=m_f,
              mu_ve2=mu2, N_ve2=N2, A2=A2, C2=C2, m_flow2=m2,
              delta=delta, phi0=phi0, r=r, m_M=m_M, beta_m=bm)

    params = global_optimise(tensile_rate, relaxation_data, cyclic_data, p0)

    compute_metrics(params, tensile_rate, relaxation_data, cyclic_data)

    out_path = "abb_params_calibrated_v2.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParameters saved → {out_path}")

    plot_validation(params, tensile_rate, relaxation_data, cyclic_data)

    print("\n  FEA export notes:")
    print(f"  H = (1−delta)·I + delta·(a⊗a),  a = unit vector at phi0={params['phi0']:.2f}°")
    print(f"  Network A (eq):  mu={params['mu0']:.4f} MPa, N={params['N0']:.3f}")
    print(f"  Network B1 (fast ve): mu={params['mu_ve']:.4f} MPa, N={params['N_ve']:.3f}, A={params['A']:.5f}, C={params['C']:.3f}, m={params['m_flow']:.3f}")
    print(f"  Network B2 (slow ve): mu={params['mu_ve2']:.4f} MPa, N={params['N_ve2']:.3f}, A={params['A2']:.6f}, C={params['C2']:.3f}, m={params['m_flow2']:.3f}")
    print(f"  Mullins: r={params['r']:.4f}, m_M={params['m_M']:.6f}, beta_m={params['beta_m']:.6f}")
    print(f"  Bulk:   K={params['K']:.1f} MPa")


if __name__ == "__main__":
    main()