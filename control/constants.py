"""Grounded physical parameters for the physical-refactor controller. All concrete; tune later
only against bench numbers. Solver (mfg/hjb.py, mfg/fictitious_play.py) is never imported here
except transitively via equilibrium.py, which calls it unmodified.
"""
import math

# --- airframe / collision ---
R_D = 0.15                      # m, quad radius incl. props (5" class)
D_COLLISION = 2 * R_D           # m, true centre-distance collision threshold = 0.30
R_BARRIER = D_COLLISION + 0.10  # m, avoidance barrier activates below this = 0.40 (0.10 m margin)

# --- packing / density targets ---
D_STAR = 0.60                   # m, target spacing
RHO_STAR = 1.0 / D_STAR ** 2     # agents/m^2, target areal density = 2.778
RHO_NOM = 0.70                  # agents/m^2, nominal operating density

# --- arena scaling (self-similar to the original 4x3 / gate_half=0.4 baseline) ---
_BASELINE_AREA = 4.0 * 3.0       # 12 m^2
_BASELINE_GATE_HALF = 0.4        # m


def arena_for_N(N: int) -> tuple[float, float, float]:
    """Returns (W, H, gate_half) for N agents at RHO_NOM, aspect 4:3, gate scaled as
    gate_half = 0.4*sqrt(A/12) so gate_width/arena_width stays constant across N."""
    A = N / RHO_NOM
    W = math.sqrt(A * 4.0 / 3.0)
    H = math.sqrt(A * 3.0 / 4.0)
    gate_half = _BASELINE_GATE_HALF * math.sqrt(A / _BASELINE_AREA)
    return W, H, gate_half


def source_spread_for_N(N: int) -> float:
    """Initial-density Gaussian spread, scaled so the PEAK initial density stays ~RHO_STAR
    regardless of N (not the SingleGate default's fixed 0.6 m). With density_scale=N (see
    equilibrium.py) and a fixed spread, peak density -- and hence peak W(m)=c*m^2 cost --
    would grow as N^2, which numerically blew up the HJB backward solve (confirmed: NaN at
    N=100 with the default fixed spread, clean with this scaling). Flagged: this parameter
    broke physical/numerical consistency at the originally-chosen fixed value."""
    return math.sqrt(N / (2.0 * math.pi * RHO_STAR))


# --- timing (shared between equilibrium.py and rollout.py) ---
V_NOM = 1.0        # m/s, nominal cruise speed used only to size the simulation horizon --
                   # concrete, used, but not yet calibrated against bench numbers (flagged).
HORIZON_MULT = 3.0


def horizon_for_N(N: int) -> float:
    W, _, _ = arena_for_N(N)
    return HORIZON_MULT * W / V_NOM


# --- sensing noise ---
SIGMA_R = 0.10                  # m, UWB range noise (DWM3001C realistic)
SIGMA_THETA = math.radians(2.0)  # rad, vision/LED bearing noise (2 deg)

# --- TDMA schedule ---
T_SLOT = 0.005                  # s, 5 ms per slot


def t_frame(N: int) -> float:
    """Full TDMA frame period: every agent broadcasts once per frame."""
    return N * T_SLOT


# --- density estimation ---
H_BANDWIDTH = D_STAR             # m, KDE bandwidth (= d*), SEPARATE from sensing/comms range
R_SENSE = 5.0                    # m, sensing/comms range, SEPARATE from h

# --- actuator limit ---
V_MAX = 8.0                      # m/s, FROZEN (PART U). Max COMMANDED speed for a 5" quad in a
                                  # precision-maneuver / formation-flight regime -- a platform
                                  # capability anchor only. Deliberately NOT derived from any
                                  # saturation, effort-ratio, or other simulated result (the prior
                                  # 12.0 value mixed a platform claim with an effort-ratio target
                                  # measured against REAL; that tuning-to-a-result is removed).

# --- avoidance barrier ceiling, FROZEN (PART U). ABSOLUTE constant, NOT a multiple of V_MAX --
# the prior F_MAX=F_MAX_MULT*V_MAX coupling was the confound flagged by external review (raising
# V_MAX for de-saturation silently also raised avoidance strength). Derived from U2's measured
# p95|u_route|=4.8830, p95|u_cong|=17.9013 at C_DENS=1.0, V_MAX=8.0 (old barrier): target 30 failed
# F_BARRIER_MAX >= p95|u_route|+p95|u_cong|+V_MAX = 30.7843 by 0.78, so raised to the smallest
# round value clearing both: (a) F_BARRIER_MAX > realized p95|u_barrier|=15.1010 (never throttles
# normal avoidance) and (b) F_BARRIER_MAX >= p95(|u_route|+|u_cong|) + V_MAX (avoidance can command
# full escape vs the worst-case opposing command) -- both verified at 35.
F_BARRIER_MAX = 35.0

# --- control law weights ---
C_DENS = 1.0                     # FROZEN (PART R). Calibrated against m*-tracking fidelity +
                                  # multi-objective knee (PART M/P), NOT the old force-ratio note
                                  # below -- that note was imported from an unrelated, archived
                                  # control paradigm (paradigmB_costtogo) and does not apply to
                                  # this mean-field density-tracking controller (PART L finding).
                                  # Robustness panel (floor/center/ceiling): {0.3, 1.0, 3.0}.
                                  # Documented stress points beyond the panel: {5, 10, 20} --
                                  # these show escalating actuator saturation (clip-active% rising
                                  # from ~11% at 0.3 to ~70% at 20, real condition) for diminishing
                                  # m*-tracking gains past this band (see diag_partP_log.txt).
                                  # C_DENS is now fixed -- do not reopen without a new finding.
# C_COL (50.0) is SUPERSEDED: the avoidance barrier is now a directly-bounded force (see
# control.py's bounded_barrier_force_all), expressed in velocity units via F_MAX_MULT*V_MAX --
# no separate unbounded-potential gain to tune. Removed from use, not deleted, for history.

# SUPERSEDED calibration target (PART L/M/P): the note below was borrowed from a different,
# archived control paradigm's force audit and does not reflect this controller's own m*-tracking
# behavior -- empirically, 0.2-0.3x gave the WORST m*-tracking of any tested C_DENS (PART M).
# Kept for history, not as a target:
#   congestion force should be ~0.2-0.3 of the goal force (matches the measured 0.27 from the
#   costtogo investigation's D1 force audit), and m* should peak at or below RHO_STAR.
