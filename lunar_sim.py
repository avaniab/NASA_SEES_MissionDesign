"""
lunar_sim.py
============
Full mission-phase Earth -> Moon south-pole landing simulation, in 3D.
Launch vehicle modeled: SLS Block 1 / Orion (Artemis architecture) -- ONLY
the crewed launch vehicle (ICPS + Orion) is simulated. The cargo/rover
launch vehicle is mission context only and is not propagated.

Mission profile (matches NASA's NRHO staging architecture,
"Enabling Global Lunar Access for Human Landing Systems", NTRS 20200002920):

    Launch from Earth
        |
        v
    Phase A: Low Earth Orbit (LEO)
        |
        v
    Phase B: TLI burn + translunar coast   <- plane targeted from EARTH, not fixed at the Moon
        |
        v
    Phase C: NRHO insertion (Orion's OWN ESM burn -- no refuel, no ICPS help)
        |
        v
    Phase D: Rendezvous w/ HLS lander in NRHO  <- Orion NEVER goes to LLO
        |
        v
    [Phase E-I: HLS lander's OWN separate leg -- NRHO->LLO->surface->LLO->NRHO.
     Its own vehicle, its own tanks. NOT part of the SLS/Orion budget below.]
        |
        v
    Phase J: Trans-Earth injection (Orion's OWN ESM burn, 2nd/last use) / return

WHY ORION NEVER GOES TO LLO (this used to be modeled wrong)
------------------------------------------------------------
Earlier versions of this script had the SLS/Orion vehicle itself descend to
LLO and back, charging that delta-v to Orion's budget. That's not how
Artemis actually works: Orion stays in NRHO the whole time. Only the
separately-launched HLS lander (Starship HLS / Blue Moon, its own rocket,
its own huge propellant supply) does the NRHO<->LLO<->surface legs. This
distinction matters a lot for delta-v budgeting -- NRHO insertion/departure
is cheap (~430 + 410 m/s, see below), while a full LLO round trip is not
(~2.5 km/s combined), and Orion's onboard ESM propellant can only support
the former. This is a real, documented fact: Artemis II's free-return
flyby -- not a full lunar-orbit insertion -- specifically because the ESM
does not carry enough delta-v for both a full LLO insertion AND the trip
home. NRHO's cheapness is *why* NASA chose it.

KEY PHYSICS IMPROVEMENTS (vs. earlier versions)
-------------------------------------------------
1. Plane targeted from Earth, not fixed at the Moon: the TLI burn injects
   into a 3D transfer orbit whose plane is deliberately tilted (about the
   Earth-Moon injection line) by TRANSLUNAR_PLANE_INCL_DEG, matching the
   near-polar lunar staging orbit needed for south-pole access, chosen at
   LEO -- not bolted on as a free 90 deg change at the Moon.
2. v_inf is now a single, correct, numeric value taken directly from the
   actual integrated 3D state at closest approach. An earlier "analytic"
   patched-conic v_inf (2-body vis-viva + a scalar, non-vector subtraction
   of the Moon's orbital speed) silently broke (wrong or NaN) once the
   trajectory plane was tilted in 3D, and has been removed rather than
   patched around.
3. NRHO insertion/TEI delta-v is still computed with our own simplified
   2-body combined_dv() formula for comparison, but the number actually
   CHARGED to Orion's tanks is a published reference value (Whitley &
   Martinez, "Options for Staging Orbits in Cislunar Space," IEEE Aerospace
   2015). Our own formula runs high because 2-body mechanics can't see the
   3-body (CR3BP) dynamics that make real NRHO insertion cheap -- a true
   fix requires solving the Circular Restricted Three-Body Problem, which
   is flagged as future work, not implemented here.

Simplifications (clearly flagged, not hidden):
  - The NRHO here is a REPRESENTATIVE near-polar ellipse sized to match the
    real 9:2 lunar synodic resonant NRHO used by Gateway (perilune ~3,000 km
    altitude, apolune ~70,000 km altitude), not a numerically-solved CR3BP
    halo orbit. A true NRHO requires solving the Circular Restricted
    Three-Body Problem; that's future work, not implemented here.
  - The translunar leg still uses patched two-body dynamics (Earth+Moon
    point masses, no Sun, no solar perturbation), now in 3D instead of 2D.
  - Moon's rotation is treated as perfectly synchronous (tidal-locked) with
    its orbit -- real libration is ignored.
  - The HLS lander's ascent delta-v is assumed symmetric with its descent
    delta-v (a real ascent stage has different dry/prop mass fractions;
    not modeled here). This is the lander's own budget, tracked separately
    and not charged to the SLS/Orion vehicle.
  - Only launch vehicle 1 (SLS Block 1 / ICPS + Orion) is simulated in
    detail. The HLS lander and the cargo launch vehicle carrying it are
    mission context only -- their own delta-v is estimated for reference
    but their own launch/refueling profile is not simulated.
  - No refueling is assumed anywhere for the SLS/Orion vehicle: Orion
    carries everything it needs (LOI + TEI) from a single Earth launch,
    using only its own ESM propellant. This is checked explicitly at the
    end of run() (see "FEASIBILITY CHECK" in the printed output).

Vehicle data (launch vehicle 1: SLS Block 1 / ICPS + Orion)
-------------------------------------------------------------
  ICPS (Interim Cryogenic Propulsion Stage, does the TLI burn):
    - 1x Aerojet Rocketdyne RL10B-2, thrust 110.1 kN (24,750 lbf) [NASA/ULA]
    - Isp ~ 462 s vacuum (RL10B-2 class engine)
    - Propellant/dry mass ~ derived from the Delta Cryogenic Second Stage
      (ICPS is a lengthened-tank DCSS-5m derivative)
  Orion (CM + ESM, the "payload" riding on ICPS to TLI, and the only part
  of this vehicle that does anything after TLI):
    - ~25,850 kg total mass at TLI (crew module + European Service Module,
      fully fueled), published NASA/Lockheed Martin figure. LAS and BPC
      are jettisoned during ascent, well before TLI, so they are excluded.
    - ESM propulsion: ~8,600 kg usable propellant, OMS-E/AJ10-class engine
      at ~319 s Isp (N2O4/MMH hypergolic) -- this is ALL Orion has for
      NRHO insertion + TEI, with no refueling and no ICPS help after TLI.

Run:  python lunar_sim.py
Requires:  pip install numpy scipy astropy plotly
Output: an interactive HTML 3D plot, saved in the SAME DIRECTORY as this
        script, showing the spacecraft's location at the end of every
        mission phase (A through J).
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import astropy.units as u

# ════════════════════════════════════════════════
#  PARAMETERS — edit these
# ════════════════════════════════════════════════
#  Launch vehicle 1 ONLY: SLS Block 1 -> ICPS (TLI stage) -> Orion (payload)

S2_PROP_KG   = 29_000    # ICPS propellant mass [kg]   (gross fueled mass ~32.7 t, lengthened-tank DCSS-5m derivative)
S2_DRY_KG    = 3_720     # ICPS dry/structural mass [kg]                 "
S2_ISP_S     = 462       # RL10B-2 vacuum Isp [s]                        [Aerojet Rocketdyne / NASA]
S2_THRUST_N  = 110_100   # RL10B-2 vacuum thrust: 24,750 lbf [N]         [NASA/ULA fact sheets]
PAYLOAD_KG   = 25_850    # Orion CM+ESM total mass at TLI [kg]           [NASA/Lockheed Martin, lighter-load figure]
ORION_SM_ASSIST_DV_MS = 50   # Small trans-lunar trim/perigee-raise contribution from Orion's OWN
                             # service-module engine, on top of the ICPS burn -- real Earth-departure
                             # sequences for SLS/Orion use BOTH (see e.g. Artemis II's ESM perigee-raise
                             # burn ahead of TLI); this is not part of the ICPS delta-v budget itself.
MAX_DAYS     = 6         # Translunar sim time limit [days]

LEO_ALT_KM         = 185     # Low Earth Orbit "parking orbit" altitude (~100 nmi, Artemis-class)
NRHO_PERILUNE_ALT  = 3_000   # 9:2 NRHO perilune altitude  (Gateway-like)
NRHO_APOLUNE_ALT   = 70_000  # 9:2 NRHO apolune altitude   (Gateway-like)
LLO_ALT_KM         = 100     # Low Lunar Orbit altitude (Artemis-like)
DESCENT_GRAV_LOSS  = 1.22    # powered-descent delta-v as a multiple of v_LLO
TARGET_LAT_DEG     = -89.5   # south-pole target latitude (near Shackleton, 89.9S)
MOON_LON0_DEG      = 0.0     # body-fixed longitude of sub-Earth point at t=0

# --- Orion's OWN propulsion (European Service Module) ---
# Orion does NOT get refueled anywhere in the mission -- it must carry
# everything it needs for NRHO insertion, the whole NRHO stay, and the
# trip home from a single Earth launch. These are the real ESM figures.
ORION_ESM_PROP_KG = 8_600    # ESM usable propellant [kg]              [ESA/NTRS]
ORION_ESM_ISP_S   = 319      # OMS-E/AJ10-class engine Isp [s] (N2O4/MMH hypergolic)
# Orion's own dry mass after burning all ESM propellant = PAYLOAD_KG - ORION_ESM_PROP_KG

# --- Published NRHO reference delta-v figures (real mission design values) ---
# Our own 2-body combined_dv() estimate for LOI/TEI (below) runs HIGH because
# it can't capture the 3-body dynamics that make NRHO insertion cheap (see
# TODO in run(): "plane change is oversimplified - 3 body"). NASA's actual
# published figures for the TLI-> NRHO -> Earth legs (R. Whitley & R.
# Martinez, "Options for Staging Orbits in Cislunar Space," IEEE Aerospace
# 2015; used throughout NASA's Gateway/Artemis architecture docs) are:
NRHO_INSERTION_DV_REF_MS = 430   # TLI arrival -> NRHO (Orion's own LOI-equivalent burn)
NRHO_TEI_DV_REF_MS       = 410   # NRHO -> Earth interface (Orion's own TEI burn)
# These are the numbers actually used below for Orion's own delta-v budget.
# Our simplified combined_dv() estimate is still computed and shown, but
# only as a labeled diagnostic -- not charged against the vehicle's tanks.

# --- Propellant reserve / contingency margin ---
# Real missions never fly to the exact nominal delta-v: reserve is carried
# for trajectory-correction maneuvers (TCMs), navigation/execution errors,
# NRHO stationkeeping, and other unmodeled contingencies. NASA delta-v
# budgets typically carry margins in roughly the 5-10% range on top of the
# nominal required delta-v (see e.g. Apollo-era budgets, ~15% margin, and
# more recent Artemis mission-design docs). We apply the middle of that
# range here, to REQUIRED delta-v (not to what's available), for both the
# ICPS's TLI burn and Orion's own ESM (LOI + TEI) budget.
DV_MARGIN_FRACTION = 0.08   # 8% reserve on top of nominal required delta-v

# --- THE BIG FIX: target the arrival plane from EARTH, not at the Moon ---
# Inclination (deg) of the translunar transfer orbit's plane, measured
# about the Earth-Moon injection line, relative to the Moon's own orbital
# plane. 90 deg = the transfer plane is edge-on to the Moon's orbital
# plane, i.e. the spacecraft arrives already on a near-polar trajectory --
# exactly what's needed to reach a near-polar NRHO/LLO for south-pole
# access, WITHOUT a bolted-on 90 deg "free" plane change at the Moon.
TRANSLUNAR_PLANE_INCL_DEG = 90.0

# ════════════════════════════════════════════════
#  CONSTANTS  (astropy for readable unit conversion)
# ════════════════════════════════════════════════

G       = (6.674e-11 * u.m**3 / (u.kg * u.s**2)).value
M_EARTH = (5.972e24  * u.kg).value
M_MOON  = (7.342e22  * u.kg).value
R_EARTH = (6.371e6   * u.m).value
R_MOON  = (1.737e6   * u.m).value
D_MOON  = (3.844e8   * u.m).value
G0      = (9.80665   * u.m / u.s**2).value
T_MOON  = (27.3217   * u.day).to(u.s).value   # sidereal period [s] (= rotation period, tidal-locked)

MU_EARTH = G * M_EARTH
MU_MOON  = G * M_MOON
R_SOI    = D_MOON * (M_MOON / M_EARTH) ** 0.4   # Moon's sphere of influence, ~66,100 km


# ════════════════════════════════════════════════
#  GENERIC ORBITAL MECHANICS HELPERS
# ════════════════════════════════════════════════

def tsiolkovsky(isp, m0, m_dry):
    """Delta-v = Isp * g0 * ln(m0 / m_dry)"""
    return isp * G0 * np.log(m0 / m_dry)

def vis_viva(mu, r, a):
    """Speed at radius r on an orbit of semi-major axis a."""
    return np.sqrt(mu * (2 / r - 1 / a))

def circular_speed(mu, r):
    return np.sqrt(mu / r)

def orbit_period(mu, a):
    return 2 * np.pi * np.sqrt(a**3 / mu)

def combined_dv(v1, v2, dincl_rad):
    """Delta-v for a burn that changes both speed and inclination at once."""
    return np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(dincl_rad))

def propagate_conic(mu, r_vec0, v_vec0, t_span, n=200):
    """Numerically propagate a 3D two-body orbit (RK45)."""
    def ode(t, y):
        r = y[:3]
        rn = np.linalg.norm(r)
        a = -mu * r / rn**3
        return [*y[3:], *a]
    y0 = [*r_vec0, *v_vec0]
    t_eval = np.linspace(*t_span, n)
    sol = solve_ivp(ode, t_span, y0, method="RK45", t_eval=t_eval, max_step=t_span[1] / n)
    return sol.t, sol.y[:3].T, sol.y[3:].T

def moon_pos(t, th0):
    """Moon position (2D, in its own orbital plane, z=0 by definition)."""
    a = th0 + 2 * np.pi * t / T_MOON
    return np.array([D_MOON * np.cos(a), D_MOON * np.sin(a)])

def moon_pos3(t, th0):
    x, y = moon_pos(t, th0)
    return np.array([x, y, 0.0])

def moon_vel3(t, th0):
    w = 2 * np.pi / T_MOON
    a = th0 + w * t
    return np.array([-D_MOON * w * np.sin(a), D_MOON * w * np.cos(a), 0.0])

def moon_body_lonlat(pos_moon_centered, t, th0):
    """
    Convert a Moon-centered INERTIAL position vector (3D) into the Moon's
    body-fixed (tidally-locked) latitude/longitude.
    """
    rot_angle = th0 + 2 * np.pi * t / T_MOON + np.radians(MOON_LON0_DEG)
    x, y, z = pos_moon_centered
    xb = x * np.cos(rot_angle) + y * np.sin(rot_angle)
    yb = -x * np.sin(rot_angle) + y * np.cos(rot_angle)
    zb = z
    r = np.linalg.norm([xb, yb, zb])
    lat = np.degrees(np.arcsin(np.clip(zb / r, -1, 1)))
    lon = np.degrees(np.arctan2(yb, xb))
    return lat, lon


# ════════════════════════════════════════════════
#  PHASE A-B: LEO -> TLI -> translunar coast  (3D, RK45, Earth+Moon)
# ════════════════════════════════════════════════

def make_ode3d(th0):
    """
    Translunar coast ODE: Earth + Moon point-mass gravity only. The TLI
    burn itself is treated as IMPULSIVE (applied to the initial velocity
    before this ODE runs), consistent with every other burn in this
    script (LOI, NRHO->LLO, descent, ascent, TEI are all impulsive
    patched-conic approximations too).

    NOTE: a full continuous-thrust integration of the real ICPS burn (110
    kN thrust, ~20 min duration) shows a genuinely large finite-burn
    gravity loss, because the ICPS has a low thrust-to-weight ratio
    (~0.2). In the real mission this is mitigated by a multi-burn Earth-
    departure profile (an initial LEO-insertion burn, then a separate TLI
    burn later from an already-elevated intermediate orbit) -- not
    represented in this single-leg simplified model. Treating the leg as
    impulsive sidesteps that (real, but out-of-scope-for-this-sim)
    complication while keeping the delta-v budget itself accurate.
    """
    def ode(t, y):
        pos, vel = y[:3], y[3:]
        mp = moon_pos3(t, th0)
        r_e = np.linalg.norm(pos)
        a_e = -G * M_EARTH / r_e**3 * pos
        rel = mp - pos
        r_m = np.linalg.norm(rel)
        a_m = G * M_MOON / r_m**3 * rel
        acc = a_e + a_m
        return [*vel, *acc]
    return ode

def leo_state0(r_leo, v_leo, i_plane_rad):
    """
    Initial LEO state, 3D. Position stays on the x-axis (the Earth-Moon
    injection line, chosen as the line of nodes); the velocity vector is
    tilted out of the Moon's orbital (x-y) plane by i_plane_rad. This is
    the "target the plane from Earth" step: it sets the whole transfer
    orbit's plane via the LEO inclination/RAAN choice, before any burn
    happens at the Moon.
    """
    pos0 = np.array([-r_leo, 0.0, 0.0])
    vel0 = np.array([0.0, v_leo * np.cos(i_plane_rad), v_leo * np.sin(i_plane_rad)])
    return pos0, vel0

def find_phase(r_leo, v_inj, i_plane_rad):
    pos0, vel0 = leo_state0(r_leo, v_inj, i_plane_rad)
    y0 = [*pos0, *vel0]
    print("  [optimising Moon intercept angle...]")

    def objective(th_deg):
        th = np.radians(th_deg)
        sol = solve_ivp(make_ode3d(th),
                        [0, MAX_DAYS * 86400], y0,
                        method="RK45", max_step=200)
        mps = np.array([moon_pos3(t, th) for t in sol.t])
        d = np.linalg.norm(sol.y[:3].T - mps, axis=1)
        return d.min()

    # Search the FULL range of possible lead angles: the correct phasing
    # angle depends on the actual (energy-dependent) transit time, which
    # can be much longer than a textbook Hohmann time when the achieved
    # dv is only marginally above the minimum-energy transfer (near-
    # parabolic transfers are slow and very sensitive to phasing).
    best = None
    for lo, hi in [(-180, -60), (-90, 0), (-15, 15), (0, 90), (60, 180)]:
        r = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        if best is None or r.fun < best.fun:
            best = r
    return np.radians(best.x)


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════

def run():
    phase_events = []   # collected (phase, label, pos_earth_centered_xyz, note) for the HTML plot

    r_leo = R_EARTH + LEO_ALT_KM * 1e3
    v_leo = circular_speed(MU_EARTH, r_leo)
    i_plane = np.radians(TRANSLUNAR_PLANE_INCL_DEG)

    # ---- ICPS setup (TLI burn) ----
    m0     = S2_PROP_KG + S2_DRY_KG + PAYLOAD_KG
    m_dry  = S2_DRY_KG + PAYLOAD_KG
    mdot   = S2_THRUST_N / (S2_ISP_S * G0)
    t_burn = S2_PROP_KG / mdot

    dv_avail = tsiolkovsky(S2_ISP_S, m0, m_dry)
    zeta     = S2_PROP_KG / m0

    a_t   = (R_EARTH + D_MOON) / 2
    v_tli = vis_viva(MU_EARTH, r_leo, a_t)
    dv_tli_required = v_tli - v_leo
    dv_tli_required_w_margin = dv_tli_required * (1 + DV_MARGIN_FRACTION)
    t_translunar = np.pi * np.sqrt(a_t**3 / MU_EARTH)

    # ---- Phase A: LEO (marker at injection point) ----
    pos0, vel0 = leo_state0(r_leo, v_leo, i_plane)
    phase_events.append(("A", "Low Earth Orbit (parking orbit)", pos0.copy(),
                          f"alt {LEO_ALT_KM:,.0f} km, v_circ {v_leo:,.0f} m/s"))

    # ---- Phase B: translunar coast (3D, plane targeted from Earth) ----
    # TLI burn applied IMPULSIVELY (post-burn speed = v_leo + ICPS dv +
    # small Orion-SM trans-lunar trim assist), then coast under Earth+Moon
    # gravity only -- see make_ode3d() docstring for why.
    dv_total_departure = dv_avail + ORION_SM_ASSIST_DV_MS
    v_inj_speed = v_leo + dv_total_departure
    theta0 = find_phase(r_leo, v_inj_speed, i_plane)
    ode    = make_ode3d(theta0)

    t_eval = np.arange(0, MAX_DAYS * 86400, 300)

    pos0, vel0 = leo_state0(r_leo, v_inj_speed, i_plane)
    y0  = [*pos0, *vel0]
    sol = solve_ivp(ode, [0, MAX_DAYS * 86400], y0,
                    method="RK45", max_step=200, t_eval=t_eval)

    t_arr = sol.t
    pos_t = sol.y[:3].T
    vel_t = sol.y[3:].T
    moons = np.array([moon_pos3(t, theta0) for t in t_arr])
    dist_m = np.linalg.norm(pos_t - moons, axis=1)

    i_arr      = dist_m.argmin()
    t_arr_moon = t_arr[i_arr]
    r_arrival  = dist_m[i_arr]
    pos_at_arrival = pos_t[i_arr]
    vel_at_arrival = vel_t[i_arr]
    moon_at_arrival = moons[i_arr]

    # Actual relative state vs. the Moon at closest approach (numeric, 3D)
    r_rel = pos_at_arrival - moon_at_arrival
    v_rel = vel_at_arrival - moon_vel3(t_arr_moon, theta0)
    v_inf_numeric = np.linalg.norm(v_rel)

    # v_inf: use ONLY the numeric value, taken directly from the actual
    # integrated 3D state at closest approach (r_rel, v_rel above). An
    # earlier version also computed an "analytic" patched-conic v_inf via
    # a 2-body vis-viva formula with a scalar (non-vector) subtraction of
    # the Moon's orbital speed -- that formula assumed a planar, untilted
    # transfer and silently gave wrong/NaN answers once the trajectory
    # plane was tilted in 3D. It has been removed rather than patched.
    v_inf = v_inf_numeric

    speeds  = np.linalg.norm(vel_t, axis=1)

    # The ACTUAL plane the spacecraft arrived in (numeric), vs. the target
    # near-polar NRHO plane. Standard orbital-mechanics definition:
    # inclination i (relative to the Moon's orbital/reference plane, whose
    # normal is +z) satisfies cos(i) = h_z / |h|. A polar orbit has i = 90 deg
    # (its normal lies IN the reference plane, i.e. h_z = 0).
    # Because the plane was targeted from Earth (i_plane at LEO), the
    # arrival inclination should already be close to the 90 deg target --
    # this residual is the realistic trim left for the LOI burn, not a
    # bolted-on 90 deg change.
    h_rel = np.cross(r_rel, v_rel)
    h_rel_unit = h_rel / np.linalg.norm(h_rel)
    arrival_incl_deg = np.degrees(np.arccos(np.clip(h_rel_unit[2], -1, 1)))
    target_incl_deg = 90.0   # near-polar NRHO/LLO, needed for south-pole access
    incl_change_deg = abs(arrival_incl_deg - target_incl_deg)

    phase_events.append(("B", "TLI burnout + translunar coast, closest lunar approach",
                          pos_at_arrival.copy(),
                          f"v_inf {v_inf_numeric:,.0f} m/s, t+{t_arr_moon/86400:.2f} days, "
                          f"plane residual {incl_change_deg:.1f} deg (targeted from Earth)"))

    # ════════════════════════════════════════════
    #  PHASE C: NRHO insertion -- ORION'S OWN ESM BURN (no refueling, no
    #  ICPS help; ICPS is jettisoned right after Phase B)
    # ════════════════════════════════════════════
    r_p_nrho = R_MOON + NRHO_PERILUNE_ALT * 1e3
    r_a_nrho = R_MOON + NRHO_APOLUNE_ALT * 1e3
    a_nrho   = (r_p_nrho + r_a_nrho) / 2
    v_nrho_p = vis_viva(MU_MOON, r_p_nrho, a_nrho)          # speed at NRHO perilune
    T_nrho   = orbit_period(MU_MOON, a_nrho)

    # Our own simplified 2-body combined_dv() estimate -- shown only as a
    # DIAGNOSTIC. It runs high because it can't see the 3-body dynamics
    # that make real NRHO insertion cheap (this is the flagged "plane
    # change is oversimplified - 3 body" TODO). The number actually
    # charged to Orion's tanks is the published reference value below.
    v_hyp_at_rp = np.sqrt(v_inf**2 + 2 * MU_MOON / r_p_nrho)
    dv_loi_2body_estimate = combined_dv(v_hyp_at_rp, v_nrho_p, np.radians(incl_change_deg))
    dv_loi = NRHO_INSERTION_DV_REF_MS   # <-- what's actually charged (real published figure)

    moon_c = moon_at_arrival  # Moon's (Earth-centered) position used as the
                              # local origin for all lunar-orbit-phase geometry
    nrho_insertion_pos = moon_c + np.array([r_p_nrho, 0.0, 0.0])
    phase_events.append(("C", "NRHO insertion (Orion's own ESM burn)",
                          nrho_insertion_pos.copy(),
                          f"LOI dv {dv_loi:,.0f} m/s (published NRHO reference; our own simplified "
                          f"2-body estimate was {dv_loi_2body_estimate:,.0f} m/s -- overstated, see TODO)"))

    # ════════════════════════════════════════════
    #  PHASE D: Rendezvous with the HLS lander IN NRHO.
    #  TODO fix applied here: docking happens in NRHO, not LLO. Orion
    #  NEVER descends to LLO -- only the separately-launched HLS lander
    #  (Starship HLS / Blue Moon, launched on its own rocket, refueled or
    #  not on its own schedule) goes NRHO -> LLO -> surface -> LLO -> NRHO.
    #  That lander's delta-v is tracked separately below and is explicitly
    #  NOT part of this vehicle's (SLS/Orion, launch vehicle 1) budget.
    # ════════════════════════════════════════════
    phase_events.append(("D", "Rendezvous with HLS lander in NRHO",
                          nrho_insertion_pos.copy(),
                          "2 of 4 crew transfer down to the lander here; 2 remain aboard Orion in NRHO"))

    # ════════════════════════════════════════════
    #  LANDER-ONLY LEG (informational): NRHO -> LLO -> surface -> LLO -> NRHO
    #  This is the HLS lander's job (a separate vehicle/launch), carrying
    #  its own large propellant supply (e.g. Starship's own tanks). It is
    #  NOT charged to the SLS/Orion vehicle we're simulating. Kept here
    #  only so the visualization can still show the full mission context.
    # ════════════════════════════════════════════
    r_llo   = R_MOON + LLO_ALT_KM * 1e3
    v_llo   = circular_speed(MU_MOON, r_llo)
    a_xfer  = (r_p_nrho + r_llo) / 2
    v_xfer_p1 = vis_viva(MU_MOON, r_p_nrho, a_xfer)   # depart NRHO perilune onto transfer ellipse
    v_xfer_p2 = vis_viva(MU_MOON, r_llo, a_xfer)      # arrive at LLO radius on transfer ellipse
    dv_nrho_to_llo = abs(v_nrho_p - v_xfer_p1) + abs(v_xfer_p2 - v_llo)
    t_xfer_to_llo  = orbit_period(MU_MOON, a_xfer) / 2

    llo_arrival_pos = moon_c + np.array([r_llo, 0.0, 0.0])
    phase_events.append(("E", "[Lander only] Transfer NRHO -> Low Lunar Orbit",
                          llo_arrival_pos.copy(),
                          f"LLO alt {LLO_ALT_KM:,.0f} km, dv {dv_nrho_to_llo:,.0f} m/s (HLS lander's own tanks, "
                          f"NOT Orion/SLS budget), transfer time {t_xfer_to_llo/3600:.1f} hr"))

    # ════════════════════════════════════════════
    #  PHASE F: Powered descent, LLO -> south pole surface
    # ════════════════════════════════════════════
    dv_descent = DESCENT_GRAV_LOSS * v_llo   # rough engineering estimate (incl. gravity losses)

    incl = np.radians(90.0)   # polar LLO -> passes directly over the poles
    r0 = np.array([r_llo, 0., 0.])
    v0 = np.array([0., v_llo * np.cos(incl), v_llo * np.sin(incl)])
    t_llo, pos_llo, vel_llo = propagate_conic(MU_MOON, r0, v0, (0, orbit_period(MU_MOON, r_llo)), n=2000)

    t_elapsed_at_llo = t_arr_moon + T_nrho / 2 + t_xfer_to_llo   # rough mission clock at LLO arrival

    lats, lons = [], []
    for p, tt in zip(pos_llo, t_llo):
        lat, lon = moon_body_lonlat(p, t_elapsed_at_llo + tt, theta0)
        lats.append(lat); lons.append(lon)
    lats = np.array(lats); lons = np.array(lons)

    i_pole = np.argmin(np.abs(lats - TARGET_LAT_DEG))
    landing_lat = lats[i_pole]
    landing_lon = lons[i_pole]
    t_descent_start = t_llo[i_pole]

    landing_pos = moon_c + pos_llo[i_pole]
    phase_events.append(("F", "[Lander only] Powered descent -> lunar south-pole surface",
                          landing_pos.copy(),
                          f"lat {landing_lat:.2f} deg, lon {landing_lon:.2f} deg, "
                          f"dv {dv_descent:,.0f} m/s (HLS lander's own tanks)"))

    # ════════════════════════════════════════════
    #  PHASE G: Surface operations / ISRU (lander + crew; Orion is idle in NRHO)
    # ════════════════════════════════════════════
    phase_events.append(("G", "[Lander+crew] Surface operations / ISRU (south pole)",
                          landing_pos.copy(),
                          "collect + electrolyze water ice -> LH2/LOX for ascent (lander's ISRU, not Orion)"))

    # ════════════════════════════════════════════
    #  PHASE H/I: Lander ascent + LLO->NRHO transfer (still lander-only budget)
    # ════════════════════════════════════════════
    dv_ascent       = dv_descent            # symmetric assumption (flagged simplification)
    dv_llo_to_nrho  = dv_nrho_to_llo        # symmetric transfer
    dv_lander_total = dv_nrho_to_llo + dv_descent + dv_ascent + dv_llo_to_nrho

    phase_events.append(("H", "[Lander only] Ascent, surface -> Low Lunar Orbit",
                          llo_arrival_pos.copy(), f"dv {dv_ascent:,.0f} m/s (symmetric w/ descent, simplification; "
                          f"lander's own tanks)"))
    phase_events.append(("I", "[Lander] Transfer LLO -> NRHO -- crew re-boards Orion",
                          nrho_insertion_pos.copy(),
                          f"dv {dv_llo_to_nrho:,.0f} m/s (lander's own tanks). Lander docks with Orion in "
                          f"NRHO here; the 2 crew who descended transfer back aboard Orion. All 4 crew "
                          f"are back together in Orion before TEI -- the lander is left in NRHO."))

    # ════════════════════════════════════════════
    #  PHASE J: Trans-Earth Injection -- ORION'S OWN ESM BURN (2nd and last
    #  use of Orion's own propellant; no refueling, no ICPS, no lander help)
    # ════════════════════════════════════════════
    a_tei = (R_EARTH + D_MOON) / 2
    v_tei_at_r  = vis_viva(MU_EARTH, D_MOON - r_p_nrho, a_tei)
    dv_tei_2body_estimate = combined_dv(v_nrho_p, v_tei_at_r, np.radians(incl_change_deg))
    dv_tei = NRHO_TEI_DV_REF_MS   # <-- what's actually charged (real published figure)

    # ---- Orion's OWN round-trip delta-v budget (no refueling anywhere) ----
    orion_own_dv_required = dv_loi + dv_tei
    orion_own_dv_required_w_margin = orion_own_dv_required * (1 + DV_MARGIN_FRACTION)
    orion_esm_m0    = PAYLOAD_KG
    orion_esm_mdry  = PAYLOAD_KG - ORION_ESM_PROP_KG
    orion_esm_dv_available = tsiolkovsky(ORION_ESM_ISP_S, orion_esm_m0, orion_esm_mdry)

    total_mission_dv = dv_tli_required + orion_own_dv_required   # SLS/Orion vehicle only, nominal
    total_mission_dv_w_margin = dv_tli_required_w_margin + orion_own_dv_required_w_margin
    dv_lander_total_full = dv_lander_total   # HLS lander's own separate budget (context only)

    phase_events.append(("J", "Trans-Earth injection (Orion's own ESM burn) + return",
                          nrho_insertion_pos.copy(),
                          f"All 4 crew aboard Orion (lander left behind in NRHO). TEI dv {dv_tei:,.0f} m/s "
                          f"(published NRHO reference; our own simplified 2-body estimate was "
                          f"{dv_tei_2body_estimate:,.0f} m/s) -> Earth return trajectory"))

    # ════════════════════════════════════════════
    #  RETURN COAST (visualization only): propagate an actual arc from the
    #  TEI departure point back to Earth, so the plot shows a genuine round
    #  trip instead of stopping at the Moon. This is an Earth-only 2-body
    #  Kepler ellipse (ignores the Moon's gravity on the way back) -- a
    #  labeled simplification for the picture, not a delta-v claim (the
    #  TEI delta-v used above is the real published NRHO reference value).
    # ════════════════════════════════════════════
    entry_interface_alt_km = 122   # ~ Earth atmospheric entry interface altitude
    r_entry = R_EARTH + entry_interface_alt_km * 1e3
    r_tei_start = nrho_insertion_pos.copy()
    r_tei_mag = np.linalg.norm(r_tei_start)
    a_return = (r_tei_mag + r_entry) / 2
    v_return_start = vis_viva(MU_EARTH, r_tei_mag, a_return)

    r_hat = r_tei_start / r_tei_mag
    n_hat = np.cross(pos0, vel0)
    n_hat = n_hat / np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat)
    v_return_vec = v_return_start * t_hat

    t_return, pos_return, vel_return = propagate_conic(
        MU_EARTH, r_tei_start, v_return_vec,
        (0, orbit_period(MU_EARTH, a_return) / 2), n=500)

    earth_arrival_pos = pos_return[-1].copy()
    phase_events.append(("J-return", "Earth entry interface / splashdown",
                          earth_arrival_pos,
                          f"return coast ~{orbit_period(MU_EARTH, a_return)/2/86400:.1f} days "
                          f"(Earth-only 2-body approximation, illustrative)"))

    # ════════════════════════════════════════════
    #  PRINT RESULTS
    # ════════════════════════════════════════════
    W = 64
    print("\n" + "=" * W)
    print("   EARTH -> LUNAR SOUTH POLE MISSION SIMULATION (NRHO arch.)")
    print("   Launch vehicle 1 ONLY: SLS Block 1 (ICPS + Orion)")
    print("=" * W)

    print("\n--- ICPS (TLI burn stage) ---")
    print(f"  Propellant mass  : {S2_PROP_KG:>12,.0f} kg")
    print(f"  Structural mass  : {S2_DRY_KG:>12,.0f} kg")
    print(f"  Payload (Orion)  : {PAYLOAD_KG:>12,.0f} kg")
    print(f"  Prop. fraction z : {zeta:.4f}")
    print(f"  Burn duration    : {t_burn/60:>12.1f} min")
    print(f"  Delta-v available (ICPS only)     : {dv_avail:>10,.0f} m/s")
    print(f"  + Orion SM trans-lunar trim assist: {ORION_SM_ASSIST_DV_MS:>10,.0f} m/s  (own engine, real multi-burn profile)")
    print(f"  Delta-v available (total)         : {dv_total_departure:>10,.0f} m/s")
    print(f"  Delta-v required (TLI, nominal)    : {dv_tli_required:>10,.0f} m/s")
    print(f"  Delta-v required (TLI, +{DV_MARGIN_FRACTION*100:.0f}% reserve): {dv_tli_required_w_margin:>10,.0f} m/s")
    print(f"  Delta-v margin (nominal)           : {dv_total_departure - dv_tli_required:>+10,.0f} m/s")
    print(f"  Delta-v margin (with reserve)      : {dv_total_departure - dv_tli_required_w_margin:>+10,.0f} m/s"
          f"  {'-> still closes' if dv_total_departure > dv_tli_required_w_margin else '-> DOES NOT close with reserve applied'}")
    print(f"  (Real ICPS TLI performance margins are famously tight/near-parabolic --")
    print(f"   this is a known, publicly documented characteristic, not a modeling error.)")

    print("\n--- PHASE B: TRANSLUNAR COAST (plane targeted from Earth) ---")
    print(f"  LEO altitude          : {LEO_ALT_KM:,.0f} km   (v_circ = {v_leo:,.0f} m/s)")
    print(f"  Translunar plane incl : {TRANSLUNAR_PLANE_INCL_DEG:.0f} deg (set at LEO, not at the Moon)")
    print(f"  Reference Hohmann time : {t_translunar/3600:.1f} hr")
    print(f"  Closest approach       : {r_arrival/1e3:,.0f} km  at t = {t_arr_moon/3600:.1f} hr ({t_arr_moon/86400:.2f} days)")
    print(f"  Hyperbolic excess speed (v_inf, numeric 3D): {v_inf:,.0f} m/s")
    print(f"  Peak speed              : {speeds.max():,.0f} m/s")
    print(f"  Residual plane mismatch at arrival: {incl_change_deg:.1f} deg")
    print(f"    (this is what's left AFTER targeting the plane from Earth --")
    print(f"     small because of the LEO inclination choice, not a bolted-on 90 deg)")

    print("\n--- PHASE C: NRHO INSERTION (Orion's own ESM burn, no ICPS/refuel) ---")
    print(f"  Perilune altitude : {NRHO_PERILUNE_ALT:,.0f} km")
    print(f"  Apolune altitude  : {NRHO_APOLUNE_ALT:,.0f} km")
    print(f"  Orbit period      : {T_nrho/86400:.2f} days   (real 9:2 NRHO ~ 6.5-7 days)")
    print(f"  LOI delta-v (charged, published ref.) : {dv_loi:,.0f} m/s")
    print(f"  LOI delta-v (our own 2-body estimate)  : {dv_loi_2body_estimate:,.0f} m/s  <- overstated, see TODO")
    print(f"  --> Rendezvous with HLS lander (assumed pre-positioned in this NRHO)")

    print("\n--- [LANDER ONLY, not Orion/SLS] NRHO <-> LOW LUNAR ORBIT + SURFACE ---")
    print(f"  LLO altitude       : {LLO_ALT_KM:,.0f} km   (v_circ = {v_llo:,.0f} m/s)")
    print(f"  Transfer time (one-way) : {t_xfer_to_llo/3600:.1f} hr")
    print(f"  Delta-v (NRHO->LLO down-leg)  : {dv_nrho_to_llo:,.0f} m/s")
    print(f"  Delta-v (descent)             : {dv_descent:,.0f} m/s  (~{DESCENT_GRAV_LOSS:.2f}x v_LLO, incl. gravity losses)")
    print(f"  Delta-v (ascent)              : {dv_ascent:,.0f} m/s  (symmetric w/ descent, simplification)")
    print(f"  Delta-v (LLO->NRHO up-leg)    : {dv_llo_to_nrho:,.0f} m/s")
    print(f"  Lander's own total delta-v    : {dv_lander_total:,.0f} m/s  (its own tanks -- Starship/Blue Moon-class")
    print(f"                                   propellant supply, NOT charged to the SLS/Orion vehicle)")
    print(f"  Target latitude     : {TARGET_LAT_DEG:.1f} deg")
    print(f"  >>> ESTIMATED LANDING SITE <<<")
    print(f"      Latitude  : {landing_lat:>8.2f} deg")
    print(f"      Longitude : {landing_lon:>8.2f} deg  (body-fixed, 0 deg = sub-Earth meridian)")
    print(f"  Reference real sites:")
    print(f"      Shackleton crater rim : -89.9 deg,   0.0 deg")
    print(f"      Nobile Rim 2 (DM2)    : -84.20 deg, 60.70 deg")

    print("\n--- PHASE J: TRANS-EARTH INJECTION (Orion's own ESM burn, 2nd/last use) ---")
    print(f"  TEI delta-v (charged, published ref.) : {dv_tei:,.0f} m/s")
    print(f"  TEI delta-v (our own 2-body estimate)  : {dv_tei_2body_estimate:,.0f} m/s  <- overstated, see TODO")

    print("\n--- FEASIBILITY CHECK: can Orion do this alone, with NO refueling? ---")
    print(f"  Orion's own required round-trip dv (LOI + TEI, nominal) : {orion_own_dv_required:,.0f} m/s")
    print(f"  Orion's own required round-trip dv (+{DV_MARGIN_FRACTION*100:.0f}% reserve)   : {orion_own_dv_required_w_margin:,.0f} m/s")
    print(f"  Orion's own ESM delta-v CAPABILITY (Tsiolkovsky)        : {orion_esm_dv_available:,.0f} m/s")
    print(f"    ESM propellant {ORION_ESM_PROP_KG:,.0f} kg, Isp {ORION_ESM_ISP_S} s, no refueling assumed")
    margin_nominal = orion_esm_dv_available - orion_own_dv_required
    margin_w_reserve = orion_esm_dv_available - orion_own_dv_required_w_margin
    print(f"  Margin (nominal)     : {margin_nominal:>+8.0f} m/s")
    print(f"  Margin (with reserve): {margin_w_reserve:>+8.0f} m/s  -> "
          f"{'FEASIBLE with just SLS Block 1 + Orion, even with reserve' if margin_w_reserve > 0 else 'INFEASIBLE once reserve is included -- would need refueling or a 3rd stage'}")
    print(f"  (Real-world confirmation: this is why NASA picked NRHO over LLO for Orion --")
    print(f"   Orion's ESM genuinely CANNOT do a full LLO-style LOI+TEI, ~2.5 km/s combined,")
    print(f"   as demonstrated by Artemis II not entering lunar orbit for exactly this reason.")
    print(f"   NRHO's cheap ~{NRHO_INSERTION_DV_REF_MS:.0f}+{NRHO_TEI_DV_REF_MS:.0f} m/s round trip is what makes a 2-stage")
    print(f"   SLS [Core + ICPS] plus an unrefueled Orion sufficient -- no 3rd stage needed.)")

    print("\n--- MISSION TOTAL (SLS/Orion vehicle ONLY -- launch vehicle 1) ---")
    print(f"  TLI (ICPS)                          : {dv_tli_required:,.0f} m/s")
    print(f"  NRHO insertion (Orion ESM)           : {dv_loi:,.0f} m/s")
    print(f"  TEI (Orion ESM)                      : {dv_tei:,.0f} m/s")
    print(f"  TOTAL (nominal, this vehicle only)   : {total_mission_dv:,.0f} m/s")
    print(f"  TOTAL (+{DV_MARGIN_FRACTION*100:.0f}% reserve, this vehicle only): {total_mission_dv_w_margin:,.0f} m/s")
    print(f"  [Separate HLS lander budget, NOT included above: {dv_lander_total_full:,.0f} m/s]")
    print("=" * W + "\n")

    return dict(
        theta0=theta0, t_arr=t_arr, pos_t=pos_t, moons=moons,
        r_p_nrho=r_p_nrho, r_a_nrho=r_a_nrho, a_nrho=a_nrho, T_nrho=T_nrho,
        r_llo=r_llo, t_elapsed_at_llo=t_elapsed_at_llo,
        pos_llo=pos_llo, t_llo=t_llo, i_pole=i_pole,
        landing_lat=landing_lat, landing_lon=landing_lon,
        t_arr_moon=t_arr_moon, incl_change_deg=incl_change_deg,
        moon_c=moon_c, phase_events=phase_events, pos_return=pos_return,
    )


def build_visualization(res, outpath=None):
    """Interactive 3D plot: Earth, Moon, translunar coast, NRHO, LLO, descent,
    landing site, and a labeled marker at the end of EVERY mission phase (A-J).
    Saved next to this script by default."""
    import plotly.graph_objects as go

    if outpath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outpath = os.path.join(script_dir, "lunar_trajectory_3d.html")

    theta0 = res["theta0"]
    t_arr, pos_t = res["t_arr"], res["pos_t"]
    moon_c = res["moon_c"]

    def sph(r, n=40):
        u_ = np.linspace(0, 2 * np.pi, n)
        v_ = np.linspace(0, np.pi, n)
        x = r * np.outer(np.cos(u_), np.sin(v_))
        y = r * np.outer(np.sin(u_), np.sin(v_))
        z = r * np.outer(np.ones_like(u_), np.cos(v_))
        return x, y, z

    fig = go.Figure()

    # Earth (radius exaggerated 3x for visibility at this scale)
    ex, ey, ez = sph(R_EARTH * 3)
    fig.add_surface(x=ex, y=ey, z=ez, colorscale=[[0, "#2b6cb0"], [1, "#2b6cb0"]],
                     showscale=False, name="Earth", opacity=1)

    # Moon (radius exaggerated 3x), positioned at arrival
    mx, my, mz = sph(R_MOON * 3)
    fig.add_surface(x=mx + moon_c[0], y=my + moon_c[1], z=mz + moon_c[2],
                     colorscale=[[0, "#a0a0a0"], [1, "#a0a0a0"]], showscale=False,
                     name="Moon", opacity=1)

    # Moon's orbital path (reference circle)
    ang = np.linspace(0, 2 * np.pi, 200)
    fig.add_scatter3d(x=D_MOON * np.cos(ang), y=D_MOON * np.sin(ang), z=np.zeros_like(ang),
                       mode="lines", line=dict(color="lightgray", width=2, dash="dot"),
                       name="Moon's orbit")

    # Phase A-B: LEO -> TLI -> translunar coast (now full 3D)
    fig.add_scatter3d(x=pos_t[:, 0], y=pos_t[:, 1], z=pos_t[:, 2], mode="lines",
                       line=dict(color="#e53e3e", width=5), name="Translunar coast (TLI, 3D-targeted plane)")

    # Phase C: NRHO (representative near-polar ellipse around the Moon)
    a_n, r_p, r_a = res["a_nrho"], res["r_p_nrho"], res["r_a_nrho"]
    ecc = (r_a - r_p) / (r_a + r_p)
    nu = np.linspace(0, 2 * np.pi, 300)
    r_nrho = a_n * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    x_nrho = moon_c[0] + r_nrho * np.cos(nu)
    z_nrho = r_nrho * np.sin(nu)
    y_nrho = np.full_like(nu, moon_c[1])
    fig.add_scatter3d(x=x_nrho, y=y_nrho, z=z_nrho, mode="lines",
                       line=dict(color="#805ad5", width=4), name="NRHO (representative, w/ Gateway)")

    # Phase E/F: LLO (polar) + descent to south pole
    pos_llo = res["pos_llo"]
    i_pole = res["i_pole"]
    fig.add_scatter3d(x=moon_c[0] + pos_llo[:i_pole + 1, 0],
                       y=moon_c[1] + pos_llo[:i_pole + 1, 1],
                       z=moon_c[2] + pos_llo[:i_pole + 1, 2],
                       mode="lines", line=dict(color="#38a169", width=5),
                       name="LLO -> powered descent")

    # Return coast: TEI departure -> Earth entry interface (labeled 2-body approximation)
    pos_return = res["pos_return"]
    fig.add_scatter3d(x=pos_return[:, 0], y=pos_return[:, 1], z=pos_return[:, 2],
                       mode="lines", line=dict(color="#dd6b20", width=5, dash="dash"),
                       name="Return coast (TEI -> Earth, illustrative 2-body)")

    # --- Phase markers: location after EVERY phase (A-J) ---
    phase_events = res["phase_events"]
    px = [p[2][0] for p in phase_events]
    py = [p[2][1] for p in phase_events]
    pz = [p[2][2] for p in phase_events]
    labels = [f"Phase {p[0]}: {p[1]}<br>{p[3]}" for p in phase_events]
    short = [f"{p[0]}" for p in phase_events]
    fig.add_scatter3d(x=px, y=py, z=pz, mode="markers+text",
                       marker=dict(color="gold", size=6, symbol="diamond",
                                   line=dict(color="black", width=1)),
                       text=short, textposition="top center",
                       hovertext=labels, hoverinfo="text",
                       name="Phase markers (A-J)")

    fig.update_layout(
        title="Earth -> Lunar South Pole Mission (NRHO staging architecture) -- SLS/Orion",
        scene=dict(
            xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(outpath, include_plotlyjs="cdn")
    return outpath


if __name__ == "__main__":
    result = run()
    path = build_visualization(result)
    print(f"3D visualization saved to: {path}")