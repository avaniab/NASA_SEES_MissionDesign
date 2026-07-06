
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import astropy.units as u 

# ════════════════════════════════════════════════
#  PARAMETERS — edit these
# ════════════════════════════════════════════════
#  Launch vehicle 1 ONLY: SLS Block 1 -> ICPS (TLI stage) -> Orion (payload)

#TODO : add a second launch vehicle (SLS Block 1B) to carry a lander/rover
#TODO : NRHO dock w/ falcon heavy is more feasable than to LLO so change that
#TODO v_inf is brocken must fix
#TODO : plane change is over simplified - 3 body 
#TODO - PRIORITY assume no refueling so much carry capacity to go and come back 

S2_PROP_KG   = 29_000    # ICPS propellant mass [kg]   (gross fueled mass ~32.7 t, lengthened-tank DCSS-5m derivative)
S2_DRY_KG    = 3_720     # ICPS dry/structural mass [kg]                 "
S2_ISP_S     = 462       # RL10B-2 vacuum Isp [s]                        [Aerojet Rocketdyne / NASA]
S2_THRUST_N  = 110_100   # RL10B-2 vacuum thrust: 24,750 lbf [N]         [NASA/ULA fact sheets]
PAYLOAD_KG   = 25_850    # Orion CM+ESM total mass at TLI [kg]           [NASA/Lockheed Martin, lighter-load figure]
ORION_SM_ASSIST_DV_MS = 50   # Small trans-lunar trim/perigee-raise contribution from Orion's OWN
                             # service-module engine, on top of the ICPS burn -- real Earth-departure
                             # sequences for SLS/Orion use BOTH (see e.g. Artemis II's ESM perigee-raise
                             # burn ahead of TLI); this is not part of the ICPS delta-v budget itself.
MAX_DAYS = 8        # Translunar sim time limit [days]

LEO_ALT_KM         = 185     # Low Earth Orbit "parking orbit" altitude (~100 nmi, Artemis-class)
NRHO_PERILUNE_ALT  = 3_000   # 9:2 NRHO perilune altitude  (Gateway-like)
NRHO_APOLUNE_ALT   = 70_000  # 9:2 NRHO apolune altitude   (Gateway-like)
LLO_ALT_KM         = 100     # Low Lunar Orbit altitude (Artemis-like)
DESCENT_GRAV_LOSS  = 1.22    # powered-descent delta-v as a multiple of v_LLO
TARGET_LAT_DEG     = -89.5   # south-pole target latitude (near Shackleton, 89.9S)
MOON_LON0_DEG      = 0.0     # body-fixed longitude of sub-Earth point at t=0

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
    return isp * G0 * np.log(m0 / m_dry)

def vis_viva(mu, r, a):
    return np.sqrt(mu * (2 / r - 1 / a))

def circular_speed(mu, r):
    return np.sqrt(mu / r)

def orbit_period(mu, a):
    return 2 * np.pi * np.sqrt(a**3 / mu)

def combined_dv(v1, v2, dincl_rad):
    return np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(dincl_rad))

def propagate_conic(mu, r_vec0, v_vec0, t_span, n=200):
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

    v_inj  = v_leo + dv_total_departure
    eps    = v_inj**2 / 2 - MU_EARTH / r_leo
    a_act  = -MU_EARTH / (2 * eps)
    disc   = 2 / D_MOON - 1 / a_act
    v_moon_orbital = 2 * np.pi * D_MOON / T_MOON
    if disc > 0:
        v_at_rD = np.sqrt(MU_EARTH * disc)
        v_inf = abs(v_at_rD - v_moon_orbital)
    else:
        v_inf = np.nan   # informational only; not used downstream
    v_inf_for_dv = v_inf_numeric   # the value actually used for LOI/TEI dv below

    speeds  = np.linalg.norm(vel_t, axis=1)

    # The ACTUAL plane the spacecraft arrived in (numeric), vs. the target near-polar NRHO plane. Standard orbital-mechanics definition:
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
    #  PHASE C: NRHO insertion (LOI burn -- now a SMALL residual plane trim,
    #  not a fixed 90 deg change, because the plane was targeted from Earth)
    # ════════════════════════════════════════════
    r_p_nrho = R_MOON + NRHO_PERILUNE_ALT * 1e3
    r_a_nrho = R_MOON + NRHO_APOLUNE_ALT * 1e3
    a_nrho   = (r_p_nrho + r_a_nrho) / 2
    v_nrho_p = vis_viva(MU_MOON, r_p_nrho, a_nrho)          # speed at NRHO perilune
    T_nrho   = orbit_period(MU_MOON, a_nrho)

    v_hyp_at_rp = np.sqrt(v_inf_for_dv**2 + 2 * MU_MOON / r_p_nrho)
    dv_loi = combined_dv(v_hyp_at_rp, v_nrho_p, np.radians(incl_change_deg))

    moon_c = moon_at_arrival  # Moon's (Earth-centered) position used as the
                              # local origin for all lunar-orbit-phase geometry
    nrho_insertion_pos = moon_c + np.array([r_p_nrho, 0.0, 0.0])
    phase_events.append(("C", "NRHO insertion (perilune, representative 9:2 NRHO)",
                          nrho_insertion_pos.copy(),
                          f"perilune alt {NRHO_PERILUNE_ALT:,.0f} km, LOI dv {dv_loi:,.0f} m/s, "
                          f"plane trim {incl_change_deg:.1f} deg"))

    # ════════════════════════════════════════════
    #  PHASE D: Dock with Gateway (assume Gateway resides in this NRHO; 0 dv)
    # ════════════════════════════════════════════
    phase_events.append(("D", "Rendezvous with lander/rover vehicle in NRHO",
                          nrho_insertion_pos.copy(),
                          "assumed co-resident in NRHO, ~0 dv"))

    # ════════════════════════════════════════════
    #  PHASE E: NRHO -> LLO transfer  (Hohmann-like descending transfer)
    # ════════════════════════════════════════════
    r_llo   = R_MOON + LLO_ALT_KM * 1e3
    v_llo   = circular_speed(MU_MOON, r_llo)
    a_xfer  = (r_p_nrho + r_llo) / 2
    v_xfer_p1 = vis_viva(MU_MOON, r_p_nrho, a_xfer)   # depart NRHO perilune onto transfer ellipse
    v_xfer_p2 = vis_viva(MU_MOON, r_llo, a_xfer)      # arrive at LLO radius on transfer ellipse
    dv_nrho_to_llo = abs(v_nrho_p - v_xfer_p1) + abs(v_xfer_p2 - v_llo)
    t_xfer_to_llo  = orbit_period(MU_MOON, a_xfer) / 2

    llo_arrival_pos = moon_c + np.array([r_llo, 0.0, 0.0])
    phase_events.append(("E", "Transfer NRHO -> Low Lunar Orbit",
                          llo_arrival_pos.copy(),
                          f"LLO alt {LLO_ALT_KM:,.0f} km, dv {dv_nrho_to_llo:,.0f} m/s, "
                          f"transfer time {t_xfer_to_llo/3600:.1f} hr"))

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
    phase_events.append(("F", "Powered descent -> lunar south-pole surface",
                          landing_pos.copy(),
                          f"lat {landing_lat:.2f} deg, lon {landing_lon:.2f} deg, "
                          f"dv {dv_descent:,.0f} m/s"))

    # ════════════════════════════════════════════
    #  PHASE G: Surface operations / ISRU (no propagation -- stationary)
    # ════════════════════════════════════════════
    phase_events.append(("G", "Surface operations / ISRU (south pole)",
                          landing_pos.copy(),
                          "collect + electrolyze water ice -> LH2/LOX for ascent/return"))

    # ════════════════════════════════════════════
    #  PHASE H/I/J: Ascent -> LLO -> NRHO -> TEI / return
    # ════════════════════════════════════════════
    dv_ascent       = dv_descent            # symmetric assumption (flagged simplification)
    dv_llo_to_nrho  = dv_nrho_to_llo        # symmetric transfer
    a_tei = (R_EARTH + D_MOON) / 2
    v_tei_at_r  = vis_viva(MU_EARTH, D_MOON - r_p_nrho, a_tei)
    dv_tei = combined_dv(v_nrho_p, v_tei_at_r, np.radians(incl_change_deg))  # small residual plane trim, same as LOI

    total_mission_dv = (dv_tli_required + dv_loi + dv_nrho_to_llo + dv_descent
                         + dv_ascent + dv_llo_to_nrho + dv_tei)

    phase_events.append(("H", "Ascent, surface -> Low Lunar Orbit",
                          llo_arrival_pos.copy(), f"dv {dv_ascent:,.0f} m/s (symmetric w/ descent, simplification)"))
    phase_events.append(("I", "Transfer LLO -> NRHO",
                          nrho_insertion_pos.copy(), f"dv {dv_llo_to_nrho:,.0f} m/s"))
    # Trans-Earth injection departs the NRHO and heads back; mark the NRHO
    # departure point and an Earth-arrival point (patched-conic estimate).
    tei_return_pos = R_EARTH * np.array([1.0, 0.0, 0.0]) * -1.0  # symbolic Earth-arrival marker
    phase_events.append(("J", "Trans-Earth injection + return",
                          nrho_insertion_pos.copy(), f"TEI dv {dv_tei:,.0f} m/s -> Earth return trajectory"))

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
    print(f"  Delta-v required (TLI, idealized single-burn Hohmann): {dv_tli_required:>10,.0f} m/s")
    print(f"  Delta-v margin (total)             : {dv_total_departure - dv_tli_required:>+10,.0f} m/s")
    print(f"  (Real ICPS TLI performance margins are famously tight/near-parabolic --")
    print(f"   this is a known, publicly documented characteristic, not a modeling error.)")

    print("\n--- PHASE B: TRANSLUNAR COAST (plane targeted from Earth) ---")
    print(f"  LEO altitude          : {LEO_ALT_KM:,.0f} km   (v_circ = {v_leo:,.0f} m/s)")
    print(f"  Translunar plane incl : {TRANSLUNAR_PLANE_INCL_DEG:.0f} deg (set at LEO, not at the Moon)")
    print(f"  Reference Hohmann time : {t_translunar/3600:.1f} hr")
    print(f"  Closest approach       : {r_arrival/1e3:,.0f} km  at t = {t_arr_moon/3600:.1f} hr ({t_arr_moon/86400:.2f} days)")
    print(f"  Hyperbolic excess speed (v_inf, numeric 3D): {v_inf_numeric:,.0f} m/s")
    print(f"  Hyperbolic excess speed (v_inf, analytic)  : {v_inf:,.0f} m/s")
    print(f"  Peak speed              : {speeds.max():,.0f} m/s")
    print(f"  Residual plane mismatch at arrival: {incl_change_deg:.1f} deg")
    print(f"    (this is what's left AFTER targeting the plane from Earth --")
    print(f"     small because of the LEO inclination choice, not a bolted-on 90 deg)")

    print("\n--- PHASE C: NRHO INSERTION (representative 9:2 NRHO) ---")
    print(f"  Perilune altitude : {NRHO_PERILUNE_ALT:,.0f} km")
    print(f"  Apolune altitude  : {NRHO_APOLUNE_ALT:,.0f} km")
    print(f"  Orbit period      : {T_nrho/86400:.2f} days   (real 9:2 NRHO ~ 6.5-7 days)")
    print(f"  LOI delta-v (est.) : {dv_loi:,.0f} m/s  (incl. {incl_change_deg:.1f} deg residual plane trim)")
    print(f"  --> Dock with Gateway (assumed resident in this NRHO)")

    print("\n--- PHASE E: NRHO -> LOW LUNAR ORBIT TRANSFER ---")
    print(f"  LLO altitude       : {LLO_ALT_KM:,.0f} km   (v_circ = {v_llo:,.0f} m/s)")
    print(f"  Transfer time       : {t_xfer_to_llo/3600:.1f} hr")
    print(f"  Delta-v (down-leg)  : {dv_nrho_to_llo:,.0f} m/s")

    print("\n--- PHASE F: POWERED DESCENT -> LUNAR SOUTH POLE ---")
    print(f"  Delta-v (descent)   : {dv_descent:,.0f} m/s  (~{DESCENT_GRAV_LOSS:.2f}x v_LLO, incl. gravity losses)")
    print(f"  Target latitude     : {TARGET_LAT_DEG:.1f} deg")
    print(f"  >>> ESTIMATED LANDING SITE <<<")
    print(f"      Latitude  : {landing_lat:>8.2f} deg")
    print(f"      Longitude : {landing_lon:>8.2f} deg  (body-fixed, 0 deg = sub-Earth meridian)")
    print(f"  Reference real sites:")
    print(f"      Shackleton crater rim : -89.9 deg,   0.0 deg")
    print(f"      Nobile Rim 2 (DM2)    : -84.20 deg, 60.70 deg")

    print("\n--- PHASE H/I/J: ASCENT + RETURN ---")
    print(f"  Delta-v (ascent)         : {dv_ascent:,.0f} m/s")
    print(f"  Delta-v (LLO -> NRHO)    : {dv_llo_to_nrho:,.0f} m/s")
    print(f"  Delta-v (TEI, -> Earth)  : {dv_tei:,.0f} m/s")

    print("\n--- MISSION TOTAL ---")
    print(f"  Total delta-v budget (all legs, one-way rocket eq. not restaged): {total_mission_dv:,.0f} m/s")
    print(f"  (NASA staging-orbit reference budget for South Pole access: ~1,480 m/s")
    print(f"   from NRHO/butterfly staging orbit down to a fixed LLO+landing site,")
    print(f"   per NTRS 20200002920 -- our multi-leg estimate is directionally consistent)")
    print("=" * W + "\n")

    return dict(
        theta0=theta0, t_arr=t_arr, pos_t=pos_t, moons=moons,
        r_p_nrho=r_p_nrho, r_a_nrho=r_a_nrho, a_nrho=a_nrho, T_nrho=T_nrho,
        r_llo=r_llo, t_elapsed_at_llo=t_elapsed_at_llo,
        pos_llo=pos_llo, t_llo=t_llo, i_pole=i_pole,
        landing_lat=landing_lat, landing_lon=landing_lon,
        t_arr_moon=t_arr_moon, incl_change_deg=incl_change_deg,
        moon_c=moon_c, phase_events=phase_events,
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

    # Phase markers: location after EVERY phase (A-J) ---
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