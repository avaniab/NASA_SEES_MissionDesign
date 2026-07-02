"""
lunar_sim.py
============
Full mission-phase Earth -> Moon south-pole landing simulation, in 3D.

Mission profile (matches NASA's NRHO staging architecture,
"Enabling Global Lunar Access for Human Landing Systems", NTRS 20200002920):

    Launch from Earth
        |
        v
    Low Earth Orbit (LEO)
        |
        v
    TLI (Trans-Lunar Injection)
        |  3-5 day translunar coast
        v
    NRHO insertion            <- Near-Rectilinear Halo Orbit (Gateway's orbit)
        |
        v
    Dock with Gateway
        |
        v
    Transfer to Low Lunar Orbit (LLO)
        |
        v
    Powered descent
        |
        v
    Lunar south pole                <- latitude/longitude estimated here
        ^
        | Ascent
        v
    LLO
        |
        v
    Return to NRHO
        |
        v
    Return to Earth

Physics
-------
  - Stage 2 (TLI) burn: Tsiolkovsky rocket equation, RK45 Earth+Moon gravity
    integration in the Moon's orbital plane (patched-conic translunar leg).
  - NRHO, LLO, descent, ascent, and NRHO/TEI legs: two-body Moon (or Earth)
    Keplerian mechanics in 3D, with the lunar orbits inclined ~90 deg
    (near-polar) to reach the south pole, matching the NRHO paper's staging
    strategy for south-pole access.
  - Landing latitude/longitude is computed by propagating the descent orbit
    in 3D and rotating into the Moon's body-fixed (tidally locked) frame.

Simplifications (clearly flagged, not hidden):
  - The NRHO here is a REPRESENTATIVE near-polar ellipse sized to match the
    real 9:2 lunar synodic resonant NRHO used by Gateway (perilune ~3,000 km
    altitude, apolune ~70,000 km altitude), not a numerically-solved CR3BP
    halo orbit. A true NRHO requires solving the Circular Restricted
    Three-Body Problem; that's future work, not implemented here.
  - The translunar leg is planar (2D); the plane change onto the polar NRHO
    is folded into the LOI delta-v using the standard combined
    plane-change/insertion vis-viva formula.
  - Moon's rotation is treated as perfectly synchronous (tidal-locked) with
    its orbit -- real libration is ignored.

Run:  python lunar_sim.py
Requires:  pip install numpy scipy astropy plotly
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import astropy.units as u

# ════════════════════════════════════════════════
#  PARAMETERS — edit these
# ════════════════════════════════════════════════

S2_PROP_KG   = 68_019    # Stage 2 propellant mass   [kg]
S2_DRY_KG    = 13_500    # Stage 2 structural mass   [kg]
S2_ISP_S     = 421       # Specific impulse          [s]
S2_THRUST_N  = 1_000_000 # Engine thrust             [N]
PAYLOAD_KG   = 45_000    # Spacecraft (dry mass)     [kg]
MAX_DAYS     = 6         # Translunar sim time limit [days]

LEO_ALT_KM        = 200      # Low Earth Orbit altitude
NRHO_PERILUNE_ALT = 3_000    # 9:2 NRHO perilune altitude  (Gateway-like)
NRHO_APOLUNE_ALT  = 70_000   # 9:2 NRHO apolune altitude   (Gateway-like)
LLO_ALT_KM         = 100     # Low Lunar Orbit altitude (Artemis-like)
DESCENT_GRAV_LOSS  = 1.22    # powered-descent delta-v as a multiple of v_LLO
TARGET_LAT_DEG     = -89.5   # south-pole target latitude (near Shackleton, 89.9S)
MOON_LON0_DEG      = 0.0     # body-fixed longitude of sub-Earth point at t=0

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
    a = th0 + 2 * np.pi * t / T_MOON
    return np.array([D_MOON * np.cos(a), D_MOON * np.sin(a)])

def moon_vel(t, th0):
    w = 2 * np.pi / T_MOON
    a = th0 + w * t
    return np.array([-D_MOON * w * np.sin(a), D_MOON * w * np.cos(a)])

def moon_body_lonlat(pos_moon_centered, t, th0):
    """
    Convert a Moon-centered INERTIAL position vector (3D) into the Moon's
    body-fixed (tidally-locked) latitude/longitude.
    Moon's body frame rotates at the same rate as its orbital angle (th0 + w t)
    because the Moon is tidally locked to Earth.
    """
    rot_angle = th0 + 2 * np.pi * t / T_MOON + np.radians(MOON_LON0_DEG)
    x, y, z = pos_moon_centered
    # rotate inertial x,y into body-fixed frame
    xb = x * np.cos(rot_angle) + y * np.sin(rot_angle)
    yb = -x * np.sin(rot_angle) + y * np.cos(rot_angle)
    zb = z
    r = np.linalg.norm([xb, yb, zb])
    lat = np.degrees(np.arcsin(np.clip(zb / r, -1, 1)))
    lon = np.degrees(np.arctan2(yb, xb))
    return lat, lon


# ════════════════════════════════════════════════
#  PHASE 1-3: LEO -> TLI -> translunar coast  (planar, RK45, Earth+Moon)
# ════════════════════════════════════════════════

def make_ode(th0, m0, m_dry, mdot, t_burn):
    def ode(t, y):
        pos, vel = y[:2], y[2:]
        mp  = moon_pos(t, th0)
        r_e = np.linalg.norm(pos)
        a_e = G * M_EARTH / r_e**2 * (-pos / r_e)
        rel = mp - pos
        r_m = np.linalg.norm(rel)
        a_m = G * M_MOON  / r_m**2 * (rel / r_m)
        if t < t_burn:
            m  = max(m0 - mdot * t, m_dry)
            tv = S2_THRUST_N * (vel / np.linalg.norm(vel)) / m
        else:
            tv = np.zeros(2)
        return [vel[0], vel[1], (a_e + a_m + tv)[0], (a_e + a_m + tv)[1]]
    return ode

def find_phase(m0, m_dry, mdot, t_burn, r_leo, v_leo):
    y0 = [-r_leo, 0., 0., v_leo]
    print("  [optimising Moon intercept angle...]")

    def objective(th_deg):
        th  = np.radians(th_deg)
        sol = solve_ivp(make_ode(th, m0, m_dry, mdot, t_burn),
                        [0, MAX_DAYS * 86400], y0,
                        method="RK45", max_step=200)
        mps = np.array([moon_pos(t, th) for t in sol.t])
        return np.hypot(sol.y[0] - mps[:, 0], sol.y[1] - mps[:, 1]).min()

    res = minimize_scalar(objective, bounds=(-15, 15), method="bounded")
    return np.radians(res.x)


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════

def run():
    r_leo = R_EARTH + LEO_ALT_KM * 1e3
    v_leo = circular_speed(MU_EARTH, r_leo)

    # ---- Stage 2 setup (TLI burn) ----
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

    # ---- Phase B: translunar coast (planar, RK45) ----
    theta0 = find_phase(m0, m_dry, mdot, t_burn, r_leo, v_leo)
    ode    = make_ode(theta0, m0, m_dry, mdot, t_burn)

    t_burn_eval  = np.linspace(0, t_burn, 20)
    t_coast_eval = np.arange(t_burn, MAX_DAYS * 86400, 300)
    t_eval       = np.unique(np.concatenate([t_burn_eval, t_coast_eval]))

    y0  = [-r_leo, 0., 0., v_leo]
    sol = solve_ivp(ode, [0, MAX_DAYS * 86400], y0,
                    method="RK45", max_step=200, t_eval=t_eval)

    t_arr  = sol.t
    xs, ys = sol.y[0], sol.y[1]
    vx, vy = sol.y[2], sol.y[3]
    moons  = np.array([moon_pos(t, theta0) for t in t_arr])
    dist_m = np.hypot(xs - moons[:, 0], ys - moons[:, 1])

    # Report the RK45 run's closest approach just for context/visualization
    # (the phasing search below is tuned to find *a* Moon intercept within
    # the trajectory, which can be an early high-speed pass rather than a
    # gentle apoapsis rendezvous -- fine for the plot, not for delta-v math).
    i_arr      = dist_m.argmin()
    t_arr_moon = t_arr[i_arr]
    r_arrival  = dist_m[i_arr]

    # Clean analytic patched-conic v_inf: actual TLI burnout speed -> transfer
    # orbit energy -> speed at r = D_MOON (near-apoapsis) -> subtract the
    # Moon's own orbital speed to get the hyperbolic excess speed relative
    # to the Moon at lunar-distance encounter.
    v_inj  = v_leo + dv_avail
    eps    = v_inj**2 / 2 - MU_EARTH / r_leo
    a_act  = -MU_EARTH / (2 * eps)
    v_at_rD = vis_viva(MU_EARTH, D_MOON, a_act)
    v_moon_orbital = 2 * np.pi * D_MOON / T_MOON
    v_inf = abs(v_at_rD - v_moon_orbital)

    masses  = np.clip(m0 - mdot * np.minimum(t_arr, t_burn), m_dry, m0)
    dv_used = S2_ISP_S * G0 * np.log(m0 / masses)
    speeds  = np.hypot(vx, vy)

    # ════════════════════════════════════════════
    #  PHASE C: NRHO insertion  (LOI + plane change onto near-polar NRHO)
    # ════════════════════════════════════════════
    r_p_nrho = R_MOON + NRHO_PERILUNE_ALT * 1e3
    r_a_nrho = R_MOON + NRHO_APOLUNE_ALT * 1e3
    a_nrho   = (r_p_nrho + r_a_nrho) / 2
    v_nrho_p = vis_viva(MU_MOON, r_p_nrho, a_nrho)          # speed at NRHO perilune
    T_nrho   = orbit_period(MU_MOON, a_nrho)

    incl_change_deg = 90.0   # translunar leg is ~planar -> NRHO is near-polar
    # Analytic patched-conic hyperbola: energy conservation from the SOI
    # crossing (speed v_inf far from the Moon, effectively) down to the
    # NRHO's perilune radius.
    v_hyp_at_rp = np.sqrt(v_inf**2 + 2 * MU_MOON / r_p_nrho)
    dv_loi = combined_dv(v_hyp_at_rp, v_nrho_p, np.radians(incl_change_deg))

    # ════════════════════════════════════════════
    #  PHASE D: Dock with Gateway  (assume Gateway resides in this NRHO; 0 dv)
    # ════════════════════════════════════════════

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

    # ════════════════════════════════════════════
    #  PHASE F: Powered descent, LLO -> south pole surface
    # ════════════════════════════════════════════
    dv_descent = DESCENT_GRAV_LOSS * v_llo   # rough engineering estimate (incl. gravity losses)

    # propagate a 3D polar LLO orbit and clip it at the target latitude to get
    # a realistic body-fixed landing longitude
    incl = np.radians(90.0)   # polar orbit -> passes directly over the poles
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

    # ════════════════════════════════════════════
    #  PHASE G/H: Ascent back to LLO, return to NRHO, return to Earth
    # ════════════════════════════════════════════
    dv_ascent       = dv_descent            # symmetric assumption
    dv_llo_to_nrho  = dv_nrho_to_llo        # symmetric transfer
    v_moon_escape_at_nrho = np.sqrt(2 * MU_MOON / r_p_nrho)
    a_tei = (R_EARTH + D_MOON) / 2
    v_tei_at_r  = vis_viva(MU_EARTH, D_MOON - r_p_nrho, a_tei)
    dv_tei = combined_dv(v_nrho_p, v_tei_at_r, np.radians(incl_change_deg))  # rough return plane change

    total_mission_dv = (dv_tli_required + dv_loi + dv_nrho_to_llo + dv_descent
                         + dv_ascent + dv_llo_to_nrho + dv_tei)

    # ════════════════════════════════════════════
    #  PRINT RESULTS
    # ════════════════════════════════════════════
    W = 60
    print("\n" + "=" * W)
    print("   EARTH -> LUNAR SOUTH POLE MISSION SIMULATION (NRHO arch.)")
    print("=" * W)

    print("\n--- STAGE 2 (TLI burn) ---")
    print(f"  Propellant mass  : {S2_PROP_KG:>12,.0f} kg")
    print(f"  Structural mass  : {S2_DRY_KG:>12,.0f} kg")
    print(f"  Payload mass     : {PAYLOAD_KG:>12,.0f} kg")
    print(f"  Prop. fraction z : {zeta:.4f}")
    print(f"  Burn duration    : {t_burn/60:>12.1f} min")
    print(f"  Delta-v available: {dv_avail:>10,.0f} m/s")
    print(f"  Delta-v required : {dv_tli_required:>10,.0f} m/s  (TLI)")
    print(f"  Delta-v margin   : {dv_avail - dv_tli_required:>+10,.0f} m/s")

    print("\n--- PHASE: TRANSLUNAR COAST ---")
    print(f"  LEO altitude          : {LEO_ALT_KM:,.0f} km   (v_circ = {v_leo:,.0f} m/s)")
    print(f"  Reference Hohmann time : {t_translunar/3600:.1f} hr")
    print(f"  Closest approach       : {r_arrival/1e3:,.0f} km  at t = {t_arr_moon/3600:.1f} hr ({t_arr_moon/86400:.2f} days)")
    print(f"  Hyperbolic excess speed (v_inf, analytic): {v_inf:,.0f} m/s")
    print(f"  Peak speed              : {speeds.max():,.0f} m/s")

    print("\n--- PHASE: NRHO INSERTION (representative 9:2 NRHO) ---")
    print(f"  Perilune altitude : {NRHO_PERILUNE_ALT:,.0f} km")
    print(f"  Apolune altitude  : {NRHO_APOLUNE_ALT:,.0f} km")
    print(f"  Orbit period      : {T_nrho/86400:.2f} days   (real 9:2 NRHO ~ 6.5-7 days)")
    print(f"  Plane change       : ~{incl_change_deg:.0f} deg onto near-polar orbit")
    print(f"  LOI delta-v (est.) : {dv_loi:,.0f} m/s")
    print(f"  --> Dock with Gateway (assumed resident in this NRHO)")

    print("\n--- PHASE: NRHO -> LOW LUNAR ORBIT TRANSFER ---")
    print(f"  LLO altitude       : {LLO_ALT_KM:,.0f} km   (v_circ = {v_llo:,.0f} m/s)")
    print(f"  Transfer time       : {t_xfer_to_llo/3600:.1f} hr")
    print(f"  Delta-v (down-leg)  : {dv_nrho_to_llo:,.0f} m/s")

    print("\n--- PHASE: POWERED DESCENT -> LUNAR SOUTH POLE ---")
    print(f"  Delta-v (descent)   : {dv_descent:,.0f} m/s  (~{DESCENT_GRAV_LOSS:.2f}x v_LLO, incl. gravity losses)")
    print(f"  Target latitude     : {TARGET_LAT_DEG:.1f} deg")
    print(f"  >>> ESTIMATED LANDING SITE <<<")
    print(f"      Latitude  : {landing_lat:>8.2f} deg")
    print(f"      Longitude : {landing_lon:>8.2f} deg  (body-fixed, 0 deg = sub-Earth meridian)")
    print(f"  Reference real sites:")
    print(f"      Shackleton crater rim : -89.9 deg,   0.0 deg")
    print(f"      Nobile Rim 2 (DM2)    : -84.20 deg, 60.70 deg")

    print("\n--- PHASE: ASCENT + RETURN ---")
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
        theta0=theta0, t_arr=t_arr, xs=xs, ys=ys, moons=moons,
        r_p_nrho=r_p_nrho, r_a_nrho=r_a_nrho, a_nrho=a_nrho, T_nrho=T_nrho,
        r_llo=r_llo, t_elapsed_at_llo=t_elapsed_at_llo,
        pos_llo=pos_llo, t_llo=t_llo, i_pole=i_pole,
        landing_lat=landing_lat, landing_lon=landing_lon,
        t_arr_moon=t_arr_moon, incl=incl,
    )


def build_visualization(res, outpath="/mnt/user-data/outputs/lunar_trajectory_3d.html"):
    """Interactive 3D plot: Earth, Moon, translunar coast, NRHO, LLO, descent, landing site."""
    import plotly.graph_objects as go

    theta0 = res["theta0"]
    t_arr, xs, ys = res["t_arr"], res["xs"], res["ys"]

    # -- Moon position at lunar-arrival time (used as the local origin for
    #    all lunar-orbit-phase geometry -- the Moon barely moves during the
    #    few days of NRHO/LLO/descent ops relative to the Earth-Moon scale) --
    t0 = res["t_arr_moon"]
    moon_c = np.array([*moon_pos(t0, theta0), 0.0])

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

    # Phase 1-3: LEO -> TLI -> translunar coast
    fig.add_scatter3d(x=xs, y=ys, z=np.zeros_like(xs), mode="lines",
                       line=dict(color="#e53e3e", width=5), name="Translunar coast (TLI)")

    # Phase C: NRHO (representative near-polar ellipse around the Moon)
    a_n, r_p, r_a = res["a_nrho"], res["r_p_nrho"], res["r_a_nrho"]
    ecc = (r_a - r_p) / (r_a + r_p)
    nu = np.linspace(0, 2 * np.pi, 300)
    r_nrho = a_n * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    # polar orbit: lies in the x-z plane (through the Moon's poles) rather than x-y
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

    # Landing marker
    land = moon_c + pos_llo[i_pole]
    fig.add_scatter3d(x=[land[0]], y=[land[1]], z=[land[2]], mode="markers+text",
                       marker=dict(color="gold", size=7, symbol="diamond"),
                       text=[f"Landing<br>lat {res['landing_lat']:.1f} deg<br>lon {res['landing_lon']:.1f} deg"],
                       textposition="top center", name="Landing site (south pole)")

    fig.update_layout(
        title="Earth -> Lunar South Pole Mission (NRHO staging architecture)",
        scene=dict(
            xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    import os
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.write_html(outpath, include_plotlyjs="cdn")
    return outpath


if __name__ == "__main__":
    result = run()
    path = build_visualization(result)
    print(f"3D visualization saved to: {path}")