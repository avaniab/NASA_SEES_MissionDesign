"""
lunar_sim.py
============
Earth → Moon trajectory simulation using astropy + scipy.

Physics
-------
  - Two-stage rocket with structural mass (astropy for unit clarity)
  - Tsiolkovsky rocket equation:  Δv = Isp · g₀ · ln(m₀ / m_dry)
  - Propellant mass fraction:     ζ  = m_prop / m₀
  - Moving Moon (circular orbit, 27.32-day period)
  - Two-body gravity: Earth + Moon (Newtonian, RK45)
  - Surface impact detection (terminal event)
  - LOI Δv estimate for 110 km lunar orbit insertion

Run:  python lunar_sim.py
Requires:  pip install numpy scipy astropy
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import astropy.units as u

# ════════════════════════════════════════════════
#  PARAMETERS  —  edit these
# ════════════════════════════════════════════════

S2_PROP_KG   = 68_019    # Stage 2 propellant mass   [kg]
S2_DRY_KG    = 13_500    # Stage 2 structural mass   [kg]
S2_ISP_S     = 421       # Specific impulse          [s]
S2_THRUST_N  = 1_000_000 # Engine thrust             [N]
PAYLOAD_KG   = 45_000    # Spacecraft (dry mass)     [kg]
MAX_DAYS     = 6         # Simulation time limit     [days]

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
T_MOON  = (27.3217   * u.day).to(u.s).value   # sidereal period [s]


# ════════════════════════════════════════════════
#  PHYSICS HELPERS
# ════════════════════════════════════════════════

def tsiolkovsky(isp, m0, m_dry):
    """Δv = Isp · g₀ · ln(m₀ / m_dry)"""
    return isp * G0 * np.log(m0 / m_dry)

def moon_pos(t, th0):
    a = th0 + 2 * np.pi * t / T_MOON
    return np.array([D_MOON * np.cos(a), D_MOON * np.sin(a)])

def moon_vel(t, th0):
    w = 2 * np.pi / T_MOON
    a = th0 + w * t
    return np.array([-D_MOON * w * np.sin(a), D_MOON * w * np.cos(a)])

def make_ode(th0, m0, m_dry, mdot, t_burn):
    """ODE closure: gravity (Earth+Moon) + prograde thrust during burn."""
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

def make_impact(th0):
    """Terminal event: crosses Moon surface."""
    def hit(t, y):
        return np.linalg.norm(y[:2] - moon_pos(t, th0)) - R_MOON
    hit.terminal = True; hit.direction = -1
    return hit

def find_phase(m0, m_dry, mdot, t_burn):
    """Optimise Moon start angle to minimise closest approach."""
    v_leo = np.sqrt(G * M_EARTH / R_EARTH)
    y0    = [-R_EARTH, 0., 0., v_leo]
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
    # ── Stage setup ──
    m0     = S2_PROP_KG + S2_DRY_KG + PAYLOAD_KG
    m_dry  = S2_DRY_KG + PAYLOAD_KG
    mdot   = S2_THRUST_N / (S2_ISP_S * G0)
    t_burn = S2_PROP_KG / mdot

    dv_avail    = tsiolkovsky(S2_ISP_S, m0, m_dry)
    zeta        = S2_PROP_KG / m0
    v_leo       = np.sqrt(G * M_EARTH / R_EARTH)
    a_t         = (R_EARTH + D_MOON) / 2
    v_tli       = np.sqrt(G * M_EARTH * (2 / R_EARTH - 1 / a_t))
    dv_required = v_tli - v_leo
    t_transit   = np.pi * np.sqrt(a_t**3 / (G * M_EARTH))

    # ── Find Moon intercept angle ──
    theta0 = find_phase(m0, m_dry, mdot, t_burn)
    ode    = make_ode(theta0, m0, m_dry, mdot, t_burn)
    impact = make_impact(theta0)

    # ── t_eval: dense during burn, coarse during coast ──
    t_burn_eval  = np.linspace(0, t_burn, 20)
    t_coast_eval = np.arange(t_burn, MAX_DAYS * 86400, 300)
    t_eval       = np.unique(np.concatenate([t_burn_eval, t_coast_eval]))

    # ── Integrate ──
    y0  = [-R_EARTH, 0., 0., v_leo]
    sol = solve_ivp(ode, [0, MAX_DAYS * 86400], y0,
                    method="RK45", max_step=200,
                    t_eval=t_eval, events=[impact])

    t_arr  = sol.t
    xs, ys = sol.y[0], sol.y[1]
    vx, vy = sol.y[2], sol.y[3]
    masses  = np.clip(m0 - mdot * np.minimum(t_arr, t_burn), m_dry, m0)
    dv_used = S2_ISP_S * G0 * np.log(m0 / masses)
    speeds  = np.hypot(vx, vy)
    moons   = np.array([moon_pos(t, theta0) for t in t_arr])
    dist_m  = np.hypot(xs - moons[:, 0], ys - moons[:, 1])

    # ── Impact analysis ──
    impacted = len(sol.t_events[0]) > 0
    if impacted:
        t_land       = sol.t_events[0][0]
        y_land       = sol.y_events[0][0]
        moon_at      = moon_pos(t_land, theta0)
        rel          = y_land[:2] - moon_at
        v_rel        = y_land[2:] - moon_vel(t_land, theta0)
        impact_speed = np.linalg.norm(v_rel)
        impact_lon   = np.degrees(np.arctan2(rel[1], rel[0]))
        nearside     = np.degrees(np.arctan2(-moon_at[1], -moon_at[0]))
        offset       = ((impact_lon - nearside) + 180) % 360 - 180
        loi_dv       = impact_speed - np.sqrt(G * M_MOON / (R_MOON + 110e3))
        region       = "near-side (Earth-facing)" if abs(offset) < 90 else "far-side"

    # ── Burn table ──
    burn_mask = t_arr <= t_burn
    t_b  = t_arr[burn_mask]
    m_b  = masses[burn_mask]
    dv_b = dv_used[burn_mask]
    rows = np.linspace(0, len(t_b) - 1, min(8, len(t_b)), dtype=int)

    # ════════════════════════════════════════════
    #  PRINT RESULTS
    # ════════════════════════════════════════════
    W = 52
    print("\n" + "=" * W)
    print("         LUNAR TRAJECTORY SIMULATION")
    print("=" * W)

    print("\n--- STAGE 2  (TLI stage, S-IVB class) ---")
    print(f"  Propellant mass  : {S2_PROP_KG:>12,.0f} kg")
    print(f"  Structural mass  : {S2_DRY_KG:>12,.0f} kg")
    print(f"  Payload mass     : {PAYLOAD_KG:>12,.0f} kg")
    print(f"  Total wet mass   : {m0:>12,.0f} kg")
    print(f"  Specific impulse : {S2_ISP_S:>12,.0f} s")
    print(f"  Thrust           : {S2_THRUST_N:>12,.0f} N")
    print(f"  Burn rate  (ṁ)   : {mdot:>12,.1f} kg/s")
    print(f"  Burn duration    : {t_burn/60:>12.1f} min")

    # Extra derived values
    mass_ratio  = m0 / m_dry
    v_exhaust   = S2_ISP_S * G0
    ke_burnout  = 0.5 * m_dry * (v_leo + dv_avail)**2
    struct_frac = S2_DRY_KG / m0
    payload_frac= PAYLOAD_KG / m0

    print("\n--- ROCKET EQUATION ---")
    print(f"  Prop. fraction ζ        : {zeta:.4f}  ({zeta*100:.1f}% of wet mass is fuel)")
    print(f"  Structural fraction     : {struct_frac:.4f}  ({struct_frac*100:.1f}% is dead structure)")
    print(f"  Payload fraction        : {payload_frac:.4f}  ({payload_frac*100:.1f}% is useful payload)")
    print(f"  Mass ratio m₀/m_dry     : {mass_ratio:.3f}x")
    print(f"  Exhaust velocity        : {v_exhaust:>10,.0f} m/s")
    print(f"  Δv available            : {dv_avail:>10,.0f} m/s")
    print(f"  Δv required (TLI)       : {dv_required:>10,.0f} m/s")
    print(f"  Δv margin               : {dv_avail - dv_required:>+10,.0f} m/s")
    print(f"  LEO circular speed      : {v_leo:>10,.0f} m/s")
    print(f"  Speed after TLI burn    : {v_leo+dv_avail:>10,.0f} m/s")
    print(f"  Earth escape velocity   : {np.sqrt(2*G*M_EARTH/R_EARTH):>10,.0f} m/s")

    print("\n--- MOVING MOON ---")
    print(f"  Orbital period   : {T_MOON/86400:.4f} days")
    print(f"  Start angle      : {np.degrees(theta0):.2f} deg")
    print(f"  Hohmann transit  : {t_transit/3600:.1f} hr  (reference)")
    print(f"  Moon travel/transit: {np.degrees(2*np.pi*t_transit/T_MOON):.1f} deg")

    print("\n--- MASS & Δv DURING BURN ---")
    print(f"  {'Time (min)':>10}  {'Mass (kg)':>12}  {'Δv used (m/s)':>14}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}")
    for i in rows:
        print(f"  {t_b[i]/60:>10.2f}  {m_b[i]:>12,.0f}  {dv_b[i]:>14,.0f}")

    # Compute distance from Earth over time
    dist_earth = np.sqrt(xs**2 + ys**2)

    print("\n--- TRAJECTORY ---")
    print(f"  Sim duration       : {t_arr[-1]/3600:.1f} hr")
    print(f"  Peak speed         : {speeds.max():>10,.0f} m/s")
    print(f"  Speed at burnout   : {speeds[burn_mask][-1]:>10,.0f} m/s")
    print(f"  Speed at impact    : {speeds[-1]:>10,.0f} m/s (Earth frame)")
    print(f"  Max dist from Earth: {dist_earth.max()/1e6:>10.1f} Mm")
    print(f"  Steps integrated   : {len(t_arr):,}")

    print("\n--- OUTCOME ---")
    if impacted:
        print(f"  ✓  IMPACT  at t = {t_land/3600:.1f} hr ({t_land/86400:.2f} days)")
        print(f"  Impact speed (Moon frame) : {impact_speed:>8,.0f} m/s")
        print(f"  Landing longitude         : {impact_lon:>8.1f} deg  (Moon-centred)")
        print(f"  Offset from near-side     : {offset:>8.1f} deg")
        print(f"  Region                    : {region}")
        print(f"  LOI Δv for 110 km orbit   : {loi_dv:>8,.0f} m/s  (retrograde)")
    else:
        i_c = dist_m.argmin()
        print(f"  ✗  NO IMPACT")
        print(f"  Closest : {dist_m[i_c]/1e3:,.0f} km at t = {t_arr[i_c]/3600:.1f} hr")
        if dv_avail < dv_required:
            print(f"  → Short on Δv by {dv_required - dv_avail:,.0f} m/s")
        else:
            print(f"  → Try increasing S2_PROP_KG slightly")

    print("=" * W + "\n")


if __name__ == "__main__":
    run()