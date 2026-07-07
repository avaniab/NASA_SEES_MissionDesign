"""
Launch Vehicle 2: Falcon Heavy -> robotic lander/rover -> NRHO delivery.

Companion to lunar_sim.py (Launch Vehicle 1: SLS Block 1 / ICPS + Orion, crew to NRHO).
This script models the SECOND launch: an (expendable) Falcon Heavy places a ~6 t
robotic lander/rover onto a trans-lunar trajectory, the lander captures into the
SAME representative 9:2 NRHO under its OWN propulsion, and rendezvouses/docks with
Orion there. Scope ends at the NRHO dock -- the integrated descent/return is
handled in lunar_sim.py.

Addresses the two lunar_sim.py TODOs:
  * "add a second launch vehicle to carry a lander/rover"   -> Falcon Heavy (this script)
  * "NRHO dock ... more feasible than to LLO so change that" -> delivery targets NRHO,
                                                                 not low lunar orbit.

Physics helpers and physical constants are imported from lunar_sim.py so the two
mission scripts share a single source of truth (no drift between them).
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

from lunar_sim import (
    G0, R_EARTH, R_MOON, D_MOON, MU_EARTH, MU_MOON,
    LEO_ALT_KM, NRHO_PERILUNE_ALT, NRHO_APOLUNE_ALT, TRANSLUNAR_PLANE_INCL_DEG,
    MAX_DAYS,
    tsiolkovsky, vis_viva, circular_speed, orbit_period, combined_dv,
    make_ode3d, leo_state0, moon_pos3, moon_vel3, find_phase,
)

# ════════════════════════════════════════════════
#  PARAMETERS — edit these
# ════════════════════════════════════════════════
#  Launch vehicle 2: Falcon Heavy (expendable)
#     -> Merlin-1D-Vacuum upper stage does the TLI burn
#     -> robotic lander/rover flies its OWN NRHO-insertion (LOI) burn

# --- Falcon Heavy second stage (the TLI stage; single Merlin 1D Vacuum) ---
FH_S2_ISP_S       = 348        # Merlin 1D Vacuum vacuum Isp [s]                [SpaceX]
FH_S2_THRUST_N    = 981_000    # Merlin 1D Vacuum vacuum thrust [N] (~220 klbf) [SpaceX]
FH_S2_DRY_KG      = 4_000      # Falcon upper-stage dry/structural mass [kg]
FH_S2_TLI_PROP_KG = 15_500     # LOX/RP-1 REMAINING in the S2 at the parking orbit,
                               # reserved for the TLI relight. NOTE: the stage's full
                               # ~92 t propellant load is mostly consumed reaching the
                               # parking orbit; on an expendable FH deep-space flight a
                               # portion is held back to perform the trans-lunar injection.
FH_TLI_CAPACITY_KG = 18_000    # Published order-of-magnitude FH-EXPENDABLE throw mass to
                               # ~TLI energy [kg]; used only as a single-launch feasibility
                               # sanity check against the lander mass.

# --- Robotic lander/rover payload (performs its OWN NRHO insertion) ---
LANDER_WET_KG      = 6_000     # lander + rover total wet mass at TLI [kg]  (VIPER/Griffin class)
LANDER_ISP_S       = 320       # storable bipropellant main engine (MMH/NTO) Isp [s]
LANDER_LOI_PROP_KG = 1_400     # propellant the lander budgets for the NRHO-insertion burn [kg]
                               # (sized to cover the ~700 m/s LOI with margin -- the gravity-
                               #  corrected v_inf makes NRHO capture cost ~697 m/s -> ~1,195 kg)

# The residual plane mismatch at the Moon is measured from the 3D arrival
# geometry (targeted from Earth), exactly as in lunar_sim.py.
TARGET_NRHO_INCL_DEG = 90.0    # near-polar NRHO (south-pole access)


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════

def run():
    phase_events = []   # (phase, label, pos_earth_centered_xyz, note) for the HTML plot

    r_leo = R_EARTH + LEO_ALT_KM * 1e3
    v_leo = circular_speed(MU_EARTH, r_leo)
    i_plane = np.radians(TRANSLUNAR_PLANE_INCL_DEG)

    # ---- Falcon Heavy S2 as the TLI stage ----
    m0     = FH_S2_TLI_PROP_KG + FH_S2_DRY_KG + LANDER_WET_KG
    m_dry  = FH_S2_DRY_KG + LANDER_WET_KG
    mdot   = FH_S2_THRUST_N / (FH_S2_ISP_S * G0)
    t_burn = FH_S2_TLI_PROP_KG / mdot

    dv_tli_avail = tsiolkovsky(FH_S2_ISP_S, m0, m_dry)
    zeta         = FH_S2_TLI_PROP_KG / m0

    # TLI required (same idealized single-burn Hohmann reference as lunar_sim.py)
    a_t             = (R_EARTH + D_MOON) / 2
    v_tli           = vis_viva(MU_EARTH, r_leo, a_t)
    dv_tli_required = v_tli - v_leo
    t_translunar    = np.pi * np.sqrt(a_t**3 / MU_EARTH)
    tli_margin      = dv_tli_avail - dv_tli_required

    # ---- Phase A: LEO parking orbit (injection marker) ----
    pos0, vel0 = leo_state0(r_leo, v_leo, i_plane)
    phase_events.append(("A", "Falcon Heavy -> LEO parking orbit", pos0.copy(),
                          f"alt {LEO_ALT_KM:,.0f} km, v_circ {v_leo:,.0f} m/s"))

    # ---- Phase B: TLI (FH S2 relight) + translunar coast (3D, Earth+Moon) ----
    # Burn applied impulsively at the delivered dv, then coast under Earth+Moon
    # gravity only; the transfer plane is targeted from LEO (see lunar_sim.py).
    v_inj_speed = v_leo + dv_tli_avail
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

    i_arr           = dist_m.argmin()
    t_arr_moon      = t_arr[i_arr]
    r_arrival       = dist_m[i_arr]
    pos_at_arrival  = pos_t[i_arr]
    vel_at_arrival  = vel_t[i_arr]
    moon_at_arrival = moons[i_arr]

    # Actual relative state vs. the Moon at closest approach (numeric, 3D)
    r_rel = pos_at_arrival - moon_at_arrival
    v_rel = vel_at_arrival - moon_vel3(t_arr_moon, theta0)
    r_close     = np.linalg.norm(r_rel)
    v_rel_close = np.linalg.norm(v_rel)   # speed rel. to the Moon AT closest approach

    # True hyperbolic excess speed: remove the lunar gravity-well term, because
    # closest approach is inside (or near) the Moon's SOI so v_rel_close is
    # sped up relative to the speed "at infinity". Using v_rel_close directly
    # would over-state v_inf AND double-count the well at LOI (see lunar_sim.py).
    v_inf_numeric = np.sqrt(max(0.0, v_rel_close**2 - 2 * MU_MOON / r_close))
    speeds = np.linalg.norm(vel_t, axis=1)

    # Plane the lander actually arrived in vs. the target near-polar NRHO plane
    h_rel = np.cross(r_rel, v_rel)
    h_rel_unit = h_rel / np.linalg.norm(h_rel)
    arrival_incl_deg = np.degrees(np.arccos(np.clip(h_rel_unit[2], -1, 1)))
    incl_change_deg = abs(arrival_incl_deg - TARGET_NRHO_INCL_DEG)

    phase_events.append(("B", "TLI burnout + translunar coast, closest lunar approach",
                          pos_at_arrival.copy(),
                          f"v_inf {v_inf_numeric:,.0f} m/s, t+{t_arr_moon/86400:.2f} days, "
                          f"plane residual {incl_change_deg:.1f} deg (targeted from Earth)"))

    # ════════════════════════════════════════════
    #  PHASE C: NRHO insertion (LOI) -- flown by the LANDER's own propulsion
    # ════════════════════════════════════════════
    r_p_nrho = R_MOON + NRHO_PERILUNE_ALT * 1e3
    r_a_nrho = R_MOON + NRHO_APOLUNE_ALT * 1e3
    a_nrho   = (r_p_nrho + r_a_nrho) / 2
    v_nrho_p = vis_viva(MU_MOON, r_p_nrho, a_nrho)   # speed at NRHO perilune
    v_nrho_a = vis_viva(MU_MOON, r_a_nrho, a_nrho)   # speed at NRHO apolune (plane change here)
    T_nrho   = orbit_period(MU_MOON, a_nrho)

    # Refined plane change (TODO 3): in-plane capture at perilune + plane change
    # at apolune where the orbital speed is ~15x lower (see lunar_sim.py).
    di = np.radians(incl_change_deg)
    v_hyp_at_rp     = np.sqrt(v_inf_numeric**2 + 2 * MU_MOON / r_p_nrho)
    dv_loi_capture  = abs(v_hyp_at_rp - v_nrho_p)                # in-plane insertion at perilune
    dv_loi_plane    = 2 * v_nrho_a * np.sin(di / 2)             # plane change at apolune
    dv_loi_split    = dv_loi_capture + dv_loi_plane             # split strategy
    dv_loi_perilune = combined_dv(v_hyp_at_rp, v_nrho_p, di)    # single burn all at perilune
    dv_loi_required = min(dv_loi_split, dv_loi_perilune)        # cheaper of the two

    # Lander's own LOI capability (Tsiolkovsky on its LOI propellant budget)
    lander_dry_after_loi = LANDER_WET_KG - LANDER_LOI_PROP_KG
    dv_loi_avail  = tsiolkovsky(LANDER_ISP_S, LANDER_WET_KG, lander_dry_after_loi)
    loi_prop_used = LANDER_WET_KG * (1 - np.exp(-dv_loi_required / (LANDER_ISP_S * G0)))
    loi_margin    = dv_loi_avail - dv_loi_required

    moon_c = moon_at_arrival   # Moon's Earth-centered position = local origin for NRHO geometry
    nrho_insertion_pos = moon_c + np.array([r_p_nrho, 0.0, 0.0])
    phase_events.append(("C", "NRHO insertion (perilune, representative 9:2 NRHO) -- lander engine",
                          nrho_insertion_pos.copy(),
                          f"perilune alt {NRHO_PERILUNE_ALT:,.0f} km, LOI dv {dv_loi_required:,.0f} m/s, "
                          f"plane trim {incl_change_deg:.1f} deg"))

    # ════════════════════════════════════════════
    #  PHASE D: Rendezvous + dock with Orion in the NRHO
    # ════════════════════════════════════════════
    phase_events.append(("D", "Rendezvous + dock with Orion in NRHO",
                          nrho_insertion_pos.copy(),
                          "co-planar in the 9:2 NRHO; terminal rendezvous ~ small dv (not modeled)"))

    delivery_dv = dv_tli_required + dv_loi_required   # Earth-departure + lunar-capture

    # ════════════════════════════════════════════
    #  PRINT RESULTS
    # ════════════════════════════════════════════
    W = 64
    ok = lambda x: "PASS" if x >= 0 else "**FAIL**"
    print("\n" + "=" * W)
    print("   FALCON HEAVY -> ROBOTIC LANDER/ROVER -> NRHO DELIVERY")
    print("   Launch vehicle 2 (companion to SLS/Orion crew launch)")
    print("=" * W)

    print("\n--- FALCON HEAVY S2 (TLI stage; Merlin 1D Vacuum) ---")
    print(f"  S2 TLI propellant : {FH_S2_TLI_PROP_KG:>12,.0f} kg  (reserved for TLI relight)")
    print(f"  S2 structural mass: {FH_S2_DRY_KG:>12,.0f} kg")
    print(f"  Lander payload    : {LANDER_WET_KG:>12,.0f} kg")
    print(f"  Prop. fraction z  : {zeta:.4f}")
    print(f"  Burn duration     : {t_burn/60:>12.1f} min")
    print(f"  Delta-v available (TLI, S2)                : {dv_tli_avail:>10,.0f} m/s")
    print(f"  Delta-v required  (TLI, idealized Hohmann) : {dv_tli_required:>10,.0f} m/s")
    print(f"  TLI delta-v margin                          : {tli_margin:>+10,.0f} m/s   [{ok(tli_margin)}]")
    print(f"  Feasibility check : lander {LANDER_WET_KG:,.0f} kg vs FH throw-to-TLI "
          f"~{FH_TLI_CAPACITY_KG:,.0f} kg  [{ok(FH_TLI_CAPACITY_KG - LANDER_WET_KG)}]")

    print("\n--- PHASE B: TRANSLUNAR COAST (plane targeted from Earth) ---")
    print(f"  LEO altitude          : {LEO_ALT_KM:,.0f} km   (v_circ = {v_leo:,.0f} m/s)")
    print(f"  Translunar plane incl : {TRANSLUNAR_PLANE_INCL_DEG:.0f} deg (set at LEO)")
    print(f"  Reference Hohmann time : {t_translunar/3600:.1f} hr")
    print(f"  Closest approach       : {r_arrival/1e3:,.0f} km  at t = {t_arr_moon/3600:.1f} hr "
          f"({t_arr_moon/86400:.2f} days)")
    print(f"  Speed rel. to Moon at closest approach ({r_close/1e3:,.0f} km): {v_rel_close:,.0f} m/s (SOI-inflated)")
    print(f"  Hyperbolic excess speed (v_inf, gravity-corrected): {v_inf_numeric:,.0f} m/s")
    print(f"  Peak speed             : {speeds.max():,.0f} m/s")
    print(f"  Residual plane mismatch at arrival: {incl_change_deg:.1f} deg")

    print("\n--- PHASE C: NRHO INSERTION (lander's own propulsion) ---")
    print(f"  Perilune / apolune alt : {NRHO_PERILUNE_ALT:,.0f} / {NRHO_APOLUNE_ALT:,.0f} km")
    print(f"  Orbit period           : {T_nrho/86400:.2f} days   (real 9:2 NRHO ~ 6.5-7 days)")
    print(f"  LOI in-plane capture   : {dv_loi_capture:,.0f} m/s  (perilune)")
    print(f"  LOI plane change       : {dv_loi_plane:,.0f} m/s  ({incl_change_deg:.1f} deg @ apolune)")
    print(f"  LOI split / single     : {dv_loi_split:,.0f} / {dv_loi_perilune:,.0f} m/s")
    print(f"  LOI delta-v required   : {dv_loi_required:,.0f} m/s  (cheaper strategy)")
    print(f"  Lander LOI propellant  : {LANDER_LOI_PROP_KG:,.0f} kg (Isp {LANDER_ISP_S} s)")
    print(f"  Lander LOI delta-v cap.: {dv_loi_avail:,.0f} m/s")
    print(f"  LOI propellant needed  : {loi_prop_used:,.0f} kg of {LANDER_LOI_PROP_KG:,.0f} kg budgeted")
    print(f"  LOI delta-v margin     : {loi_margin:>+,.0f} m/s   [{ok(loi_margin)}]")
    print(f"  --> Dock with Orion (co-resident in this NRHO)")

    print("\n--- DELIVERY TOTAL ---")
    print(f"  Delivery delta-v (TLI + LOI, to NRHO): {delivery_dv:,.0f} m/s")
    print(f"  (This script delivers the lander to NRHO only; descent/return are")
    print(f"   handled by the integrated stack in lunar_sim.py.)")
    print("=" * W + "\n")

    return dict(
        theta0=theta0, t_arr=t_arr, pos_t=pos_t, moon_c=moon_c,
        r_p_nrho=r_p_nrho, r_a_nrho=r_a_nrho, a_nrho=a_nrho,
        phase_events=phase_events,
    )


def build_visualization(res, outpath=None):
    """Interactive 3D plot: Earth, Moon, the Falcon Heavy translunar coast, the
    representative 9:2 NRHO, and a labeled marker at the end of every phase (A-D).
    Saved next to this script by default."""
    import plotly.graph_objects as go

    if outpath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outpath = os.path.join(script_dir, "falcon_heavy_nrho_3d.html")

    pos_t  = res["pos_t"]
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

    # Phase A-B: Falcon Heavy TLI -> translunar coast (3D)
    fig.add_scatter3d(x=pos_t[:, 0], y=pos_t[:, 1], z=pos_t[:, 2], mode="lines",
                       line=dict(color="#dd6b20", width=5),
                       name="Falcon Heavy translunar coast (TLI, 3D-targeted plane)")

    # Phase C: NRHO (representative near-polar ellipse around the Moon)
    a_n, r_p, r_a = res["a_nrho"], res["r_p_nrho"], res["r_a_nrho"]
    ecc = (r_a - r_p) / (r_a + r_p)
    nu = np.linspace(0, 2 * np.pi, 300)
    r_nrho = a_n * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    x_nrho = moon_c[0] + r_nrho * np.cos(nu)
    z_nrho = r_nrho * np.sin(nu)
    y_nrho = np.full_like(nu, moon_c[1])
    fig.add_scatter3d(x=x_nrho, y=y_nrho, z=z_nrho, mode="lines",
                       line=dict(color="#805ad5", width=4),
                       name="NRHO (delivery target, dock w/ Orion)")

    # Phase markers A-D
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
                       name="Phase markers (A-D)")

    fig.update_layout(
        title="Falcon Heavy -> Robotic Lander/Rover -> NRHO Delivery (Launch Vehicle 2)",
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
