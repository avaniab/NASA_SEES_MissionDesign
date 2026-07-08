"""
mission_sim.py
==============
Class-based rebuild of the two-launch Artemis-style mission:

    FalconHeavy  -- launch vehicle 2: does TLI for the robotic Lander
    Lander       -- robotic lander/rover: own NRHO insertion, two-stage
                    descent (stays on surface) + ascent (returns to NRHO)
    SLS          -- launch vehicle 1: does TLI for Orion (crew), then
                    inserts into NRHO -- EITHER independently (traditional
                    perilune LOI) OR by combining with the Lander already
                    parked in NRHO (see combine_at_apolune below)
    Mission      -- the main class: owns the mission clock, runs FalconHeavy
                    + Lander first, then SLS + Orion, and decides which NRHO-
                    insertion strategy for Orion is actually cheaper.

Spec constants live at the TOP of each vehicle class -- edit there.

ABOUT THE "COMBINE AT APOLUNE" IDEA
------------------------------------
The request behind this method: since the Lander is already parked in NRHO,
and apolune is where an orbit's own speed is at its minimum, could Orion
save propellant by meeting the Lander there instead of doing its own
independent NRHO-insertion burn?

It's a real technique (it's the basis of genuine low-energy/weak-capture
lunar transfers) -- but it only pays off for a SLOW arrival (near-zero
v_infinity). For the kind of few-day, direct transfer this sim flies
(v_inf typically 1,000+ m/s), it's actually MORE expensive than a
traditional perilune capture, not less. Why: this NRHO is highly eccentric
(e ~ 0.88), so its perilune speed is already close to the local escape
speed -- the Moon's own gravity does most of the "matching" for free.
Apolune speed, by contrast, is very low (~90 m/s), so nearly all of the
incoming v_inf has to be cancelled by hand. SLS.combine_at_apolune() is
implemented for real and its result is compared honestly against
SLS.capture_at_perilune() every run -- Mission uses whichever is actually
cheaper, rather than assuming either one wins.

Run:  python mission_sim.py
Requires:  pip install numpy scipy astropy plotly
Output: an interactive HTML 3D plot (mission_sim_3d.html), saved next to
        this script, showing both launches and every mission phase.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import astropy.units as u

# ════════════════════════════════════════════════
#  PHYSICAL CONSTANTS
# ════════════════════════════════════════════════

G       = (6.674e-11 * u.m**3 / (u.kg * u.s**2)).value
M_EARTH = (5.972e24  * u.kg).value
M_MOON  = (7.342e22  * u.kg).value
R_EARTH = (6.371e6   * u.m).value
R_MOON  = (1.737e6   * u.m).value
D_MOON  = (3.844e8   * u.m).value
G0      = (9.80665   * u.m / u.s**2).value
T_MOON  = (27.3217   * u.day).to(u.s).value

MU_EARTH = G * M_EARTH
MU_MOON  = G * M_MOON
R_SOI    = D_MOON * (M_MOON / M_EARTH) ** 0.4

# Shared mission-architecture constants (both launches target the same NRHO)
LEO_ALT_KM               = 185
NRHO_PERILUNE_ALT_KM     = 3_000
NRHO_APOLUNE_ALT_KM      = 70_000
LLO_ALT_KM               = 100
DESCENT_GRAV_LOSS        = 1.22
TARGET_LAT_DEG           = -89.5
SURFACE_STAY_DAYS        = 6.5
MOON_LON0_DEG            = 0.0
TRANSLUNAR_PLANE_INCL_DEG = 90.0   # plane targeted from Earth, not fixed at the Moon
MAX_DAYS                 = 6       # translunar coast search window, per launch


# ════════════════════════════════════════════════
#  SHARED ORBITAL MECHANICS HELPERS
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
    """Delta-v for a burn that changes both speed and inclination at once."""
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

def make_ode3d(th0):
    """Earth + Moon point-mass gravity only; burns are applied impulsively
    before this ODE runs (see each vehicle's launch_and_coast)."""
    def ode(t, y):
        pos, vel = y[:3], y[3:]
        mp = moon_pos3(t, th0)
        r_e = np.linalg.norm(pos)
        a_e = -G * M_EARTH / r_e**3 * pos
        rel = mp - pos
        r_m = np.linalg.norm(rel)
        a_m = G * M_MOON / r_m**3 * rel
        return [*vel, *(a_e + a_m)]
    return ode

def leo_state0(r_leo, v_leo, i_plane_rad):
    """Position on the Earth-Moon injection line; velocity tilted out of
    the Moon's orbital plane by i_plane_rad -- targets the arrival PLANE
    from Earth, rather than fixing it with a burn at the Moon."""
    pos0 = np.array([-r_leo, 0.0, 0.0])
    vel0 = np.array([0.0, v_leo * np.cos(i_plane_rad), v_leo * np.sin(i_plane_rad)])
    return pos0, vel0

def find_phase(r_leo, v_inj, i_plane_rad):
    pos0, vel0 = leo_state0(r_leo, v_inj, i_plane_rad)
    y0 = [*pos0, *vel0]

    def objective(th_deg):
        th = np.radians(th_deg)
        sol = solve_ivp(make_ode3d(th), [0, MAX_DAYS * 86400], y0,
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

def translunar_coast(r_leo, v_inj_speed, i_plane_rad):
    """Run the impulsive-burn + 3D Earth/Moon coast common to BOTH launch
    vehicles. Returns arrays + the closest-approach state."""
    theta0 = find_phase(r_leo, v_inj_speed, i_plane_rad)
    ode = make_ode3d(theta0)
    pos0, vel0 = leo_state0(r_leo, v_inj_speed, i_plane_rad)
    y0 = [*pos0, *vel0]
    t_eval = np.arange(0, MAX_DAYS * 86400, 300)
    sol = solve_ivp(ode, [0, MAX_DAYS * 86400], y0, method="RK45", max_step=200, t_eval=t_eval)

    t_arr = sol.t
    pos_t = sol.y[:3].T
    vel_t = sol.y[3:].T
    moons = np.array([moon_pos3(t, theta0) for t in t_arr])
    dist_m = np.linalg.norm(pos_t - moons, axis=1)

    i_arr = dist_m.argmin()
    t_arr_moon = t_arr[i_arr]
    r_arrival = dist_m[i_arr]
    pos_at_arrival = pos_t[i_arr]
    vel_at_arrival = vel_t[i_arr]
    moon_at_arrival = moons[i_arr]

    r_rel = pos_at_arrival - moon_at_arrival
    v_rel = vel_at_arrival - moon_vel3(t_arr_moon, theta0)
    v_inf = np.linalg.norm(v_rel)

    h_rel = np.cross(r_rel, v_rel)
    h_rel_unit = h_rel / np.linalg.norm(h_rel)
    arrival_incl_deg = np.degrees(np.arccos(np.clip(h_rel_unit[2], -1, 1)))
    incl_change_deg = abs(arrival_incl_deg - 90.0)   # target: near-polar NRHO

    return dict(
        theta0=theta0, t_arr=t_arr, pos_t=pos_t, vel_t=vel_t, moons=moons,
        t_arr_moon=t_arr_moon, r_arrival=r_arrival,
        pos_at_arrival=pos_at_arrival, vel_at_arrival=vel_at_arrival,
        moon_at_arrival=moon_at_arrival, v_inf=v_inf, incl_change_deg=incl_change_deg,
    )


# ════════════════════════════════════════════════
#  VEHICLE CLASSES -- specs at the top of each
# ════════════════════════════════════════════════

class FalconHeavy:
    """Launch vehicle 2: Falcon Heavy (expendable). Does TLI for the Lander."""

    # ---- SPECS ----
    S2_ISP_S       = 348        # Merlin 1D Vacuum vacuum Isp [s]                [SpaceX]
    S2_THRUST_N    = 981_000    # Merlin 1D Vacuum vacuum thrust [N] (~220 klbf) [SpaceX]
    S2_DRY_KG      = 4_000      # Falcon upper-stage dry/structural mass [kg]
    S2_TLI_PROP_KG = 33_000     # Propellant remaining at the parking orbit for the TLI
                                 # relight [kg]. CALIBRATED via Tsiolkovsky against the
                                 # published ~18,000 kg expendable throw-to-TLI figure
                                 # (an earlier 15,500 kg guess was inconsistent with it).
    TLI_CAPACITY_KG = 18_000    # Published FH-expendable throw-to-TLI-class payload [kg]
    S1_REDIRECT_RESERVE_FRACTION = 0.05   # booster-side reserve: post-sep redirection/
                                            # disposal burn, NOT available for payload lift
    S2_GNC_RESERVE_FRACTION      = 0.10   # held out of TLI-relight propellant for RCS/
                                            # attitude-control/ullage, NOT the main engine

    def __init__(self):
        self.s2_tli_prop_usable_kg = self.S2_TLI_PROP_KG * (1 - self.S2_GNC_RESERVE_FRACTION)
        self.effective_tli_capacity_kg = self.TLI_CAPACITY_KG * (1 - self.S1_REDIRECT_RESERVE_FRACTION)

    def dv_available(self, payload_kg):
        m0 = self.s2_tli_prop_usable_kg + self.S2_DRY_KG + payload_kg
        m_dry = self.S2_DRY_KG + payload_kg
        return tsiolkovsky(self.S2_ISP_S, m0, m_dry)

    def solve_max_payload_kg(self, dv_required):
        """Bisect for the max payload the delta-v budget allows, then cap it
        against the (reserve-adjusted) mass ceiling too."""
        lo, hi = 100.0, 200_000.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if self.dv_available(mid) > dv_required:
                lo = mid
            else:
                hi = mid
        dv_limited = lo
        mass_limited = self.effective_tli_capacity_kg - self.S2_DRY_KG
        binding = "S2 delta-v" if dv_limited < mass_limited else "FH mass ceiling"
        return min(dv_limited, mass_limited), binding

    def launch_and_coast(self, payload_kg, r_leo, v_leo, i_plane_rad):
        dv_avail = self.dv_available(payload_kg)
        v_inj_speed = v_leo + dv_avail
        coast = translunar_coast(r_leo, v_inj_speed, i_plane_rad)
        coast["dv_avail"] = dv_avail
        return coast


class Lander:
    """Robotic lander/rover, delivered by FalconHeavy. Performs its OWN NRHO
    insertion (LOI), then a two-stage descent (descent stage stays on the
    surface) + ascent (ascent stage alone returns to NRHO). wet_kg is set
    by Mission after asking FalconHeavy how much it can actually deliver."""

    # ---- SPECS ----
    ISP_S = 320                          # storable bipropellant (MMH/NTO) main engine
    LOI_PROP_MARGIN_FRACTION = 0.10      # propellant margin on top of the bare LOI requirement
    ASCENT_MASS_FRACTION_OF_LANDED = 0.50  # tunable split: ascent stage vs. permanent surface mass

    def __init__(self, wet_kg):
        self.wet_kg = wet_kg
        self.dry_kg_after_loi = None
        self.nrho = None   # dict of orbital elements, filled by insert_into_nrho()

    def insert_into_nrho(self, v_inf, r_p_nrho, r_a_nrho, incl_change_deg):
        a_nrho = (r_p_nrho + r_a_nrho) / 2
        v_nrho_p = vis_viva(MU_MOON, r_p_nrho, a_nrho)
        v_nrho_a = vis_viva(MU_MOON, r_a_nrho, a_nrho)
        T_nrho = orbit_period(MU_MOON, a_nrho)

        v_hyp_at_rp = np.sqrt(v_inf**2 + 2 * MU_MOON / r_p_nrho)
        di = np.radians(incl_change_deg)
        dv_loi_capture = abs(v_hyp_at_rp - v_nrho_p)
        v_nrho_a_forplane = v_nrho_a
        dv_loi_plane = 2 * v_nrho_a_forplane * np.sin(di / 2)
        dv_loi_split = dv_loi_capture + dv_loi_plane
        dv_loi_perilune = combined_dv(v_hyp_at_rp, v_nrho_p, di)
        dv_loi_required = min(dv_loi_split, dv_loi_perilune)

        dv_w_margin = dv_loi_required * (1 + self.LOI_PROP_MARGIN_FRACTION)
        prop_needed = self.wet_kg * (1 - np.exp(-dv_w_margin / (self.ISP_S * G0)))
        self.dry_kg_after_loi = self.wet_kg - prop_needed

        self.nrho = dict(r_p=r_p_nrho, r_a=r_a_nrho, a=a_nrho, v_p=v_nrho_p,
                          v_a=v_nrho_a, T=T_nrho, incl_change_deg=incl_change_deg)
        return dict(dv_loi_required=dv_loi_required, prop_needed=prop_needed,
                    dry_kg_after_loi=self.dry_kg_after_loi)

    def descend_ascend_two_stage(self):
        """NRHO -> LLO -> surface (descent stage discarded) -> LLO -> NRHO
        (ascent stage alone). Uses the wet mass remaining after LOI."""
        nrho = self.nrho
        r_llo = R_MOON + LLO_ALT_KM * 1e3
        v_llo = circular_speed(MU_MOON, r_llo)
        a_xfer = (nrho["r_p"] + r_llo) / 2
        v_xfer_p1 = vis_viva(MU_MOON, nrho["r_p"], a_xfer)
        v_xfer_p2 = vis_viva(MU_MOON, r_llo, a_xfer)
        dv_nrho_to_llo = abs(nrho["v_p"] - v_xfer_p1) + abs(v_xfer_p2 - v_llo)
        dv_descent = DESCENT_GRAV_LOSS * v_llo
        dv_ascent = dv_descent
        dv_llo_to_nrho = dv_nrho_to_llo
        t_xfer = orbit_period(MU_MOON, a_xfer) / 2

        dv_descent_segment = dv_nrho_to_llo + dv_descent   # LOI already spent separately
        wet_at_nrho = self.dry_kg_after_loi
        mass_landed = wet_at_nrho * np.exp(-dv_descent_segment / (self.ISP_S * G0))

        ascent_wet_kg = mass_landed * self.ASCENT_MASS_FRACTION_OF_LANDED
        permanent_surface_mass_kg = mass_landed - ascent_wet_kg

        dv_ascent_segment = dv_ascent + dv_llo_to_nrho
        ascent_prop_needed = ascent_wet_kg * (1 - np.exp(-dv_ascent_segment / (self.ISP_S * G0)))
        ascent_dry_at_redock = ascent_wet_kg - ascent_prop_needed

        return dict(
            v_llo=v_llo, dv_nrho_to_llo=dv_nrho_to_llo, dv_descent=dv_descent,
            dv_ascent=dv_ascent, dv_llo_to_nrho=dv_llo_to_nrho, t_xfer=t_xfer,
            mass_landed=mass_landed, ascent_wet_kg=ascent_wet_kg,
            permanent_surface_mass_kg=permanent_surface_mass_kg,
            ascent_prop_needed=ascent_prop_needed, ascent_dry_at_redock=ascent_dry_at_redock,
        )


class SLS:
    """Launch vehicle 1: SLS Block 1 (ICPS) + Orion (crew). Does TLI, then
    inserts into NRHO either independently (capture_at_perilune) or by
    combining with the Lander already parked there (combine_at_apolune)."""

    # ---- SPECS ----
    S2_PROP_KG   = 29_000    # ICPS propellant mass [kg]
    S2_DRY_KG    = 3_720     # ICPS dry/structural mass [kg]
    S2_ISP_S     = 462       # RL10B-2 vacuum Isp [s]
    S2_THRUST_N  = 110_100   # RL10B-2 vacuum thrust [N]
    PAYLOAD_KG   = 25_850    # Orion CM+ESM total mass at TLI [kg]
    ORION_SM_ASSIST_DV_MS = 50   # small trans-lunar trim from Orion's own ESM, on top of ICPS
    ORION_ESM_PROP_KG = 8_600    # ESM usable propellant [kg]
    ORION_ESM_ISP_S   = 319      # OMS-E/AJ10-class engine Isp [s]
    NRHO_INSERTION_DV_REF_MS = 430   # published reference (Whitley & Martinez), traditional
                                      # perilune LOI -- used as a cross-check on capture_at_perilune()
    NRHO_TEI_DV_REF_MS       = 410   # published reference, NRHO -> Earth interface
    DV_MARGIN_FRACTION = 0.08   # 8% reserve on required delta-v (not on what's available)

    def __init__(self):
        self.m0 = self.S2_PROP_KG + self.S2_DRY_KG + self.PAYLOAD_KG
        self.m_dry = self.S2_DRY_KG + self.PAYLOAD_KG
        self.dv_tli_avail_icps = tsiolkovsky(self.S2_ISP_S, self.m0, self.m_dry)

    def launch_and_coast(self, r_leo, v_leo, i_plane_rad):
        dv_total = self.dv_tli_avail_icps + self.ORION_SM_ASSIST_DV_MS
        v_inj_speed = v_leo + dv_total
        coast = translunar_coast(r_leo, v_inj_speed, i_plane_rad)
        coast["dv_total_departure"] = dv_total
        return coast

    def capture_at_perilune(self, v_inf, r_p_nrho, r_a_nrho, incl_change_deg):
        """Traditional independent LOI at perilune -- Orion captures on its
        own, without using the Lander's state at all."""
        a_nrho = (r_p_nrho + r_a_nrho) / 2
        v_nrho_p = vis_viva(MU_MOON, r_p_nrho, a_nrho)
        v_hyp_at_rp = np.sqrt(v_inf**2 + 2 * MU_MOON / r_p_nrho)
        dv_own_estimate = combined_dv(v_hyp_at_rp, v_nrho_p, np.radians(incl_change_deg))
        # what's actually charged: the published reference (our own 2-body
        # estimate runs high -- see docstring / lunar_sim.py TODO on 3-body dynamics)
        return dict(dv=self.NRHO_INSERTION_DV_REF_MS, dv_own_estimate=dv_own_estimate)

    def combine_at_apolune(self, coast, r_a_nrho, lander_v_nrho_a):
        """Find where Orion's incoming trajectory crosses the Lander's NRHO
        apolune radius, then compute the delta-v to match the Lander's
        (already-established, slow) apolune velocity there. Real technique,
        but only cheap for near-zero v_inf arrivals -- see module docstring."""
        pos_t, vel_t, t_arr = coast["pos_t"], coast["vel_t"], coast["t_arr"]
        moons, theta0 = coast["moons"], coast["theta0"]
        dist_m = np.linalg.norm(pos_t - moons, axis=1)

        below = np.where(dist_m <= r_a_nrho)[0]
        if len(below) == 0:
            return dict(dv=np.inf, feasible=False, reason="trajectory never reaches r_a_nrho")
        i_ra = below[0]   # first crossing inbound

        r_rel = pos_t[i_ra] - moons[i_ra]
        v_rel = vel_t[i_ra] - moon_vel3(t_arr[i_ra], theta0)
        v_hyp_at_ra = np.linalg.norm(v_rel)

        h_rel = np.cross(r_rel, v_rel)
        h_rel_unit = h_rel / np.linalg.norm(h_rel)
        arrival_incl_deg = np.degrees(np.arccos(np.clip(h_rel_unit[2], -1, 1)))
        incl_change_deg = abs(arrival_incl_deg - 90.0)

        dv = combined_dv(v_hyp_at_ra, lander_v_nrho_a, np.radians(incl_change_deg))
        return dict(dv=dv, feasible=True, v_hyp_at_ra=v_hyp_at_ra,
                    incl_change_deg=incl_change_deg, t_at_ra=t_arr[i_ra])

    def trans_earth_injection(self):
        return self.NRHO_TEI_DV_REF_MS

    def orion_esm_dv_available(self):
        return tsiolkovsky(self.ORION_ESM_ISP_S, self.PAYLOAD_KG,
                           self.PAYLOAD_KG - self.ORION_ESM_PROP_KG)


# ════════════════════════════════════════════════
#  MISSION -- the main class: owns the clock, runs both launches
# ════════════════════════════════════════════════

class Mission:
    LEO_ALT_KM = LEO_ALT_KM
    SLS_LAUNCH_BUFFER_DAYS = 1.0   # SLS launches this many days after the Lander reaches NRHO

    def __init__(self):
        self.fh = FalconHeavy()
        self.lander = None
        self.sls = SLS()
        self.phase_events = []   # (phase, label, pos_xyz, note, day)
        self.log = []            # printed report lines, built up as we go

    def _p(self, line=""):
        self.log.append(line)

    def run(self):
        r_leo = R_EARTH + self.LEO_ALT_KM * 1e3
        v_leo = circular_speed(MU_EARTH, r_leo)
        i_plane = np.radians(TRANSLUNAR_PLANE_INCL_DEG)
        r_p_nrho = R_MOON + NRHO_PERILUNE_ALT_KM * 1e3
        r_a_nrho = R_MOON + NRHO_APOLUNE_ALT_KM * 1e3

        # ══════════════ LEG 1: Falcon Heavy + Lander -> NRHO ══════════════
        self._p("=" * 70)
        self._p("LEG 1: FALCON HEAVY + LANDER -> NRHO")
        self._p("=" * 70)

        a_t = (R_EARTH + D_MOON) / 2
        v_tli = vis_viva(MU_EARTH, r_leo, a_t)
        dv_tli_required = v_tli - v_leo

        max_payload_kg, binding = self.fh.solve_max_payload_kg(dv_tli_required)
        lander_wet_kg = max_payload_kg * 0.90   # 10% design margin off the theoretical ceiling
        self.lander = Lander(lander_wet_kg)
        self._p(f"Falcon Heavy max deliverable payload: {max_payload_kg:,.0f} kg "
                f"(binding constraint: {binding})")
        self._p(f"Lander wet mass (10% design margin applied): {lander_wet_kg:,.0f} kg")

        fh_coast = self.fh.launch_and_coast(lander_wet_kg, r_leo, v_leo, i_plane)
        self.phase_events.append(("FH-A", "Falcon Heavy launch, LEO", -leo_state0(r_leo, v_leo, i_plane)[0],
                                  f"v_circ {v_leo:,.0f} m/s", 0.0))
        day_lander_arrival = fh_coast["t_arr_moon"] / 86400
        self.phase_events.append(("FH-B", "Lander TLI burnout + closest lunar approach",
                                  fh_coast["pos_at_arrival"],
                                  f"v_inf {fh_coast['v_inf']:,.0f} m/s, plane residual "
                                  f"{fh_coast['incl_change_deg']:.1f} deg", day_lander_arrival))
        self._p(f"Falcon Heavy TLI dv: {fh_coast['dv_avail']:,.0f} m/s")
        self._p(f"Lander v_inf at arrival: {fh_coast['v_inf']:,.0f} m/s, "
                f"day {day_lander_arrival:.2f}")

        loi = self.lander.insert_into_nrho(fh_coast["v_inf"], r_p_nrho, r_a_nrho,
                                            fh_coast["incl_change_deg"])
        moon_c = fh_coast["moon_at_arrival"]
        nrho_pos = moon_c + np.array([r_p_nrho, 0.0, 0.0])
        day_lander_nrho = day_lander_arrival
        self.phase_events.append(("FH-C", "Lander NRHO insertion (own engine)", nrho_pos,
                                  f"LOI dv {loi['dv_loi_required']:,.0f} m/s, "
                                  f"dry mass after LOI {loi['dry_kg_after_loi']:,.0f} kg",
                                  day_lander_nrho))
        self._p(f"Lander LOI dv: {loi['dv_loi_required']:,.0f} m/s -> "
                f"dry mass in NRHO: {loi['dry_kg_after_loi']:,.0f} kg")

        roundtrip = self.lander.descend_ascend_two_stage()
        self._p(f"Lander two-stage round trip: landed mass {roundtrip['mass_landed']:,.0f} kg, "
                f"ascent stage {roundtrip['ascent_wet_kg']:,.0f} kg wet -> "
                f"{roundtrip['ascent_dry_at_redock']:,.0f} kg dry redocking with Orion")
        self._p(f"Permanent surface mass (rover/ISRU + discarded descent structure): "
                f"{roundtrip['permanent_surface_mass_kg']:,.0f} kg")

        # ══════════════ LEG 2: SLS + Orion -> NRHO (using the Lander) ══════════════
        self._p("")
        self._p("=" * 70)
        self._p("LEG 2: SLS + ORION -> NRHO")
        self._p("=" * 70)

        day_sls_launch = day_lander_nrho + self.SLS_LAUNCH_BUFFER_DAYS
        self._p(f"SLS launches on mission day {day_sls_launch:.2f} "
                f"(Lander's NRHO arrival + {self.SLS_LAUNCH_BUFFER_DAYS:.1f} day buffer)")

        sls_coast = self.sls.launch_and_coast(r_leo, v_leo, i_plane)
        day_orion_arrival = day_sls_launch + sls_coast["t_arr_moon"] / 86400
        self.phase_events.append(("SLS-A", "SLS launch, LEO", -leo_state0(r_leo, v_leo, i_plane)[0],
                                  f"v_circ {v_leo:,.0f} m/s", day_sls_launch))
        self.phase_events.append(("SLS-B", "Orion TLI burnout + closest lunar approach",
                                  sls_coast["pos_at_arrival"],
                                  f"v_inf {sls_coast['v_inf']:,.0f} m/s, plane residual "
                                  f"{sls_coast['incl_change_deg']:.1f} deg", day_orion_arrival))
        self._p(f"SLS/Orion TLI dv: {sls_coast['dv_total_departure']:,.0f} m/s")
        self._p(f"Orion v_inf at arrival: {sls_coast['v_inf']:,.0f} m/s, day {day_orion_arrival:.2f}")

        # ---- Compare the two NRHO-insertion strategies ----
        perilune = self.sls.capture_at_perilune(sls_coast["v_inf"], r_p_nrho, r_a_nrho,
                                                 sls_coast["incl_change_deg"])
        apolune = self.sls.combine_at_apolune(sls_coast, r_a_nrho, self.lander.nrho["v_a"])

        self._p("")
        self._p("--- NRHO INSERTION STRATEGY COMPARISON (for Orion) ---")
        self._p(f"  Traditional independent capture at PERILUNE : {perilune['dv']:,.0f} m/s "
                f"(published ref.; our own estimate {perilune['dv_own_estimate']:,.0f} m/s)")
        if apolune["feasible"]:
            self._p(f"  Combine w/ Lander at APOLUNE (this Lander's {self.lander.nrho['v_a']:,.0f} m/s): "
                    f"{apolune['dv']:,.0f} m/s")
        else:
            self._p(f"  Combine w/ Lander at APOLUNE: infeasible ({apolune['reason']})")

        if apolune["feasible"] and apolune["dv"] < perilune["dv"]:
            chosen_dv = apolune["dv"]
            chosen_strategy = "combine at apolune"
        else:
            chosen_dv = perilune["dv"]
            chosen_strategy = "traditional perilune capture"
        self._p(f"  >>> CHEAPER STRATEGY: {chosen_strategy} ({chosen_dv:,.0f} m/s) <<<")
        if chosen_strategy == "traditional perilune capture":
            self._p(f"  (Apolune combine costs MORE here because this NRHO is highly eccentric --")
            self._p(f"   its perilune speed is already close to the local escape speed, so the Moon's")
            self._p(f"   own gravity does most of the capture 'for free' there. Apolune's target speed")
            self._p(f"   is very low, so nearly all of Orion's {sls_coast['v_inf']:,.0f} m/s v_inf would")
            self._p(f"   have to be killed by hand instead. Apolune-combine only wins for a much slower,")
            self._p(f"   near-ballistic arrival -- a genuinely different, longer translunar trajectory.)")

        moon_c_sls = sls_coast["moon_at_arrival"]
        nrho_pos_sls = moon_c_sls + np.array([r_p_nrho, 0.0, 0.0])
        self.phase_events.append(("SLS-C", f"Orion NRHO insertion ({chosen_strategy})", nrho_pos_sls,
                                  f"dv {chosen_dv:,.0f} m/s", day_orion_arrival))
        self.phase_events.append(("SLS-D", "Rendezvous with Lander in NRHO", nrho_pos_sls,
                                  "2 of 4 crew transfer to the lander; 2 remain aboard Orion",
                                  day_orion_arrival))

        # ---- Orion's own feasibility check (no refueling) ----
        dv_tei = self.sls.trans_earth_injection()
        orion_own_dv_required = chosen_dv + dv_tei
        orion_own_dv_w_margin = orion_own_dv_required * (1 + self.sls.DV_MARGIN_FRACTION)
        orion_esm_dv_avail = self.sls.orion_esm_dv_available()
        margin_nominal = orion_esm_dv_avail - orion_own_dv_required
        margin_w_reserve = orion_esm_dv_avail - orion_own_dv_w_margin

        self._p("")
        self._p("--- ORION FEASIBILITY (no refueling) ---")
        self._p(f"  Required (NRHO insertion + TEI, nominal) : {orion_own_dv_required:,.0f} m/s")
        self._p(f"  Required (+{self.sls.DV_MARGIN_FRACTION*100:.0f}% reserve)                  : {orion_own_dv_w_margin:,.0f} m/s")
        self._p(f"  Orion ESM delta-v capability              : {orion_esm_dv_avail:,.0f} m/s")
        self._p(f"  Margin (nominal / with reserve)           : {margin_nominal:>+,.0f} / {margin_w_reserve:>+,.0f} m/s "
                f"-> {'FEASIBLE' if margin_w_reserve > 0 else 'INFEASIBLE with reserve'}")

        self.results = dict(
            fh_coast=fh_coast, sls_coast=sls_coast, moon_c=moon_c, moon_c_sls=moon_c_sls,
            r_p_nrho=r_p_nrho, r_a_nrho=r_a_nrho, lander=self.lander,
            perilune=perilune, apolune=apolune, chosen_dv=chosen_dv, chosen_strategy=chosen_strategy,
            roundtrip=roundtrip, day_lander_nrho=day_lander_nrho, day_orion_arrival=day_orion_arrival,
        )
        return self.results

    def print_report(self):
        print("\n".join(self.log))

    def build_visualization(self, outpath=None):
        import plotly.graph_objects as go

        if outpath is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            outpath = os.path.join(script_dir, "mission_sim_3d.html")

        res = self.results
        moon_c = res["moon_c"]

        def sph(r, n=40):
            u_ = np.linspace(0, 2 * np.pi, n)
            v_ = np.linspace(0, np.pi, n)
            x = r * np.outer(np.cos(u_), np.sin(v_))
            y = r * np.outer(np.sin(u_), np.sin(v_))
            z = r * np.outer(np.ones_like(u_), np.cos(v_))
            return x, y, z

        fig = go.Figure()

        ex, ey, ez = sph(R_EARTH * 3)
        fig.add_surface(x=ex, y=ey, z=ez, colorscale=[[0, "#2b6cb0"], [1, "#2b6cb0"]],
                        showscale=False, name="Earth", opacity=1)

        mx, my, mz = sph(R_MOON * 3)
        fig.add_surface(x=mx + moon_c[0], y=my + moon_c[1], z=mz + moon_c[2],
                        colorscale=[[0, "#a0a0a0"], [1, "#a0a0a0"]], showscale=False,
                        name="Moon", opacity=1)

        ang = np.linspace(0, 2 * np.pi, 200)
        fig.add_scatter3d(x=D_MOON * np.cos(ang), y=D_MOON * np.sin(ang), z=np.zeros_like(ang),
                          mode="lines", line=dict(color="lightgray", width=2, dash="dot"),
                          name="Moon's orbit")

        fh_pos = res["fh_coast"]["pos_t"]
        fig.add_scatter3d(x=fh_pos[:, 0], y=fh_pos[:, 1], z=fh_pos[:, 2], mode="lines",
                          line=dict(color="#dd6b20", width=5), name="Falcon Heavy + Lander coast")

        sls_pos = res["sls_coast"]["pos_t"]
        fig.add_scatter3d(x=sls_pos[:, 0], y=sls_pos[:, 1], z=sls_pos[:, 2], mode="lines",
                          line=dict(color="#e53e3e", width=5), name="SLS + Orion coast")

        r_p, r_a = res["r_p_nrho"], res["r_a_nrho"]
        a_n = (r_p + r_a) / 2
        ecc = (r_a - r_p) / (r_a + r_p)
        nu = np.linspace(0, 2 * np.pi, 300)
        r_nrho = a_n * (1 - ecc**2) / (1 + ecc * np.cos(nu))
        fig.add_scatter3d(x=moon_c[0] + r_nrho * np.cos(nu), y=np.full_like(nu, moon_c[1]),
                          z=r_nrho * np.sin(nu), mode="lines",
                          line=dict(color="#805ad5", width=4), name="NRHO (Lander + Orion)")

        phase_events = self.phase_events
        px = [p[2][0] for p in phase_events]
        py = [p[2][1] for p in phase_events]
        pz = [p[2][2] for p in phase_events]
        labels = [f"{p[0]}: {p[1]}  [Day {p[4]:.2f}]<br>{p[3]}" for p in phase_events]
        short = [p[0] for p in phase_events]
        fig.add_scatter3d(x=px, y=py, z=pz, mode="markers+text",
                          marker=dict(color="gold", size=6, symbol="diamond",
                                      line=dict(color="black", width=1)),
                          text=short, textposition="top center",
                          hovertext=labels, hoverinfo="text", name="Phase markers")

        fig.update_layout(
            title="Two-Launch Artemis-Style Mission: Falcon Heavy+Lander & SLS+Orion -> NRHO",
            scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]", aspectmode="data"),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.write_html(outpath, include_plotlyjs="cdn")
        return outpath


if __name__ == "__main__":
    mission = Mission()
    mission.run()
    mission.print_report()
    path = mission.build_visualization()
    print(f"\n3D visualization saved to: {path}")