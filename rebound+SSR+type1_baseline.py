#!/usr/bin/env python

#SBATCH --job-name=planetesimal-0326
#SBATCH --output=logs/planetesimal-0326_%j.out
#SBATCH --error=logs/planetesimal-0326_%j.err
#SBATCH --account=pi-abbot
#SBATCH --nodes=1
#SBATCH --exclusive


###notes 3/26
# compared to 3/24
# dens0=1700g/cm^2
# f_e=1
# data directory changed
# revised index for disk_potential
# tend=2e7
# add if len(sim.particles)>1


import functools
import multiprocessing
import os

import rebound
import numpy as np
import math


data_dir = '/project2/abbot/xuanji/5.SSR/data-032624/'
MJUP_TO_MSUN = 9.545e-4


@functools.lru_cache(maxsize=64)
def z_k_exact(disk_k, n_terms=100000):
    """Exact local-disk force coefficient z_k for Sigma ~ r^-k.

    The full razor-thin power-law disk exerts F = -2 pi G Sigma(r) z_k/(2-k)
    with z_k = 1 + (2-k)(k-1) sum_l (4l+1) A_l / ((2l-1+k)(2l+2-k)).
    Verified against direct integration (z_1 = 1 exactly, the Mestel disk;
    z_1.5 = 1.0942, matching the legacy hard-coded 1.094).
    """
    total = 0.0
    a_l = 1.0
    for ell in range(1, n_terms + 1):
        a_l *= (2.0 * ell - 1.0) ** 2 / (4.0 * ell * ell)
        total += (4.0 * ell + 1.0) * a_l / ((2.0 * ell - 1.0 + disk_k) * (2.0 * ell + 2.0 - disk_k))
    return 1.0 + (2.0 - disk_k) * (disk_k - 1.0) * total


# ### Parameters Input
def define_variables(var):
    params = {
        "dens0": 1700.0 * 1.125e-7, # in unit of m_sun/au^2
        "di": 1.5,
        "h0": 0.025,
        "hi": 1.25,
        # None -> compute the exact z_k for the disk index di (recommended);
        # a number overrides it (legacy value was 1.094, exact for di=1.5).
        "zk": None,
        "tdep": 1000000.0,
        # 0.0 disables the inner-disk cutoff (disk extends to r=0); a
        # positive value truncates the disk inside that radius.
        "disk_inner_cutoff": 0.0,
        "k_gap_dep": 0.0,
        "n_terms": 30,
        # Speed of light in the REBOUND units (AU/yr).
        "c_code": 63241.077,
        "include_gr": 1,
        "use_gap": 1,
        "f_e": 1.0,
        "f_a": 1.0,
    }
    if isinstance(var, dict):
        params.update(var)
    dens0 = params["dens0"]
    di = params["di"]
    h0 = params["h0"]
    hi = params["hi"]
    zk = params["zk"]
    if zk is None:
        zk = z_k_exact(di)
    tdep = params["tdep"]
    disk_inner_cutoff = params["disk_inner_cutoff"]
    k_gap_dep = params["k_gap_dep"]
    n_terms = params["n_terms"]
    c_code = params["c_code"]
    include_gr = params["include_gr"]
    use_gap = params["use_gap"]
    f_e = params["f_e"]
    f_a = params["f_a"]
    return (
        dens0, di, h0, hi, zk, tdep, disk_inner_cutoff, k_gap_dep,
        n_terms, c_code, include_gr, use_gap, f_e, f_a,
    )


def a_l_coefficients(n_terms):
    coeffs = np.zeros(n_terms)
    for idx in range(n_terms):
        ell = idx + 1
        coeffs[idx] = (math.factorial(2 * ell) / 2.0 ** (2 * ell) / math.factorial(ell) ** 2) ** 2
    return coeffs


def sigma_disk(dens0, radius, disk_k, time, tdep):
    return dens0 * pow(radius, -disk_k) * np.exp(-time / tdep)


def full_disk_acceleration(G, dens, zk, disk_k):
    if abs(2.0 - disk_k) < 1.0e-10:
        raise ValueError("The local radial-force prescription is singular for disk_k = 2.")
    return -2.0 * math.pi * G * dens * zk / (2.0 - disk_k)


def exterior_disk_acceleration(radius, edge, G, dens, disk_k, coeffs):
    # F = +2 pi G Sigma(r) sum_{l>=1} 2l A_l/(2l-1+k) (r/edge)^{2l-1+k},
    # verified against direct integration of the razor-thin disk (2026-08-23);
    # the l=0 term vanishes.
    if edge <= 0.0:
        return 0.0
    edge_sum = 0.0
    for ell, coeff in enumerate(coeffs, start=1):
        edge_sum += (
            2 * ell
            * coeff
            * pow(radius / edge, 2 * ell - 1 + disk_k)
            / (2 * ell - 1 + disk_k)
        )
    return 2.0 * math.pi * G * dens * edge_sum


def inner_disk_acceleration(radius, edge, G, dens, disk_k, coeffs):
    # F = -2 pi G Sigma(r) sum_{l>=0} (2l+1) A_l/(2l+2-k) (edge/r)^{2l+2-k},
    # verified against direct integration (2026-08-23).  The l=0 term
    # (A_0 = 1) is the interior disk's monopole, exactly -G M_in/r^2.
    if edge <= 0.0 or radius <= edge:
        return 0.0
    edge_sum = pow(edge / radius, 2.0 - disk_k) / (2.0 - disk_k)  # l = 0
    for ell, coeff in enumerate(coeffs, start=1):
        edge_sum += (
            (2 * ell + 1)
            * coeff
            * pow(edge / radius, 2 * ell + 2 - disk_k)
            / (2 * ell + 2 - disk_k)
        )
    return -2.0 * math.pi * G * dens * edge_sum


def disk_acceleration_no_gap(radius, G, dens0, disk_k, zk, t, tdep, disk_inner_cutoff, coeffs):
    dens = sigma_disk(dens0, radius, disk_k, t, tdep)
    if disk_inner_cutoff > 0.0 and radius <= disk_inner_cutoff:
        return exterior_disk_acceleration(radius, disk_inner_cutoff, G, dens, disk_k, coeffs)

    accel = full_disk_acceleration(G, dens, zk, disk_k)
    if disk_inner_cutoff > 0.0:
        accel -= inner_disk_acceleration(radius, disk_inner_cutoff, G, dens, disk_k, coeffs)
    return accel


def reference_gap_edges(a_giant, e_giant, giant_mass, stellar_mass):
    """Fixed one-Hill-radius gap from ref_code.txt::cal_gap()."""
    hill_ratio = (giant_mass / (3.0 * stellar_mass)) ** (1.0 / 3.0)
    if e_giant > hill_ratio:
        return (
            a_giant * (1.0 - e_giant) * (1.0 - hill_ratio),
            a_giant * (1.0 + e_giant) * (1.0 + hill_ratio),
        )
    return a_giant * (1.0 - hill_ratio), a_giant * (1.0 + hill_ratio)


def disk_acceleration_multi_hr_inner(
    radius,
    r_in,
    G,
    dens0,
    disk_k,
    zk,
    t,
    tdep,
    k_gap_dep,
    coeffs,
):
    """Multi_HR5183 R_in-corrected disk force for an inner embryo."""
    dens = sigma_disk(dens0, radius, disk_k, t, tdep)
    edge_sum = 0.0
    for ell, coeff in enumerate(coeffs[:9], start=1):
        edge_sum += (
            ell
            * coeff
            * pow(radius / r_in, 2 * ell - 1 + disk_k)
            / (2 * ell - 1 + disk_k)
        )

    full_disk = -4.0 * math.pi * G * dens * zk
    empty_gap = -4.0 * math.pi * G * dens * (zk + edge_sum)
    return empty_gap + k_gap_dep * (full_disk - empty_gap)


def disk_acceleration_empty_gap(
    radius,
    r_in,
    r_out,
    G,
    dens0,
    disk_k,
    zk,
    t,
    tdep,
    disk_inner_cutoff,
    coeffs,
):
    dens = sigma_disk(dens0, radius, disk_k, t, tdep)
    inner_cut_accel = inner_disk_acceleration(radius, disk_inner_cutoff, G, dens, disk_k, coeffs)

    if disk_inner_cutoff > 0.0 and radius <= disk_inner_cutoff:
        no_gap_cut = exterior_disk_acceleration(radius, disk_inner_cutoff, G, dens, disk_k, coeffs)
        removed_gap_annulus = (
            exterior_disk_acceleration(radius, r_in, G, dens, disk_k, coeffs)
            - exterior_disk_acceleration(radius, r_out, G, dens, disk_k, coeffs)
        )
        return no_gap_cut - removed_gap_annulus

    # An empty gap removes only the annulus [r_in, r_out]; the disk beyond
    # the far edge survives, so its contribution is added back.
    if radius < r_in:
        return (
            full_disk_acceleration(G, dens, zk, disk_k)
            - exterior_disk_acceleration(radius, r_in, G, dens, disk_k, coeffs)
            + exterior_disk_acceleration(radius, r_out, G, dens, disk_k, coeffs)
            - inner_cut_accel
        )
    if radius > r_out:
        return (
            full_disk_acceleration(G, dens, zk, disk_k)
            - inner_disk_acceleration(radius, r_out, G, dens, disk_k, coeffs)
            + inner_disk_acceleration(radius, r_in, G, dens, disk_k, coeffs)
            - inner_cut_accel
        )
    return (
        inner_disk_acceleration(radius, r_in, G, dens, disk_k, coeffs)
        + exterior_disk_acceleration(radius, r_out, G, dens, disk_k, coeffs)
        - inner_cut_accel
    )


def disk_acceleration_depleted_gap(
    radius,
    r_in,
    r_out,
    G,
    dens0,
    disk_k,
    zk,
    t,
    tdep,
    disk_inner_cutoff,
    k_gap_dep,
    coeffs,
):
    no_gap = disk_acceleration_no_gap(radius, G, dens0, disk_k, zk, t, tdep, disk_inner_cutoff, coeffs)
    empty_gap = disk_acceleration_empty_gap(
        radius,
        r_in,
        r_out,
        G,
        dens0,
        disk_k,
        zk,
        t,
        tdep,
        disk_inner_cutoff,
        coeffs,
    )
    return empty_gap + k_gap_dep * (no_gap - empty_gap)


def disk_acceleration_gap_by_location(
    radius,
    semimajor_axis,
    r_in,
    r_out,
    G,
    dens0,
    disk_k,
    zk,
    t,
    tdep,
    disk_inner_cutoff,
    k_gap_dep,
    coeffs,
):
    """Hybrid gap force with the Multi_HR5183 inner-embryo correction."""
    if semimajor_axis < r_in:
        return disk_acceleration_multi_hr_inner(
            radius,
            r_in,
            G,
            dens0,
            disk_k,
            zk,
            t,
            tdep,
            k_gap_dep,
            coeffs,
        )
    if semimajor_axis <= r_in or semimajor_axis >= r_out:
        return disk_acceleration_no_gap(
            radius, G, dens0, disk_k, zk, t, tdep, disk_inner_cutoff, coeffs
        )
    return disk_acceleration_depleted_gap(
        radius,
        r_in,
        r_out,
        G,
        dens0,
        disk_k,
        zk,
        t,
        tdep,
        disk_inner_cutoff,
        k_gap_dep,
        coeffs,
    )


def apply_radial_acceleration(particle, primary, accel):
    x = particle.x - primary.x
    y = particle.y - primary.y
    z = particle.z - primary.z
    radius = np.sqrt(x*x + y*y + z*z)
    if radius <= 0.0 or not np.isfinite(accel):
        return
    particle.ax += accel * x / radius
    particle.ay += accel * y / radius
    particle.az += accel * z / radius


def apply_gr_acceleration(particle, primary, G, c_code):
    x = particle.x - primary.x
    y = particle.y - primary.y
    z = particle.z - primary.z
    vx = particle.vx - primary.vx
    vy = particle.vy - primary.vy
    vz = particle.vz - primary.vz
    r2 = x*x + y*y + z*z
    if r2 <= 0.0:
        return
    radius = np.sqrt(r2)
    v2 = vx*vx + vy*vy + vz*vz
    rv = x*vx + y*vy + z*vz
    gm = G * primary.m
    prefactor = gm / (c_code * c_code * radius**3)
    common = 4.0 * gm / radius - v2
    particle.ax += prefactor * (common * x + 4.0 * rv * vx)
    particle.ay += prefactor * (common * y + 4.0 * rv * vy)
    particle.az += prefactor * (common * z + 4.0 * rv * vz)



def run_simulation(fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,Hill_or_MMR,open_disk,open_typeI=1,force_params=None):
    if open_typeI not in (0, 1, 2):
        raise ValueError("open_typeI must be 0 (off), 1 (tidal eccentricity damping), or 2 (damping plus migration)")
    run_options = force_params if isinstance(force_params, dict) else {}
    embryo_true_anomalies = run_options.get("embryo_true_anomalies")
    if embryo_true_anomalies is not None and len(embryo_true_anomalies) != n_p2:
        raise ValueError("embryo_true_anomalies must contain one angle per embryo")
    if n_j > 0:
        fixed_gap_r_in, fixed_gap_r_out = reference_gap_edges(
            float(np.atleast_1d(a_j)[0]),
            float(np.atleast_1d(e_j)[0]),
            float(np.atleast_1d(m_j)[0]) * MJUP_TO_MSUN,
            float(mstar),
        )
    else:
        fixed_gap_r_in = fixed_gap_r_out = np.nan

    ## Addtional Force
    def diskpotential_w_gap(reb_sim):

        dens0,di,h0,hi,zk,tdep,disk_inner_cutoff,k_gap_dep,n_terms,c_code,include_gr,use_gap,f_e,f_a = define_variables(force_params)
        #2-D gas disk potential
        G = sim.G;
        t = sim.t;
        particles = sim.particles;
        com = particles[0];

        coeffs = a_l_coefficients(n_terms)
        N = sim.N;
        if N <= 1 or n_j <= 0:
            return

        for n in range(sim.N - 1):
            p = particles[n + 1]
            x = p.x - com.x
            y = p.y - com.y
            z = p.z - com.z
            rp = np.sqrt(x*x + y*y + z*z)
            if rp <= 0.0:
                continue
            orbit = p.orbit(primary=com)
            if orbit.a <= 0.0:
                continue
            accel = disk_acceleration_gap_by_location(
                rp,
                orbit.a,
                fixed_gap_r_in,
                fixed_gap_r_out,
                G,
                dens0,
                di,
                zk,
                t,
                tdep,
                disk_inner_cutoff,
                k_gap_dep,
                coeffs,
            )
            apply_radial_acceleration(p, com, accel)
         
    def diskpotential_wo_gap(reb_sim):

        dens0,di,h0,hi,zk,tdep,disk_inner_cutoff,k_gap_dep,n_terms,c_code,include_gr,use_gap,f_e,f_a = define_variables(force_params)
        #2-D gas disk potential
        G = sim.G;
        t = sim.t;
        particles = sim.particles;
        com = particles[0];
        coeffs = a_l_coefficients(n_terms)

        for n in range(sim.N-1):
            p = particles[n+1];
            x = p.x - com.x;
            y = p.y - com.y;
            z = p.z - com.z;
            rp = np.sqrt(x*x + y*y + z*z);
            if rp <= 0.0:
                continue
            accel = disk_acceleration_no_gap(rp, G, dens0, di, zk, t, tdep, disk_inner_cutoff, coeffs)
            apply_radial_acceleration(p, com, accel)

        

    def typeI(reb_sim):
        dens0,di,h0,hi,zk,tdep,disk_inner_cutoff,k_gap_dep,n_terms,c_code,include_gr,use_gap,f_e,f_a = define_variables(force_params)
        # Angular-momentum-conserving eccentricity damping, with an optional
        # independent Type-I migration torque.
        G = sim.G
        t = sim.t
        particles = sim.particles;
        com = particles[0];
        if sim.N>1+n_j:
            for n in range(sim.N-n_j-1):
                p =(particles[n+1+n_j]);
                x = p.x - com.x;
                y = p.y - com.y;
                z = p.z - com.z;
                rp = np.sqrt(x*x + y*y + z*z);
                vx = p.vx - com.vx;
                vy = p.vy - com.vy;
                vz = p.vz - com.vz;
                o = p.orbit(primary=com)
                if o.a >0:
                    if disk_inner_cutoff > 0.0 and o.a <= disk_inner_cutoff:
                        continue
                    # The tidal timescale is defined at the osculating a.
                    dens = dens0 * pow(o.a,-di) * np.exp(-t/tdep);
                    h = h0 * pow(o.a,hi);
                    Omega_K = np.sqrt(G*com.m/o.a**3)
                    h_over_a = h / o.a
                    t_damp = (
                        (com.m / p.m)
                        * (com.m / (dens * o.a**2))
                        * h_over_a**4
                        / Omega_K
                    )
                    vr = (x*vx + y*vy + z*vz) / rp
                    if f_e > 0.0:
                        t_e = t_damp / f_e
                        damping = 2.0 * vr / (t_e * rp)
                        p.ax -= damping * x
                        p.ay -= damping * y
                        p.az -= damping * z

                    if open_typeI == 2 and f_a > 0.0:
                        # Tanaka et al. (2002), Sigma proportional to r^-di:
                        # t_a = [(2.7+1.1di) f_a]^-1
                        #       (M*/Mp)(M*/Sigma a^2)(H/a)^2/Omega_K.
                        tanaka_coefficient = 2.7 + 1.1 * di
                        t_a = t_damp / (
                            f_a * tanaka_coefficient * h_over_a**2
                        )
                        dvx = vx - vr*x/rp
                        dvy = vy - vr*y/rp
                        dvz = vz - vr*z/rp
                        dv = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
                        if dv <= 0.0:
                            continue
                        temp = np.sqrt(G * com.m / rp) / (2.0 * t_a)
                        p.ax -= temp*dvx/dv
                        p.ay -= temp*dvy/dv
                        p.az -= temp*dvz/dv

    def gr(reb_sim):
        dens0,di,h0,hi,zk,tdep,disk_inner_cutoff,k_gap_dep,n_terms,c_code,include_gr,use_gap,f_e,f_a = define_variables(force_params)
        if include_gr != 1:
            return
        G = sim.G
        particles = sim.particles
        com = particles[0]
        for n in range(sim.N - n_j - 1):
            apply_gr_acceleration(particles[n + 1 + n_j], com, G, c_code)

    def additional_forces(sim):
        if open_disk ==1 and gap ==1:
            diskpotential_w_gap(sim)
        if open_disk ==1 and gap ==0:
            diskpotential_wo_gap(sim)
        gr(sim)
        if open_typeI > 0:
            typeI(sim)


    # integration time in yrs
    t_int = run_options.get("t_int", 2e7)
    dt_int = run_options.get("dt_int", 1e3)
    output_dir = run_options.get("output_dir", data_dir)


    gap = define_variables(force_params)[11]

    # Restart support: with "resume": 1 in force_params and an existing
    # archive, reload the last snapshot and continue appending to the same
    # file instead of starting over.  The disk force depends only on sim.t,
    # so continuation is seamless.  Callbacks (additional_forces,
    # collision_resolve) are not stored in the binary and must be re-set.
    archive_path = os.path.join(output_dir, fname)
    if run_options.get("resume", 0) and os.path.exists(archive_path):
        sim = rebound.Simulation(archive_path)  # last snapshot of the archive
        if sim.t >= t_int:
            return fname
        sim.additional_forces = additional_forces
        sim.force_is_velocity_dependent = 1
        sim.collision = "direct"
        sim.collision_resolve = "merge"
        sim.collision_resolve_keep_sorted = 1
        sim.track_energy_offset = 1
        sim.save_to_file(archive_path, interval=dt_int, delete_file=False)
        sim.integrate(t_int)
        return fname

    sim = rebound.Simulation()
    sim.softening=1e-3

    # collision
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.collision_resolve_keep_sorted = 1
    sim.track_energy_offset = 1

    # planet unit conversion

    m_j = m_j*MJUP_TO_MSUN # in unit of msun
    m_p2 = m_p2*3.00e-6 # in unit of msun

    r_j = r_j*4.78e-4;  # in unit of au
    r_p2 = r_p2*4.26352e-5;  # in unit of au
    rstar = rstar*4.65e-3 # in unit of au



    sim.units = ('yr', 'AU', 'Msun')
    sim.additional_forces = additional_forces
    sim.force_is_velocity_dependent = 1;



    sim.add(m=float(mstar),r=float(rstar))

    # Physical radii are passed so "direct" collision detection can
    # trigger embryo-embryo (and embryo-giant) mergers.
    for n in range(n_j):
        sim.add(m=float(m_j[n]),r=float(r_j[n]),a=float(a_j[n]), e=float(e_j[n]),inc=0,Omega=0,omega=0,f=0,primary = sim.particles[0])

    if Hill_or_MMR == 0:
        r_h=pow(m_p2/(3*mstar),1/3)
        k_hill=((1+k*r_h)/(1-k*r_h))
        for n in range(n_p2):
            f = 0.0 if embryo_true_anomalies is None else float(embryo_true_anomalies[n])
            sim.add(m=float(m_p2),r=float(r_p2),a=float(a_p2*k_hill**n), e=float(e_p2),inc=0,Omega=0,omega=0,f=f,primary = sim.particles[0])
    elif Hill_or_MMR == 1:
        for n in range(n_p2):
            p_p2 = a_p2**(3/2)
            f = 0.0 if embryo_true_anomalies is None else float(embryo_true_anomalies[n])
            sim.add(m=float(m_p2),r=float(r_p2),a=a_p2*(p_p2*(2)**n)**(2/3), e=float(e_p2),inc=0,Omega=0,omega=0,f=f,primary = sim.particles[0])
    elif Hill_or_MMR == 2:
        for n in range(n_p2):
            p_p2 = a_p2**(3/2)
            f = 0.0 if embryo_true_anomalies is None else float(embryo_true_anomalies[n])
            sim.add(m=float(m_p2),r=float(r_p2),a=a_p2*(p_p2*(3/2)**n)**(2/3), e=float(e_p2),inc=0,Omega=0,omega=0,f=f,primary = sim.particles[0])
    else:
        for n in range(n_p2):
            a_random = 1+np.random.random()*1
            f = 0.0 if embryo_true_anomalies is None else float(embryo_true_anomalies[n])
            sim.add(m=float(m_p2),r=float(r_p2),a=a_random, e=float(e_p2),inc=0,Omega=0,omega=0,f=f,primary = sim.particles[0])



    sim.N_active=sim.N

    sim.move_to_com()

    sim.integrator = "mercurius"
    dt_fraction = run_options.get("dt_fraction", 0.1)
    if dt_fraction <= 0.0:
        raise ValueError("dt_fraction must be positive")
    sim.dt = sim.particles[n_j+1].P * dt_fraction

    os.makedirs(output_dir, exist_ok=True)
    sim.save_to_file(os.path.join(output_dir, fname), interval=dt_int, delete_file=True)
    sim.integrate(t_int)

    return fname 




    



def build_default_mmr_runs():
    k = 8
    n_j = 1
    n_p2 = 4
    mstar = 1.0
    rstar = 1.0
    m_j = np.array([1.0])
    r_j = np.array([1.0])
    a_j = np.array([20.0])
    a_p2 = 1.0
    e_p2 = 0.0

    input_items = []
    for m_p2, r_p2 in zip([0.1, 1.0], [0.38, 1.0]):
        for e_j in [np.array([0.8]), np.array([0.0])]:
            for hill_or_mmr in [1, 2]:
                fname = (
                    "MMR_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_"
                    "mp2_%.2f_ap2_%.2f_ep2_%.2f_MMR_%d.bin"
                    % (n_j, n_p2, mstar, m_j[0], a_j[0], e_j[0], m_p2, a_p2, e_p2, hill_or_mmr)
                )
                input_items.append(
                    (fname, k, n_j, n_p2, mstar, rstar, m_j, r_j, a_j, e_j,
                     m_p2, r_p2, a_p2, e_p2, hill_or_mmr, 1, 1)
                )

                fname_nodisk = (
                    "MMR_nodisk_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_"
                    "mp2_%.2f_ap2_%.2f_ep2_%.2f_MMR_%d.bin"
                    % (n_j, n_p2, mstar, m_j[0], a_j[0], e_j[0], m_p2, a_p2, e_p2, hill_or_mmr)
                )
                input_items.append(
                    (fname_nodisk, k, n_j, n_p2, mstar, rstar, m_j, r_j, a_j, e_j,
                     m_p2, r_p2, a_p2, e_p2, hill_or_mmr, 0, 0)
                )
    return input_items


def main():
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    input_items = build_default_mmr_runs()
    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(run_simulation, input_items):
            print(result)


if __name__ == "__main__":
    main()
