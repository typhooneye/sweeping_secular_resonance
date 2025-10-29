#!/usr/bin/env python

#SBATCH --job-name=planetesimal-0326
#SBATCH --output=logs/planetesimal-0326_%j.out
#SBATCH --error=logs/planetesimal-0326_%j.err
#SBATCH --account=pi-abbot
#SBATCH --nodes=1
#SBATCH --exclusive


###notes 3/26
# compared to 3/24
# dens0=50g/cm^2
# f_e=1
# data directory changed
# revised index for disk_potential
# tend=2e7
# add if len(sim.particles)>1


import multiprocessing
import sys
import os

sys.path.append(os.getcwd()) 


import rebound 
import numpy as np
import math
import seaborn as sns


data_dir = '/project2/abbot/xuanji/5.SSR/data-032624/'


# ### Parameters Input 
def define_variables(var):
    dens0 = 50.0 * 1.125e-7; # in unit of m_sun/au^2
    di = 1.01
    h0 = 0.05
    hi = 1.25
    zk = 1.094
    tdep = 1000000.0
    mcrit = 0.00028
    gama = 1.4
    rt = 0.01
    rm = 0.001
    coef = 10.0
    q1 = 3e-07
    q2 = 0.001
    f_e = 1.0
    f_a = 1.0
    return dens0,di,h0,hi,zk,tdep,mcrit,gama,rt,rm,coef,q1,q2,f_e,f_a



def run_simulation(fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,Hill_or_MMR,open_disk,open_typeI):

    ## Addtional Force
    def diskpotential_w_gap(reb_sim):

        dens0,di,h0,hi,zk,tdep,mcrit,gama,rt,rm,coef,q1,q2,f_e,f_a = define_variables(1)
        #2-D gas disk potential
        G = sim.G;
        t = sim.t;
        particles = sim.particles;
        com = particles[0];

        A=np.zeros(10);
        A[0] = 1.0;
        for j in np.arange(1,10,1):
            A[j] = A[j-1]*(2.*j-1)*(2.*j-1)/4./j/j;
        
        N = sim.N;
        
        if N>1:
            for n in range(n_j):
                p = particles[n+1];

                x = p.x - com.x;
                y = p.y - com.y;
                z = p.z - com.z;
                rp = np.sqrt(x*x + y*y + z*z);

                o = p.calculate_orbit(primary=com)
                dens = dens0 * pow(o.a,-di) * np.exp(-t/tdep);
                ain = o.a*(1-o.e)*(1 - pow(p.m/com.m/3., 0.333));
                aout = o.a*(1+o.e)*(1 + pow(p.m/com.m/3., 0.333));
                if o.a > 0:
                    #printf("%e\t%e\n", ain, aout);
                    nt_sum = 0.;
                    for j in range(10):
                        nt = 2*(j)*A[j]*pow(o.a/aout,2*(j)-1+di)/(2*(j)-1+di) - (2*(j)+1)*A[j]*pow(ain/o.a,2*(j)+2-di)/(2*(j)+2-di);
                        nt_sum += nt;

                    f = 2.*M_PI*G*dens*nt_sum;
            #         f = -4.*M_PI*G*dens*zk
                else:
                    f = 0.;
                p.ax += f*x/rp;
                p.ay += f*y/rp;
                p.az += f*z/rp;
        
        if N>1+n_j:
            for n in range(sim.N-1-n_j):
                p2 = particles[n+1+n_j];

                x2 = p2.x - com.x;
                y2 = p2.y - com.y;
                z2 = p2.z - com.z;
                rp2 = np.sqrt(x2*x2 + y2*y2 + z2*z2);

                o2 = p2.calculate_orbit(primary=com)
                dens2 = dens0 * pow(o2.a,-di) * np.exp(-t/tdep);
                if o2.a > 0:
                    #printf("%e\t%e\n", ain, aout);
                    f2 = -4.*M_PI*G*dens2*zk
                    p2.ax += f2*x2/rp2;
                    p2.ay += f2*y2/rp2;
                    p2.az += f2*z2/rp2;
                else:
                    f2 = 0
                    p2.ax += 0
                    p2.ay += 0
                    p2.az += 0
#                 sim.remove(n+2)
         
    def diskpotential_wo_gap(reb_sim):

        dens0,di,h0,hi,zk,tdep,mcrit,gama,rt,rm,coef,q1,q2,f_e,f_a = define_variables(1)
        #2-D gas disk potential
        G = sim.G;
        t = sim.t;
        particles = sim.particles;
        com = particles[0];


        N = sim.N;
        
        for n in range(sim.N-1):
            p = particles[n+1];

            x = p.x - com.x;
            y = p.y - com.y;
            z = p.z - com.z;
            rp = np.sqrt(x*x + y*y + z*z);

            o = p.calculate_orbit(primary=com)
            dens = dens0 * pow(rp,-di) * np.exp(-t/tdep);
            if o.a > 0:
                f = -4.*M_PI*G*dens*zk
                p.ax += f*x/rp;
                p.ay += f*y/rp;
                p.az += f*z/rp;

            # else:
                # sim.remove(n+1)

        

    def typeI(reb_sim):
        dens0,di,h0,hi,zk,tdep,mcrit,gama,rt,rm,coef,q1,q2,f_e,f_a = define_variables(1)
        # Velocity dependent type I damping
        G = sim.G
        t = sim.t
        particles = sim.particles;
        com = particles[0];
        N = sim.N;
        
        if N>1+n_j:
            for n in range(sim.N-n_j-1):
                p =(particles[n+1+n_j]);
                x = p.x - com.x;
                y = p.y - com.y;
                z = p.z - com.z;
                rp = np.sqrt(x*x + y*y + z*z);
                vx = p.vx - com.vx;
                vy = p.vy - com.vy;
                vz = p.vz - com.vz;
                mu = G*(com.m + p.m);
                q = p.m / com.m;

                o = p.calculate_orbit(primary=com)
                if o.a >0:
                    dens = dens0 * pow(rp,-di) * np.exp(-t/tdep);
                    h = h0 * pow(rp,hi);
                    Omega_K=np.sqrt(G*com.m/o.a**3)
                    cs = Omega_K * h
                    t_damp = pow(p.m,-1)*pow(dens*o.a**2,-1)*pow(cs/(o.a*Omega_K),4)*pow(Omega_K,-1);

                    #force of eccentricity damping
                    t_e = t_damp / f_e;
                    vr = (x*vx + y*vy + z*vz)/rp;
                    temp = vr/t_e;
                    p.ax -= temp*x/rp;
                    p.ay -= temp*y/rp;
                    p.az -= temp*z/rp;

                if (open_typeI == 2):
                    #force of type I migration
                    rtt2 = pow(rp/rt,2);
                    f_nsc = coef * (1 - 2*rtt2/(1+rtt2));
                    rrm4 = pow(rp/rm,4);
                    f_lb = 1 - 2*rrm4/(1+rrm4);      
                    qq12 = pow(q/q1,2);
                    qq22 = pow(q/q2,-2);
                    qq = qq12*qq22/(2*qq12*qq22+qq12+qq22);

                    fa = (f_nsc*qq + f_lb);
                    t_a = t_damp /f_a/fa/h/h;

                    dvx = vx - vr*x/rp;
                    dvy = vy - vr*y/rp;
                    dvz = vz - vr*z/rp;
                    dv = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
                    temp = np.sqrt(G * com.m / rp)/ 2. / t_a;
                    p.ax += temp*dvx/dv;
                    p.ay += temp*dvy/dv;
                    p.az += temp*dvz/dv;

    def additional_forces(sim):
        if open_disk ==1 and gap ==1:
            diskpotential_w_gap(sim)
        if open_disk ==1 and gap ==0:
            diskpotential_wo_gap(sim)
        if open_typeI > 0:
            typeI(sim)


    M_PI = math.pi

    # integration time in yrs
    t_int = 2e7
    dt_int = 1e3


    gap = 1
    
    sim = rebound.Simulation()
    sim.N=0
    sim.softening=1e-3

    # collision
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.collision_resolve_keep_sorted = 1
    sim.track_energy_offset = 1

    # planet unit conversion

    m_j = m_j*9.545e-4 # in unit of msun
    m_p2 = m_p2*3.00e-6 # in unit of msun

    r_j = r_j*4.78e-4;  # in unit of au
    r_p2 = r_p2*4.26352e-5;  # in unit of au
    rstar = rstar*4.65e-3 # in unit of au



    sim.units = ('yr', 'AU', 'Msun')
    sim.additional_forces = additional_forces
    sim.force_is_velocity_dependent = 1;



    sim.add(m=float(mstar),r=float(rstar))

    for n in range(n_j):
        sim.add(m=float(m_j[n]),a=float(a_j[n]), e=float(e_j[n]),inc=0,Omega=0,omega=0,f=0,primary = sim.particles[0])

    if Hill_or_MMR == 0:
        r_h=pow(m_p2/(3*mstar),1/3)
        k_hill=((1+k*r_h)/(1-k*r_h))
        for n in range(n_p2):
            sim.add(m=float(m_p2),a=float(a_p2*k_hill**n), e=float(e_p2),inc=0,Omega=0,omega=0,f=0,primary = sim.particles[0])
    elif Hill_or_MMR == 1:
        for n in range(n_p2):
            p_p2 = a_p2**(3/2)
            sim.add(m=float(m_p2),a=a_p2*(p_p2*(2)**n)**(2/3), e=float(e_p2),inc=0,Omega=0,omega=0,f=0,primary = sim.particles[0])
    elif Hill_or_MMR == 2:
        for n in range(n_p2):
            p_p2 = a_p2**(3/2)
            sim.add(m=float(m_p2),a=a_p2*(p_p2*(3/2)**n)**(2/3), e=float(e_p2),inc=0,Omega=0,omega=0,f=0,primary = sim.particles[0])
    else:
        for n in range(n_p2):
            a_random = 1+np.random.random()*1
            sim.add(m=float(m_p2),a=a_random, e=float(e_p2),inc=0,Omega=0,omega=0,f=0,primary = sim.particles[0])



    #all planets are active particles
    sim.N_active=sim.N
    Ncount=sim.N

    #Move to center of momentum and center of mass frame
    sim.move_to_com()

    E0=sim.calculate_energy()

    # sim._output_timing_last=tmax

    sim.integrator = "Mercurius"
    sim.dt = sim.particles[n_j+1].P*0.1

    sim.automateSimulationArchive(data_dir+fname,interval=dt_int,deletefile=True)
    sim.integrate(t_int)

    return fname 




    



if __name__ == '__main__':


    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    # create pool of ncpus workers
    p = multiprocessing.Pool(ncpus)

    

    #======================================



    # #----------Single Planet-------------

    # # Hill radius coeficient 
    # k=8.

    # # number of jupiter
    # n_j = 1
    # # numer of embryos
    # n_p2 = 1
    


    # # in unit of sun
    # mstar = 1.07
    # rstar = 1.53

    # # in unit of Jupiter
    # m_j = np.array([3.31])
    # r_j = np.array([1.17])
    # a_j = np.array([18])
    # e_j = np.array([0.84])

    # # in unit of Earth
    # m_p2 = 1
    # r_p2 = 1
    # a_p2 = 1
    # e_p2 = 0


    # input_items = []
    # fname = 'HR5183_single_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    # input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,1,1))
    # fname2 = 'HR5183_single_nodisk_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    # input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,0,0))

    # mstar = 1.
    # rstar = 1.
    # m_j = np.array([1])
    # r_j = np.array([1])


    # e_j_s = np.array([[0.8],[0.4],[0.0]])
    # a_j_s = np.array([[20],[10]])

    # # in unit of Earth
    # m_p2 = 1
    # r_p2 = 1
    # a_p2 = 1
    # e_p2 = 0

    # for e_j in e_j_s:
    #     for a_j in a_j_s:
    #         fname = 'single_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    #         input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,1,1))
    #         fname2 = 'single_nodisk_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    #         input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,0,0))

    # n_a = 0
    # for result in p.starmap(run_simulation,input_items):
    #     print(result)
    #     n_a+=1



    # #----------K_Hill-------------

    # # Hill radius coeficient 
    # k=8.

    # # number of jupiter
    # n_j = 1
    # # numer of embryos
    # n_p2 = 4
    


    # # in unit of sun
    # mstar = 1.07
    # rstar = 1.53

    # # in unit of Jupiter
    # m_j = np.array([3.31])
    # r_j = np.array([1.17])
    # a_j = np.array([18])
    # e_j = np.array([0.84])

    # # in unit of Earth
    # m_p2_s = [0.1,1]
    # r_p2_s = [0.38,1]
    # a_p2 = 1
    # e_p2 = 0


    # input_items = []
    # for n,m_p2 in enumerate(m_p2_s):
    #     r_p2 = r_p2_s[n]
    #     fname = 'HR5183_khill_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    #     input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,1,1))
    #     fname2 = 'HR5183_khill_nodisk_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    #     input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,0,0))


    # mstar = 1.
    # rstar = 1.
    # m_j = np.array([1])
    # r_j = np.array([1])


    # e_j_s = np.array([[0.8],[0.4],[0.0]])
    # a_j_s = np.array([[20],[10]])

    # # in unit of Earth
    # m_p2_s = [0.1,1]
    # r_p2_s = [0.38,1]
    # a_p2 = 1
    # e_p2 = 0


    # for n,m_p2 in enumerate(m_p2_s):
    #     r_p2 = r_p2_s[n]
    #     for e_j in e_j_s:
    #         for a_j in a_j_s:
    #             fname = 'khill_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    #             input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,1,1))
    #             fname2 = 'khill_nodisk_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    #             input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,0,0,0))


    # n_a = 0
    # for result in p.starmap(run_simulation,input_items):
    #     print(result)
    #     n_a+=1


    #----------------MMR------------------------------


    k=8

    # number of jupiter
    n_j = 1
    # numer of embryos
    n_p2 = 4
    

    input_items = []


    # # in unit of sun
    # mstar = 1.07
    # rstar = 1.53

    # # in unit of Jupiter
    # m_j = np.array([3.31])
    # r_j = np.array([1.17])
    # a_j = np.array([18])
    # e_j = np.array([0.84])


    # # in unit of Earth
    # m_p2_s = [0.1,1]
    # r_p2_s = [0.38,1]
    # a_p2 = 1
    # e_p2 = 0

    # for n,m_p2 in enumerate(m_p2_s):
    #     for Hill_or_MMR in [1,2]:
    #         r_p2 = r_p2_s[n]
    #         fname = 'HR5183_MMR_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f_MMR_%d.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2,Hill_or_MMR)
    #         input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,Hill_or_MMR,1,1))
    #         fname2 = 'HR5183_MMR_nodisk_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f_MMR_%d.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2,Hill_or_MMR)
    #         input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,Hill_or_MMR,0,0))


    mstar = 1.
    rstar = 1.
    m_j = np.array([1])
    r_j = np.array([1])



    e_j_s = np.array([[0.8],[0.0]])
    a_j_s = [20,10]
    a_j = np.array([20])
    # a_j_s = np.array([[18],[30],[10]])

    # in unit of Earth
    m_p2_s = [0.1,1]
    r_p2_s = [0.38,1]
    a_p2 = 1
    e_p2 = 0

    for n,m_p2 in enumerate(m_p2_s):
        r_p2 = r_p2_s[n]
        for e_j in e_j_s:
            for Hill_or_MMR in [1,2]:
                fname = 'MMR_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f_MMR_%d.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2,Hill_or_MMR)
                input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,Hill_or_MMR,1,1))
                fname2 = 'MMR_nodisk_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f_MMR_%d.bin'%(n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2,Hill_or_MMR)
                input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,Hill_or_MMR,0,0))

    n_a = 0
    for result in p.starmap(run_simulation,input_items):
        print(result)
        n_a+=1


    # #----------planetesimal-------------

    # # Hill radius coeficient 
    # k=8.

    # # number of jupiter
    # n_j = 1
    # # numer of embryos
    # n_p2 = 100
    
    # # in unit of sun
    # mstar = 1.07
    # rstar = 1.53

    # # in unit of Jupiter
    # m_j = np.array([3.31])
    # r_j = np.array([1.17])
    # a_j = np.array([18])
    # e_j = np.array([0.84])

    # # in unit of Earth
    # m_p2 = 0.0025
    # r_p2 = 0.0025**(1/3)
    # a_p2 = 1
    # e_p2 = 0


    # input_items = []



    # fname = 'HR5183_planetesimals_1-2au_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    # input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,100,1,1))

    # fname2 = 'HR5183_nodisk_planetesimals_1-2au_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    # input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,100,0,0))

    # mstar = 1.
    # rstar = 1.
    # m_j = np.array([1])
    # r_j = np.array([1])


    # e_j_s = np.array([0.8])
    # a_j_s = np.array([18])

    # # in unit of Earth
    # m_p2 = 0.0025
    # r_p2 = 0.0025**(1/3)
    # a_p2 = 1
    # e_p2 = 0


    # fname = 'planetesimals_1-2au_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    # input_items.append((fname,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,100,1,1))

    # fname2 = 'planetesimals_nodisk_1-2au_%d_nj_%d_np2_%d_mstar_%.2f_mj_%.2f_aj_%.2f_ej_%.2f_mp2_%.2f_ap2_%.2f_ep2_%.2f.bin'%(k,n_j,n_p2,mstar,m_j[0],a_j[0],e_j[0],m_p2,a_p2,e_p2)
    # input_items.append((fname2,k,n_j,n_p2,mstar,rstar,m_j,r_j,a_j,e_j,m_p2,r_p2,a_p2,e_p2,100,0,0))

    # n_a = 0
    # for result in p.starmap(run_simulation,input_items):
    #     print(result)
    #     n_a+=1
    # 