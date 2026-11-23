# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# %% Parameters (Bakker paper, Example section)
k = 2           #hydraulic conductivity (m/day)
n = 0.2         #effective porosity
nus = 0.025     #dimensionless density difference
zt = 0          #top of aquifer (m)
zb = -45        #bottom of aquifer (m)
H = zt - zb     #aquifer thickness (m)
T = k*H         #transmissivity (m^2/day)

Q0 = 0.1        #uniform flow toward the coast (m^2/day)
Qd = 0.05       #extraction from the ditch (m^2/day)
x_dit = 200     #ditch distance from the coast (m)

#Qd_b = 0.05     #extraction from the second ditch (m^2/day)
#x_dit_b = 300   #second ditch distance from the coast (m)


h0 = 0.05       #head at the coast (m), Glover (1959)
zeta0 = -2      #interface at the coast (m), Glover (1959)

# %% Grid and time stepping
L = 1000        #length of cross-section (m)
nx = 101        #number of cells
x = np.linspace(0,L,nx)     #set x grid, x = 0 is the coast
dx = x[1] - x[0]            #calculate grid space size
i_dit = int(round(x_dit/dx))    #cell that holds the ditch
#i_dit_b = int(round(x_dit_b/dx))    #cell that holds the second ditch

tf = 500                #final time (yr)
dt = 0.5                #time step length (yr)
nt = int(tf/dt)         #number of time steps
t = np.arange(nt+1)*dt  #vector holding each time
dt_day = dt*365.25      #time step in days, to match k

# %% Pre-allocate and set the initial condition
#Pre-allocate
h_sol = np.zeros((nx,nt+1))
zeta_sol = np.zeros((nx,nt+1))

#Initial condition: steady interface for uniform flow Q0 (Ghyben-Herzberg)
zeta = -np.sqrt(zeta0**2 + 2*Q0*x/(k*nus))
zeta[zeta < zb] = zb
i_toe = np.argmax(zeta <= zb)     #first cell with no saltwater
h = -nus*zeta
h[i_toe:] = h[i_toe] + Q0*(x[i_toe:] - x[i_toe])/T

h_sol[:,0] = h
zeta_sol[:,0] = zeta


# %% Build the head matrix for eq. (1)
#Head matrix for eq. (1), S = 0 so it never changes
data = np.array([(T/dx)*np.ones(nx), (-2*T/dx)*np.ones(nx), (T/dx)*np.ones(nx)])
diags = np.array([-1, 0, 1])
M_h = sp.sparse.spdiags(data, diags, nx, nx).toarray()

M_h[0,:] = 0        #fixed head at the coast (Dirichlet)
M_h[0,0] = 1
M_h[-1,-1] = -T/dx  #inflow Q0 at the inland edge (Neumann)

M_h_inv = np.linalg.inv(M_h)   #M_h never changes, so invert once

# %% Time loop: implicit block-centered finite differences
#Solve eq. (1) for h with zeta fixed, then eq. (2) for zeta with h fixed
for i in range(nt):
    #salt transmissivity on the cell faces (uses saltwater thickness)
    tau = k*nus*((zeta[1:] + zeta[:-1])/2 - zb)
    tau[tau < 0] = 0

    #eq. (1): right-hand side for head
    F_zeta = tau*(zeta[1:] - zeta[:-1])/dx
    R_h = np.zeros(nx)
    R_h[1:] = R_h[1:] + F_zeta
    R_h[1:-1] = R_h[1:-1] - F_zeta[1:]
    R_h[-1] = R_h[-1] - Q0          #inflow at the inland edge
    R_h[i_dit] = R_h[i_dit] + Qd    #extraction from the ditch
    #R_h[i_dit_b] = R_h[i_dit_b] + Qd_b    #extraction from the second ditch
    R_h[0] = h0

    h = M_h_inv @ R_h

    #eq. (2): matrix changes every step because tau depends on zeta
    lower = np.append(tau, 0)/dx
    upper = np.append(0, tau)/dx
    main = -n*dx/dt_day - (np.append(0, tau) + np.append(tau, 0))/dx
    data = np.array([lower, main, upper])
    M_z = sp.sparse.spdiags(data, diags, nx, nx).toarray()
    M_z[0,:] = 0        #fixed interface at the coast
    M_z[0,0] = 1

    F_h = tau*(h[1:] - h[:-1])/(dx*nus)
    R_z = -n*dx/dt_day*zeta
    R_z[1:] = R_z[1:] + F_h
    R_z[1:-1] = R_z[1:-1] - F_h[1:]
    R_z[0] = zeta0

    zeta = np.linalg.inv(M_z) @ R_z
    zeta[zeta < zb] = zb
    zeta[zeta > zt] = zt

    h_sol[:,i+1] = h
    zeta_sol[:,i+1] = zeta

# %% Results: pumping rate vs. rise of the interface under the ditch
#Science question: how does the pumping rate change the rise under the ditch over 100 yr?
tf2 = 100
nt2 = int(tf2/dt)
t2 = np.arange(nt2+1)*dt

zeta_dit = []
zeta_end = []

for frac in [.25, .5, .75]:
    Qd2 = frac*Q0

    #reset to the initial condition
    zeta = zeta_sol[:,0].copy()
    z_d = np.zeros(nt2+1)
    z_d[0] = zeta[i_dit]

    for i in range(nt2):
        tau = k*nus*((zeta[1:] + zeta[:-1])/2 - zb)
        tau[tau < 0] = 0

        F_zeta = tau*(zeta[1:] - zeta[:-1])/dx
        R_h = np.zeros(nx)
        R_h[1:] = R_h[1:] + F_zeta
        R_h[1:-1] = R_h[1:-1] - F_zeta[1:]
        R_h[-1] = R_h[-1] - Q0
        R_h[i_dit] = R_h[i_dit] + Qd2
        R_h[0] = h0
        h = M_h_inv @ R_h

        lower = np.append(tau, 0)/dx
        upper = np.append(0, tau)/dx
        main = -n*dx/dt_day - (np.append(0, tau) + np.append(tau, 0))/dx
        M_z = sp.sparse.spdiags(np.array([lower, main, upper]), diags, nx, nx).toarray()
        M_z[0,:] = 0
        M_z[0,0] = 1

        F_h = tau*(h[1:] - h[:-1])/(dx*nus)
        R_z = -n*dx/dt_day*zeta
        R_z[1:] = R_z[1:] + F_h
        R_z[1:-1] = R_z[1:-1] - F_h[1:]
        R_z[0] = zeta0

        zeta = np.linalg.inv(M_z) @ R_z
        zeta[zeta < zb] = zb
        zeta[zeta > zt] = zt
        z_d[i+1] = zeta[i_dit]

    zeta_dit.append(z_d)
    zeta_end.append(zeta)
    print(f'Q = {frac} Q0: interface under the gallery after {tf2} yr = {round(z_d[-1], 1)} m')

# %% Figure style (same look for every figure)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False})
c_rate = ['#2a78d6', '#eb6834', '#1baf7a']   #25%, 50%, 75% of Q0 (same colors in every figure)
rates = [.25, .5, .75]

# %% Figure 1: initial interface (steady state with no pumping)
plt.figure(figsize=(9, 4.5))
plt.fill_between(x, zeta_sol[:,0], zt, color='#cde2fb', label='Freshwater')
plt.fill_between(x, zb, zeta_sol[:,0], color='#d9d8d4', label='Saltwater')
plt.plot(x, zeta_sol[:,0], 'k-', linewidth=2, label=r'Interface $\zeta$')
plt.plot(x_dit, zt, 'kv', markersize=10, label=f'Gallery (x = {x_dit} m)')
plt.xlim(0, L)
plt.ylim(zb, zt + 3)
plt.xlabel('Distance from the coast, x (m)')
plt.ylabel('Elevation, z (m)')
plt.title('Figure 1. Initial interface for uniform flow $Q_0$ = 0.1 m$^2$/d, no pumping')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('fig1_initial_interface.png', dpi=200)
plt.show()

# %% Figure 2: interface every 50 years for 500 years at 50% pumping, with the steady solution
#steady solution seaward of the gallery (Glover 1959): flow drops to Q0 - Qd
zeta_ss = -np.sqrt(zeta0**2 + 2*(Q0 - Qd)*x/(k*nus))
zeta_ss[zeta_ss < zb] = zb

years = np.arange(50, tf+1, 50)
blues = plt.cm.Blues(np.linspace(0.35, 1, len(years)))
plt.figure(figsize=(9, 4.5))
plt.plot(x, zeta_sol[:,0], 'k-', linewidth=2, label='Initial (t = 0)')
for j in range(len(years)):
    plt.plot(x, zeta_sol[:,int(years[j]/dt)], color=blues[j], linewidth=1.5)
plt.plot(x[:i_dit+1], zeta_ss[:i_dit+1], 'k--', linewidth=1.5, label='Steady solution, seaward of gallery')
plt.axvline(x_dit, color='gray', linestyle=':', label='Gallery')
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(blues), norm=plt.Normalize(25, tf+25))
plt.colorbar(sm, ax=plt.gca(), ticks=years[::2], label='Time (yr)')
plt.xlim(0, 800)
plt.ylim(zb, zt)
plt.xlabel('Distance from the coast, x (m)')
plt.ylabel(r'Interface elevation, $\zeta$ (m)')
plt.title('Figure 2. Interface every 50 yr, pumping 50% of $Q_0$ for 500 yr')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('fig2_interface_every_50yr.png', dpi=200)
plt.show()

# %% Figure 3: interface beneath the gallery over 100 years at each pumping rate
plt.figure(figsize=(9, 4.5))
for j in range(3):
    zeta_ss_j = -np.sqrt(zeta0**2 + 2*(Q0 - rates[j]*Q0)*x_dit/(k*nus))
    plt.plot(t2, zeta_dit[j], color=c_rate[j], linewidth=2, label=f'{int(rates[j]*100)}% of $Q_0$')
    plt.axhline(zeta_ss_j, color=c_rate[j], linestyle='--', linewidth=1)
    plt.text(tf2 + 1, zeta_dit[j][-1], f'{zeta_dit[j][-1]:.1f} m'.replace('-', '\u2212'), va='center', fontsize=10)
plt.plot([], [], 'k--', linewidth=1, label='Steady solution')
plt.xlim(0, tf2 + 12)
plt.ylim(-29, -12)
plt.xlabel('Time since pumping began (yr)')
plt.ylabel(r'$\zeta$ beneath the gallery (m)')
plt.title(f'Figure 3. Rise of the interface beneath the gallery (x = {x_dit} m)')
plt.legend(loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('fig3_rise_beneath_gallery.png', dpi=200)
plt.show()

# %% Figure 4: interface along the cross-section after 100 years at each pumping rate
plt.figure(figsize=(9, 4.5))
plt.plot(x, zeta_sol[:,0], 'k-', linewidth=2, label='Initial')
for j in range(3):
    plt.plot(x, zeta_end[j], color=c_rate[j], linewidth=2, label=f'{int(rates[j]*100)}% of $Q_0$, 100 yr')
plt.axvline(x_dit, color='gray', linestyle=':', label='Gallery')
plt.xlim(0, 800)
plt.ylim(zb, zt)
plt.xlabel('Distance from the coast, x (m)')
plt.ylabel(r'Interface elevation, $\zeta$ (m)')
plt.title('Figure 4. Interface after 100 yr at each pumping rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('fig4_interface_after_100yr.png', dpi=200)
plt.show()

# %% Figure 5: does the answer change with the time step or the grid spacing? (50% pumping, 100 yr)
dt_test = [1, 0.5, 0.25, 0.1]     #time steps to try (yr)
nx_test = [51, 101, 201]          #number of cells to try (dx = 20, 10, 5 m)
cases = [(nx, d) for d in dt_test] + [(m, dt) for m in nx_test]
zeta_case = []
for (nx_c, dt_c) in cases:
    x_c = np.linspace(0,L,nx_c)
    dx_c = x_c[1] - x_c[0]
    i_c = int(round(x_dit/dx_c))
    nt_c = int(round(tf2/dt_c))
    dtd_c = dt_c*365.25
    zeta = -np.sqrt(zeta0**2 + 2*Q0*x_c/(k*nus))
    zeta[zeta < zb] = zb
    data = np.array([(T/dx_c)*np.ones(nx_c), (-2*T/dx_c)*np.ones(nx_c), (T/dx_c)*np.ones(nx_c)])
    M_c = sp.sparse.spdiags(data, diags, nx_c, nx_c).toarray()
    M_c[0,:] = 0
    M_c[0,0] = 1
    M_c[-1,-1] = -T/dx_c
    M_c_inv = np.linalg.inv(M_c)
    for i in range(nt_c):
        tau = k*nus*((zeta[1:] + zeta[:-1])/2 - zb)
        tau[tau < 0] = 0
        F_zeta = tau*(zeta[1:] - zeta[:-1])/dx_c
        R_h = np.zeros(nx_c)
        R_h[1:] = R_h[1:] + F_zeta
        R_h[1:-1] = R_h[1:-1] - F_zeta[1:]
        R_h[-1] = R_h[-1] - Q0
        R_h[i_c] = R_h[i_c] + 0.5*Q0
        R_h[0] = h0
        h = M_c_inv @ R_h
        lower = np.append(tau, 0)/dx_c
        upper = np.append(0, tau)/dx_c
        main = -n*dx_c/dtd_c - (np.append(0, tau) + np.append(tau, 0))/dx_c
        M_z = sp.sparse.spdiags(np.array([lower, main, upper]), diags, nx_c, nx_c).toarray()
        M_z[0,:] = 0
        M_z[0,0] = 1
        F_h = tau*(h[1:] - h[:-1])/(dx_c*nus)
        R_z = -n*dx_c/dtd_c*zeta
        R_z[1:] = R_z[1:] + F_h
        R_z[1:-1] = R_z[1:-1] - F_h[1:]
        R_z[0] = zeta0
        zeta = np.linalg.inv(M_z) @ R_z
        zeta[zeta < zb] = zb
        zeta[zeta > zt] = zt
    zeta_case.append(zeta[i_c])
    print(f'dx = {dx_c:.0f} m, dt = {dt_c} yr: interface under the gallery after {tf2} yr = {zeta[i_c]:.2f} m')

fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
ax[0].plot(dt_test, zeta_case[:4], 'o-', color=c_rate[1], markersize=8)
ax[0].set_xscale('log')
ax[0].set_xticks(dt_test)
ax[0].set_xticklabels([str(d) for d in dt_test])
ax[0].minorticks_off()
ax[0].set_xlabel(r'Time step, $\Delta t$ (yr)')
ax[0].set_ylabel(r'$\zeta$ beneath the gallery at 100 yr (m)')
ax[0].set_title(r'Time step ($\Delta x$ = 10 m)')
ax[1].plot(L/(np.array(nx_test) - 1), zeta_case[4:], 's-', color=c_rate[1], markersize=8)
ax[1].set_xticks([5, 10, 20])
ax[1].set_xlabel(r'Grid spacing, $\Delta x$ (m)')
ax[1].set_title(r'Grid spacing ($\Delta t$ = 0.5 yr)')
ax[0].set_ylim(np.mean(zeta_case) - 0.5, np.mean(zeta_case) + 0.5)
fig.suptitle('Figure 5. Sensitivity to numerical choices, 50% of $Q_0$')
plt.tight_layout()
plt.savefig('fig5_numerical_sensitivity.png', dpi=200)
plt.show()
