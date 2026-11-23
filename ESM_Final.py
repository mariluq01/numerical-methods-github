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

plt.figure()
plt.plot(x, zeta, 'k-', linewidth=2)
plt.xlabel('x (m)')
plt.ylabel(r'Interface elevation $\zeta$ (m)')
plt.title('Initial interface')
plt.show()

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

# %% Figure: interface every 50 years (compare with Bakker Fig. 1a)
plt.figure(figsize=(10, 5))
for yr in range(0, tf+1, 50):
    j = int(yr/dt)
    if yr == 0:
        plt.plot(x, zeta_sol[:,j], 'k-', linewidth=2, label='Initial')
    else:
        plt.plot(x, zeta_sol[:,j], color='gray', linewidth=1)
plt.axvline(x_dit, color='b', linestyle=':', label='Gallery')
#plt.axvline(x_dit_b, color='b', linestyle=':')
plt.ylim(zb, 5)
plt.xlabel('x (m)')
plt.ylabel(r'Interface elevation $\zeta$ (m)')
plt.title('Upconing of the interface below the gallery, every 50 yr')
plt.legend()
plt.show()

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

plt.figure()
for j in range(3):
    plt.plot(t2, zeta_dit[j], label=f'Q = {[.25, .5, .75][j]} Q0')
plt.xlabel('Time (yr)')
plt.ylabel(r'$\zeta$ under the gallery (m)')
plt.title('Rise of the interface below the ditch')
plt.legend()
plt.show()

plt.figure()
for j in range(3):
    plt.plot(x, zeta_end[j], label=f'Q = {[.25, .5, .75][j]} Q0')
plt.axvline(x_dit, color='b', linestyle=':')
#plt.axvline(x_dit_b, color='b', linestyle=':')
plt.ylim(zb, 5)
plt.xlabel('x (m)')
plt.ylabel(r'Interface elevation $\zeta$ (m)')
plt.title(f'Interface after {tf2} yr')
plt.legend()
plt.show()