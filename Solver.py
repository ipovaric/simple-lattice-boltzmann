import numpy as np
from matplotlib import pyplot

plot_every = 20

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    Nx = 400
    Ny = 100
    # tau = 0.53 # kinematic viscosity and time scale
    tau = 0.6           # collision timescale
    Nt = 30000          # duration of simulation

    # lattice speeds and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    # F[:, :, 3] = 2.9    # init flow rate
    F[:, :, 3] = 3.1    # init flow rate

    # obstacle
    cylinder = np.full((Ny, Nx), False)     # init a mask for entire domain

    for y in range(0, Ny):
        for x in range(0, Nx):
            # define origin and size of cylinder
            cyl_x_origin = Nx//4
            cyl_y_origin = Ny//2
            cyl_size = 20
            if(distance(cyl_x_origin, cyl_y_origin, x, y) < cyl_size):
                cylinder[y][x] = True

    # main loop
    for it in range(Nt):
        print(it) 

        # wall boundary conditions (zou-he condition)
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        # streaming to neighboring lattices
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

        # set collision boundary conditions (opposites)
        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # set to opposite direction

        # fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        F[cylinder, :] = bndryF     # set velocities to opposites on the boundary
        ux[cylinder] = 0            # vel w/in boundry 
        uy[cylinder] = 0            # vel w/in boundry 

        # collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2
            ) 
        F = F + -(1/tau) * (F-Feq)

        if(it%plot_every == 0):
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]   # curl of x 
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]   # curl of y
            curl = dfydx - dfxdy
            pyplot.imshow(curl, cmap="bwr")
            # pyplot.imshow(np.sqrt(ux**2+uy**2))
            pyplot.pause(0.01)
            pyplot.cla()

if __name__ == "__main__":
    main()