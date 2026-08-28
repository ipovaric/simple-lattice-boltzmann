import numpy as np
from matplotlib import pyplot

def main():
    Nx = 400
    Ny = 100
    tau = 0.53 # kinematic viscosity and time scale
    Nt = 3000

    # lattice speeds and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3    # init flow rate

    # obstacle
    cylinder = np.full((Ny, Nx), False)     # init a mask for entire domain

if __name__ == "__main__":
    main()