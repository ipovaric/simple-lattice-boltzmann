import numpy as np
from matplotlib import pyplot

def main():
    Nx = 400
    Ny = 100
    tau = 0.53 # kinematic viscosity and time scale
    Nt = 3000

    # lattice speeds and weights
    NL = 9
    cxs = np.array[[0, 0, 1, 1,  1,  0, -1, -1, -1]]
    cys = np.array[[0, 1, 1, 0, -1, -1, -1,  0,  1]]

if __name__ == "__main__":
    main()