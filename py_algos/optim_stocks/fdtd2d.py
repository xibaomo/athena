import numpy as np
import matplotlib.pyplot as plt

# Parameters
Nx, Ny = 200, 200
dx = dy = 1e-3
c0 = 299792458
mu0 = 4e-7 * np.pi
eps0 = 1 / (mu0 * c0**2)
dt = 0.99 / (c0 * np.sqrt((1/dx**2 + 1/dy**2)))
Nt = 1000
npml = 20  # PML thickness

# Wave vector
theta = np.pi / 4  # 45 degrees
kx = np.cos(theta)
ky = np.sin(theta)

# Initialize fields
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

# cPML parameters
sigma_max = 1.0
pml_order = 3
Ezx, Ezy = np.zeros_like(Ez), np.zeros_like(Ez)
psi_Ezx, psi_Ezy = np.zeros_like(Ez), np.zeros_like(Ez)

# PML conductivity profile
def pml_sigma(n, N, npml, sigma_max, order):
    d = npml - n
    return sigma_max * (d / npml)**order if d > 0 else 0

sigma_x = np.array([pml_sigma(i, Nx, npml, sigma_max, pml_order) for i in range(Nx)])
sigma_y = np.array([pml_sigma(j, Ny, npml, sigma_max, pml_order) for j in range(Ny)])
sigma_x_2D, sigma_y_2D = np.meshgrid(sigma_y, sigma_x)  # note order!

# Main loop
for n in range(Nt):
    # Update H
    Hx[:, :-1] -= dt / (mu0 * dy) * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += dt / (mu0 * dx) * (Ez[1:, :] - Ez[:-1, :])

    # TFSF source
    t = n * dt
    x0, y0 = Nx // 2, Ny // 2
    delay = 50
    width = 15
    pulse = np.exp(-((t - delay * dt) / (width * dt))**2)
    i = np.arange(Nx)
    j = np.arange(Ny)
    I, J = np.meshgrid(i, j, indexing='ij')
    phase = 2 * np.pi * (I * dx * kx + J * dy * ky) / (c0 * dt)
    Ez += pulse * np.sin(phase)

    # cPML updates
    curl_H = (Hy[1:, :] - Hy[:-1, :]) / dx - (Hx[:, 1:] - Hx[:, :-1]) / dy
    curl_H = np.pad(curl_H, ((0,1), (0,1)))  # pad back to (Nx, Ny)

    # update auxiliary variables for Ez
    psi_Ezx = psi_Ezx * np.exp(-sigma_x_2D * dt / eps0) + curl_H
    psi_Ezy = psi_Ezy * np.exp(-sigma_y_2D * dt / eps0) + curl_H

    Ez += dt / eps0 * (curl_H + psi_Ezx + psi_Ezy)

    # PEC object (scatterer)
    Ez[80:120, 80:120] = 0

    if n % 50 == 0:
        plt.clf()
        plt.imshow(Ez.T, cmap='RdBu', origin='lower')
        plt.title(f'Time step {n}')
        plt.colorbar()
        plt.pause(0.01)

plt.show()
