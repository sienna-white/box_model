import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xarray as xr
import matplotlib.colors as mcolors
import cmocean as cmo
dst = xr.open_dataset("population_dataset_time.nc")

N = len(dst.kappa)
NT = len(dst.t)
print(N)

print(dst)


def field_at_point(R, H, kappa, m1, m2, d1, d2, n1):
    h1 = R*H 
    h2 = H - h1     
    hr2s = 1/3600
    wm = 0.5 * hr2s
    wd = 0.05 * hr2s

    n2 = 1e-5

    Ld = 0.006 * hr2s
    Lm  = 0.004 * hr2s
    growth_m = 0.05 * hr2s
    growth_d = 0.008 * hr2s
         
    #       advection        diffusion       loss      growth
    f1 =  (wm/h1)*m2  + kappa/h1 * (m2 - m1) - Lm*m1 + growth_m*n1*m1    # surface Microcystis
    f2 = (-wm/h2)*m2  + kappa/h2 * (m1 - m2) - Lm*m2                     # bottom Microcystis
    f3 = -wd/h1*d1    + kappa/h1 * (d2 - d1) - Ld*d1 + growth_d*n1*d1      # surface diatoms
    f4 =  wd/h2*d1    + kappa/h2 * (d1 - d2) - Ld*d2                       # bottom diatoms 
    f5 = kappa/h1 * (n2 - n1) - (m1*growth_m)*n1  - (d1*growth_d)*n1 

    return f1, f2, f3, f4, f5


id = 4 
ik = 13
ir = 5 

depth = dst.depth.values[id]
ratio = dst.ratio.values[ir]
kappa = dst.kappa.values[ik]

m1 = dst.m1.values[:, id ,ir,ik]
d1 = dst.m2.values[:, id ,ir,ik]

print(max(m1), max(d1))
print("depth = %2.2f, ratio = %2.2f, kappa = %1.0e" % (depth, ratio, kappa))

NN = 3

MM = 1
x = np.linspace(0, MM, NN)
y = np.linspace(0, MM, NN)
X, Y = np.meshgrid(x, y)


for it in range(0, 300, 3):
    print(it)
    d1 = dst.d1.values[it, id ,ir,ik]
    d2 = dst.d2.values[it, id ,ir,ik] # (time, depth, ratio, kappa) 
    m1 = dst.m1.values[it, id ,ir,ik]
    m2 = dst.m2.values[it, id ,ir,ik]
    n1 = dst.n1.values[it, id ,ir,ik]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharey=False, sharex=False)
    txt = "Depth = %2.2f m, R = %2.1f, Kappa = %1.0e" % (depth, ratio, kappa)
    fig.suptitle("Depth = %2.2f m, R = %2.1f, Kappa = %1.0e" % (depth, ratio, kappa))
    axs = axs.flatten()

    U, V = np.zeros_like(X), np.zeros_like(Y)
    # Compute derivatives on grid
    for i in range(NN):
        for j in range(NN):
                f1, f2, f3, f4, f5 = field_at_point(ratio, depth, kappa, X[i,j], m2, Y[i,j], d2, n1)
                U[i,j] = f1
                V[i,j] = f3

    gs = axs[1].get_gridspec()
    axs[1].remove()
    axs[2].remove()
    axbig = fig.add_subplot(gs[1:3])
    ax2 = axbig.twinx()

    biomass_m = dst.m1.values[:, id ,ir,ik] + dst.m2.values[:, id ,ir,ik]
    biomass_d = dst.d1.values[:, id ,ir,ik] + dst.d2.values[:, id ,ir,ik]
    ax2.plot(dst.t.values/3600, dst.n1.values[:, id ,ir,ik], label=r"Surface nutrients", color='#BF0F0F', linewidth=3)
    axbig.plot(dst.t.values/3600, biomass_m, label=r"Microcystis", color='#0F9BF2', linewidth=3)
    axbig.plot(dst.t.values/3600, biomass_d, label=r"Diatoms", color='#5CA612', linewidth=3)
    axbig.plot(dst.t.values[it]/3600, biomass_m[it] , 'o', color='#0F9BF2')
    axbig.plot(dst.t.values[it]/3600, biomass_d[it] , 'o', color='#5CA612')
    ax2.plot(dst.t.values[it]/3600, dst.n1.values[it, id ,ir,ik] , 'o', color='#BF0F0F')

    axbig.plot([], [], color='#BF0F0F', label=r"Surface nutrients")

    ax2.set_ylabel("Surface nutrients")
    axbig.set_ylabel("Biomass")
    axbig.grid(alpha=0.2)
    axbig.set_xlabel("Time (hours)")
    axbig.set_xlim(0, dst.t.values[-1]/3600)
    axbig.legend(loc='upper left')

    axs[0].grid(alpha=0.2)

    axs[0].plot(dst.m1.values[0, id ,ir,ik], dst.d1.values[0, id ,ir,ik], '^', color='k', alpha=0.3, linewidth=4)
    axs[0].plot(dst.m1.values[0:it, id ,ir,ik], dst.d1.values[0:it, id ,ir,ik], '-', color='#5CA612', alpha=0.3, linewidth=4)
    axs[0].plot(m1, d1, 'o', color='#5CA612', markersize=10)
    axs[0].streamplot(x, y, U, V,  color="gray")
    axs[0].plot([0, MM], [0, MM], '--', color='black', alpha=0.15)
    axs[0].set_xlabel(r"$M_1$")
    axs[0].set_ylabel(r"$D_1$")
    plt.tight_layout()
    fig.savefig("gif/t_%03d.png" % it, dpi=150)
    plt.close()


print("Done!")