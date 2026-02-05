import numpy as np 


hr2s = 1/3600
day2s = 1/(3600*24)


mol_p = 30.97 # g/mol molar mass for p 
umol_p = mol_p * 1e-6 # g/umol molar mass for p


# Settling / swimming velocities
wm = 0.5 * hr2s
wd = 0.05 * hr2s

# Respiration  
Ld  = 0.008 * hr2s # 0.006
Lm  = 0.004 * hr2s

# half saturation for growth 
gamma_m = 0.18 # half saturation constant for microcystis growth
gamma_d = 0.097 # half saturation constant for diatom growth

# maximum uptake rates 
uptake_d = 4.16e-12 # nutrient uptake rate for diatoms
uptake_m = 2.23e-12 # [µmol P / cell s] nutrient uptake rate for microcystis

# half saturation for uptake 
gamma_nm = 1.23    # [µmol P / L] half saturation constant for nutrient uptake microcystis
gamma_nd = 2.8     # [µmol P / L] half saturation constant for nutrient uptake diatoms    
        # ^ (0.7 - 2.8)

# Growth
# alpha = 0.008 * hr2s
alpha = 1 * day2s # 0.27 per day 0.27/24
beta = 2 * day2s    # 2 per day 


n2 = 4

parameters = {}
parameters["hr2s"] = hr2s
parameters["n2"] = n2
parameters["wm"] = wm
parameters["wd"] = wd
parameters["Lm"] = Lm
parameters["Ld"] = Ld
parameters["alpha"] = alpha
parameters["beta"] = beta
parameters["gamma_m"] = gamma_m
parameters["gamma_d"] = gamma_d
parameters["uptake_m"] = uptake_m
parameters["uptake_d"] = uptake_d
parameters["gamma_nm"] = gamma_nm
parameters["gamma_nd"] = gamma_nd



def jacobian_2s(kappa, h1, h2, m1, m2, d1, d2, n1):

    monod_growth_d = beta  * n1 / (gamma_d + n1)
    monod_growth_m = alpha * n1 / (gamma_m + n1)

    monod_uptake_m = uptake_m * n1 / (gamma_nm + n1)

    # Row 1
    df1_dm1 = (-kappa/h1 + monod_growth_m - Lm)
    df1_dm2 = (kappa/h1 + wm/h1)
    df1_dn1 = alpha*m1*gamma_m/(gamma_m + n1)**2

    # Row 2 
    df2_dm1 = kappa/h2
    df2_dm2 = (-wm/h2 - kappa/h2 - Lm)

    # Row 3
    df3_dd1 = -wd/h1  - kappa/h1 + monod_growth_d - Ld
    df3_dd2 = kappa/h1 
    df3_dn1 = beta*d1*gamma_d/(gamma_d + n1)**2

    # Row 4 
    df4_dd1 = wd/h2 + kappa/h2
    df4_dd2 = -Ld -  kappa/h2

    # Row 5 
    df5_dm1 = - uptake_m * (n1 / (n1 + gamma_nm))
    df5_dd1 = - uptake_d * (n1 / (n1 + gamma_nd))
    df5_dn1 = -kappa/h1 - uptake_m*m1 * gamma_nm/(gamma_nm + n1)**2 - uptake_d * d1 * gamma_nd/(gamma_nd + n1)**2 

    # Jacobian matrix
    j = np.array([[df1_dm1,   df1_dm2,       0,          0,     df1_dn1 ],
                  [df2_dm1,   df2_dm2,       0,          0,     0       ],
                  [0,         0,       df3_dd1,    df3_dd2,     df3_dn1 ],
                  [0,         0,       df4_dd1,    df4_dd2,     0       ],
                  [df5_dm1,   0,       df5_dd1,          0,     df5_dn1]])

    eigenvalues, eigenvectors = np.linalg.eig(j)

    # Sort by eigenvalue 
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    return (eigenvalues, eigenvectors)  



def jacobian_1s(kappa, h1, h2, m1, m2, n1):

    monod_growth_m = alpha * n1 / (gamma_m + n1)
    monod_uptake_m = uptake_m * n1 / (gamma_nm + n1)

    # Row 1
    df1_dm1 = (-kappa/h1 + monod_growth_m - Lm)
    df1_dm2 = (kappa/h1 + wm/h1)
    df1_dn1 = alpha*m1*gamma_m/(gamma_m + n1)**2

    # Row 2 
    df2_dm1 = kappa/h2
    df2_dm2 = (-wm/h2 - kappa/h2 - Lm)


    # Row 5 
    df5_dm1 = - uptake_m * (n1 / (n1 + gamma_nm))
    df5_dd1 = - uptake_d * (n1 / (n1 + gamma_nd))
    df5_dn1 = -kappa/h1 - uptake_m*m1 * gamma_nm/(gamma_nm + n1)**2 

    # Jacobian matrix
    j = np.array([[df1_dm1,   df1_dm2,        df1_dn1 ],
                  [df2_dm1,   df2_dm2,        0       ],
                  [df5_dm1,   0,              df5_dn1]])

    eigenvalues, eigenvectors = np.linalg.eig(j)

    # Sort by eigenvalue 
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    return (eigenvalues, eigenvectors)  

def field_at_point_2s(R, H, kappa, m1, m2, d1, d2, n1, n2):
    h1 = R*H 
    h2 = H - h1     
    #       advection        diffusion       loss      growth
    f1 =  (wm/h1)*m2  + kappa/h1 * (m2 - m1) - Lm*m1 + alpha*m1*(n1/(gamma_m + n1))   # surface Microcystis
    f2 = (-wm/h2)*m2  + kappa/h2 * (m1 - m2) - Lm*m2                                  # bottom Microcystis
    f3 = -wd/h1*d1  + kappa/h1 * (d2 - d1) - Ld*d1 + beta*d1*(n1/(gamma_d + n1))      # surface diatoms
    f4 = wd/h2*d1   + kappa/h2 * (d1 - d2) - Ld*d2                                    # bottom diatoms 
    f5 = kappa/h1 * (n2 - n1) - uptake_m*m1*(n1/(n1+gamma_nm)) - uptake_d*d1*(n1/(n1+gamma_nd))     # surface nutrients 
    
    return f1, f2, f3, f4, f5

def steady_state_2s(R, H, kappa, d2, m2, n1):
    d1_ss = d2*kappa*(gamma_d + n1)/(H*Ld*R*gamma_d + H*Ld*R*n1 - H*R*beta*n1 + gamma_d*kappa + gamma_d*wd + kappa*n1 + n1*wd)
    m1_ss = m2*(gamma_m*kappa + gamma_m*wm + kappa*n1 + n1*wm)/(H*Lm*R*gamma_m + H*Lm*R*n1 - H*R*alpha*n1 + gamma_m*kappa + kappa*n1)
    return d1_ss, m1_ss

#         (d2*gamma_d*kappa + d2*kappa*n1)/(H*Ld*R*gamma_d + H*Ld*R*n1 - H*R*growth_d*n1 + gamma_d*kappa + gamma_d*wd + kappa*n1 + n1*wd)
#     m1: (gamma_m*kappa*m2 + gamma_m*m2*wm + kappa*m2*n1 + m2*n1*wm)/(H*Lm*R*gamma_m + H*Lm*R*n1 - H*R*growth_m*n1 + gamma_m*kappa + kappa*n1)}]
# Solution 1: