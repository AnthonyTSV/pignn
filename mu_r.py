import numpy as np
import matplotlib.pyplot as plt

T_c = 770
const = 20 # experimental value from team 36
T1 = T_c + const * np.log(0.9)
T2 = T1 + 0.1 * const * np.log(0.1)
relative_permeability_at_20 = {
    "0": 500,
    "500": 350,
    "1000": 500,
    "1500": 600,
    "2000": 525,
    "2500": 450,
    "3000": 390,
    "4000": 305,
    "8000": 164,
    "15900": 89.2,
    "39900": 39.7,
    "79700": 21,
    "239100": 7.8,
    "318800": 6.1,
    "357700": 5.5,
    "398500": 5.1,
    "477000": 4.4,
    "557000": 3.8,
} # H: mu_r_20

def temp_func(T):
    if T < T1:
        return 1 - np.exp((T - T_c) / const)
    else:
        return np.exp((10*(T2 - T)) / const)

def intepolated_relative_permeability_at_20(H):
    H_values = np.array(list(relative_permeability_at_20.keys()), dtype=float)
    mu_r_values = np.array(list(relative_permeability_at_20.values()), dtype=float)
    return np.interp(H, H_values, mu_r_values)

def temp_dependent_relative_permeability(T, H):
    return 1 + temp_func(T) * (intepolated_relative_permeability_at_20(H) - 1)

def magnetic_flux_density(H):
    mu_0 = 4 * 3.1415926535e-7
    mu_r_i = 600
    B_s = 2.05
    a = 0.5
    H_a = mu_0 * H * (mu_r_i - 1) / B_s
    return mu_0 * H + B_s * (H_a + 1 - np.sqrt((H_a + 1)**2 - 4 * H_a * (1 - a))) / (2 * (1 - a))

def cenos_temp_func(T):
    if T < T_c:
        return 1 - (T / T_c)**6
    else:
        return 0

def cenos_mu(T, H):
    mu_0 = 4 * 3.1415926535e-7
    b = magnetic_flux_density(H)
    mu_r = b / ( H * mu_0)
    return 1 + (mu_r - 1) * cenos_temp_func(T)

if __name__ == "__main__":
    print(magnetic_flux_density(50e3))
    # temp_values = np.linspace(20, 800, 100)
    # mu_values = [temp_dependent_relative_permeability(T, 160e3) for T in temp_values]
    # cenos_mu_values = [cenos_mu(T, 160e3) for T in temp_values]
    # plt.plot(temp_values, mu_values, label="TEAM 36")
    # plt.plot(temp_values, cenos_mu_values, label="CENOS")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    h_values = np.array(list(relative_permeability_at_20.keys()), dtype=float)
    b_values = magnetic_flux_density(h_values).astype(float)
    dict_values = {h: b for h, b in zip(h_values, b_values)}
    print(dict_values)
    # plt.plot(h_values, b_values)
    # plt.xlabel("H (A/m)")
    # plt.ylabel("B (T)")
    # plt.grid(True)
    # plt.show()