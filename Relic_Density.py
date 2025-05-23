import numpy as np
from scipy.special import kv
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Define constants
y = np.sqrt(3/5) * (16/77)
mu = md = mc = ms = me = mmu = mtau = mb = 0
mt = 173
mw = 80
mzz = 91
mh = 125
gB = 0.40049
mpl = 1.22e19
g = 4

# s-channel dark matter annihilation cross section to standard model fermions
def sigma_m(mf, Cvf, Caf, mx, mz, Nc, Ng):
    s = 4 * mx**2  # Assuming s = 4*mx^2 for center-of-mass energy, adjust if needed
    return Nc * Ng * (gB**4 * np.sqrt(s - 4 * mf**2) * (2 * mx**2 + s) * 
                     (Caf**2 * (s - 4 * mf**2) + Cvf**2 * (2 * mf**2 + s))) / \
           (1728 * np.pi * s * np.sqrt(s - 4 * mx**2) * (mz**4 + mz**2 * (Gamma(mx, mz)**2 - 2 * s) + s**2))

# Properties of specific final state fermions
fermions2 = [
    [173, (2/3) + (10/12)*y, (1/2)*y, None, None, 3, 1],  # Top quark
    [0, (2/3) + (10/12)*y, (1/2)*y, None, None, 3, 2],    # Up-type quarks
    [0, (2/3) - (1/6)*y, (-1/2)*y, None, None, 3, 3],     # Down-type quarks
    [0, (-3/2)*y, (-1/2)*y, None, None, 1, 3],            # Leptons
    [0, (-1/2)*y, (1/2)*y, None, None, 1, 3]              # Neutrinos
]

# Total cross section
def sigma_mT(s, mx, mz):
    # Sum over fermions
    total = 0
    for i in range(5):
        fermions2[i][3] = mx  # Set mx
        fermions2[i][4] = mz  # Set mz
        total += sigma_m(*fermions2[i])
    # The output expression from Mathematica's Simplify is used directly
    return (gB**4 * (2 * mx**2 + s) * (
        s**(3/2) * (103 * y**2 + 28 * y + 40) +
        np.sqrt(s - 119716) * s * (17 * y**2 + 20 * y + 8) +
        29929 * np.sqrt(s - 119716) * (7 * y**2 + 40 * y + 16)
    )) / (10368 * np.pi * s * np.sqrt(s - 4 * mx**2) * (mz**4 + mz**2 * (Gamma(mx, mz)**2 - 2 * s) + s**2))

# Total decay width of the gauge boson
def Gamma(mx, mz):
    return (1 / (864 * np.pi * mz**2)) * gB**2 * (
        162 * (mz**2)**(3/2) * y**2 +
        9 * (mz**2)**(3/2) * (5 * y**2 - 4 * y + 8) +
        6 * (mz**2)**(3/2) * (17 * y**2 + 20 * y + 8) +
        3 * np.sqrt(mz**2 - 119716) * (mz**2 * (17 * y**2 + 20 * y + 8) + 29929 * (7 * y**2 + 40 * y + 16)) +
        2 * np.sqrt(mz**2 - 15210000) * (mz**2 + 7605000)
    )

# Bessel function approximations for large values
def LargeBessel2(w):
    return np.exp(2 * w) * (-15 / (2 * np.pi) - 2115 / (64 * np.pi * w**2) + 285 / (16 * np.pi * w) + (2 * w) / np.pi)

def LargeBessel1(w):
    return np.exp(-w) * ((3 * np.sqrt(np.pi / 2)) / (8 * w**(3/2)) + np.sqrt(np.pi / 2) / np.sqrt(w))

# Piecewise Bessel functions
def BetterBessel1(w):
    return kv(1, w) if w < 100 else LargeBessel1(w)

def BetterBessel2(w):
    return 1 / (kv(2, w)**2) if w < 100 else LargeBessel2(w)

# Thermally averaged cross section times relative velocity (with temperature)
def f(T, mx, mz):
    integrand = lambda s: sigma_mT(s, mx, mz) * (s - 1) * np.sqrt(s) * BetterBessel1(2 * mx * np.sqrt(s) / T)
    result, _ = quad(integrand, 1, np.inf, epsabs=1e-5, epsrel=1e-5)
    return 4 * mx**2 * np.real(result)

# Thermally averaged cross section in terms of x = m/T
def z(x, mx, mz):
    integrand = lambda s: sigma_mT(s, mx, mz) * (s - 1) * np.sqrt(s) * BetterBessel1(2 * np.sqrt(s) * x)
    result, _ = quad(integrand, 1, np.inf, epsabs=1e-5, epsrel=1e-5)
    return 4 * mx**2 * np.real(result)

# Relativistic degrees of freedom
def gr(T, mx):
    return (7/8) * (
        12 * np.heaviside(T - mu, 1) +
        12 * np.heaviside(T - md, 1) +
        12 * np.heaviside(T - mc, 1) +
        12 * np.heaviside(T - ms, 1) +
        12 * np.heaviside(T - mt, 1) +
        12 * np.heaviside(T - mb, 1) +
        4 * np.heaviside(T - me, 1) +
        4 * np.heaviside(T - mmu, 1) +
        4 * np.heaviside(T - mtau, 1) +
        12
    ) + 4 * np.heaviside(T - mx, 1) + 16 + 2 + 6 * np.heaviside(T - mw, 1) + 3 * np.heaviside(T - mzz, 1) + np.heaviside(T - mh, 1)

# Freeze-out temperature
def FreezeOut(mx, mz):
    def equation(T):
        return 2 * ((mx * T / (2 * np.pi))**(3/2)) * np.exp(-mx / T) * (1 / (8 * mx**4 * T)) * \
               BetterBessel2(mx / T) * (8 * mx**3) * f(T, mx, mz) - 1.66 * np.sqrt(gr(T, mx)) * (T**2) / mpl
    result = root_scalar(equation, bracket=[1, 1000], x0=200)
    return result.root

# Entropy density propagation
def chill(mx, mz):
    integrand = lambda x: (np.sqrt(gr(mx / x, mx)) / x) * BetterBessel2(x) * z(x, mx, mz)
    result, _ = quad(integrand, mx / FreezeOut(mx, mz), np.inf, epsabs=1e-5, epsrel=1e-5, limit=300)
    return (1 / (mx**2)) * result

# Entropy density at present day
def Yd(mx, mz):
    T_fo = FreezeOut(mx, mz)
    term1 = 1 / (0.145 * (g / gr(T_fo, mx)) * ((mx / T_fo)**(3/2)) * np.exp(-mx / T_fo))
    term2 = (1/2) * np.sqrt(np.pi / 45) * mx * mpl * chill(mx, mz)
    return 1 / (term1 + term2)

# Dark matter relic density
def Omega(mx, mz):
    return (2.8e8) * Yd(mx, mz) * mx

# Generate relic density data for different gauge boson masses
# 4 TeV gauge boson, mx from 1500 to 2500 GeV
MyList = []
for i in range(51):
    mx = 1500 + ((2500 - 1500) / 50) * i
    MyList.append([mx, Omega(mx, 4000)])

# 6 TeV gauge boson, mx from 2500 to 3500 GeV
MyList2 = []
for i in range(51):
    mx = 2500 + ((3500 - 2500) / 50) * i
    MyList2.append([mx, Omega(mx, 6000)])

# 8 TeV gauge boson, mx from 3500 to 4500 GeV
MyList3 = []
for i in range(51):
    mx = 3500 + ((4500 - 3500) / 50) * i
    MyList3.append([mx, Omega(mx, 8000)])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot([x[0] for x in MyList], [x[1] for x in MyList], label='4 TeV')
plt.plot([x[0] for x in MyList2], [x[1] for x in MyList2], label='6 TeV')
plt.plot([x[0] for x in MyList3], [x[1] for x in MyList3], label='8 TeV')

# Observed relic density band
x_range = np.linspace(1000, 5000, 100)
plt.fill_between(x_range, 0.1172, 0.1226, color='gray', alpha=0.3, label='Observed Ωh²')

plt.xlim(1000, 5000)
plt.ylim(0, 0.5)
plt.grid(True)
plt.xlabel('mχ (GeV)')
plt.ylabel('Ωh²')
plt.legend()
plt.show()