import numpy as np
import numpy.typing as npt
from scipy.special import expit
from scipy.optimize import root_scalar
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Data types
NumpyFloatArray = npt.NDArray[np.float64]

# *************************************************************************************************
# Physical constants
# Faraday's constant in C/mol
F: float = 96485.332
# Ideal gas constant in J/(mol*K)
R: float = 8.3145
# Temperature in K
T: float = 298.15

# Equilibrium constant for water in M^2 (molarity squared)
pKw: float = 14.0
Kw_std: float = 10.0 ** (-pKw)

# Acid dissociation constant for acetic acid in M
pKa: float = 4.76
Ka_std: float = 10.0 ** (-pKa)

# Convert Kw and Ka into millimolar based units
# Kw in mM^2
Kw: float = Kw_std * 1.0E6
# Ka in mM
Ka: float = Ka_std * 1.0E3

# Number of species in the system
ns: int = 5

# Conversion factor from cm to meters
cm2m: float = 1.0E-2
# Conversion factor for diffusion coefficients from cm^2/s to m^2/s
D_factor: float = cm2m * cm2m

# Diffusion coefficients
# All coefficients are quoted in cm^2/s and converted to m^2/s (multipy by 10^-4)

# Diffusivity of protons in water
D_H: float = 9.30E-5 * D_factor
# Assume the diffusivity of hydroxide the same as protons, which are really hydronium ions H3O+
D_OH: float = D_H
# Diffusivity of acetic acid in water
D_HA: float = 1.11E-5 * D_factor
# Assume that the diffusivity of acetate is the same as acetic acid
D_A: float = D_HA
# Diffusivity of sodium ions in water
D_Na: float = 1.33E-5 * D_factor
# Bundle the diffusion coefficients
D: NumpyFloatArray = np.array([D_H, D_Na, D_HA, D_A, D_OH])

# Charge number z for each species
charge_num: NumpyFloatArray = np.array([1, 1, 0, -1, -1], dtype=np.float64)
# Prefactor for kappa in S / m
F2_over_RT: float = (F * F) / (R * T)
kappa_fact: NumpyFloatArray = F2_over_RT * charge_num * charge_num * D

# Permeability of HA throught the membrane in m/s; convert quoted value in cm/s to m/s
perm_HA: float = 0.0

# Crossover coefficients for OH-; used in solver.py, placed here for easy access
crossover_OH_alpha: float = 7.810E-8
crossover_OH_beta: float = 5.143E-11

# *************************************************************************************************
# Directory for figures
plot_dir: Path = Path('figs')
# Directory for saved data
data_dir: Path = Path('data')

# Number of digits in file names
ndig: int = 6

# Keywords for matplotlib plotting
mpl_kwargs: dict = {
    'dpi': 2400,
    'format': 'png',
    'bbox_inches': 'tight'
}

# *************************************************************************************************
@dataclass 
class Geometry:
    """Class to hold the geometry of the problem"""
    # Length in the flow direction; cm
    Lx: float
    # Width; distance between two electrodes; cm
    Ly: float
    # Nominal height; cm
    Lz: float

    def __init__(self, Lx: float, Ly: float, Lz: float):
        """Initialize the geometry"""
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

# *************************************************************************************************
@dataclass
class Conditions:
    """Class to hold the operating conditions of the acid problem"""
    # Concentration of all species in mM at the inlet; H+, Na+, OH-, HA, A-
    c0: NumpyFloatArray
    # Flow rate in mL / minute
    Q: float
    # Potential difference between the two plates in millivolts
    V0: float
    # Fraction of positive charge on the anode due to protons
    H_frac: float
    # OH concentration in the cathode in mM
    OH_cathode: float

# *************************************************************************************************
@dataclass
class SolverOptions:
    """Class to hold the solver options"""
    # Number of grid points in the x direction (parallel to flow, orthogonal to ion transport)
    m: int
    # Number of grid points in the y direction (orthogonal to flow, aligned to ion transport)
    n: int
    # Maximum value of n for multigrid solver
    n_max: int = 0

    # Dimensionless CFL factor for sizing time steps
    cfl: float = 0.5
    # Maximum concentration change in a time step in mM for convergence
    max_chng: float = 1.0E-4

    # Should the solver impose electroneutrality?
    do_neutralize: bool = False

    # Maximum number of time steps in simulation
    nstep_max: int = 100000
    # Maximum number of stagnant steps before early termination
    nstagnant_max: int = 1000
    # Tolerance for convergence
    tol: float = 1.0E-6

    # Interval for plots
    plot_int: int = 0
    # Interval for saving checkpoints
    save_int: int = 0
    # Interval for summary to console
    summary_int: int = 20000
    # Interval to print a dot to console
    dot_int: int = 100

    def __post_init__(self):
        if self.n_max == 0:
            self.n_max = self.n

# No argument function to create an empty array; used as default factory in SimulationResults dataclass
def empty_array() -> NumpyFloatArray:
    return np.zeros(0)

# *************************************************************************************************
@dataclass
class SimulationResults:
    """Results of simulations; arrays for plotting and analysis"""
    # The number of results; must be consistent for all arrays
    n: int = 0
    # The total concentration at the inlet; shape (n, 5)
    c0: NumpyFloatArray = field(default_factory=empty_array)
    # The flow rate in mL / minute
    flow_rate: NumpyFloatArray = field(default_factory=empty_array)
    # The applied voltage in mV
    voltage: NumpyFloatArray = field(default_factory=empty_array)
    # The total current in mA
    current: NumpyFloatArray = field(default_factory=empty_array)
    # The total current density in mA / cm^2
    curr_dens: NumpyFloatArray = field(default_factory=empty_array)
    # The crossover ratio; dimensionless
    crossover: NumpyFloatArray = field(default_factory=empty_array)
    # The file name
    fname: str = ''

    def __post_init__(self):
        if self.fname != '':
            self.load(self.fname)
        elif self.n == 0:
            raise ValueError('SimulationResults must have a non-zero number of results.')

    def save(self, fname: str) -> None:
        """Save the simulation results to a file"""
        path = data_dir / fname
        array_tbl = {
            'n':            self.n,
            'c0':           self.c0,
            'flow_rate':    self.flow_rate,
            'voltage':      self.voltage,
            'current':      self.current,
            'curr_dens':    self.curr_dens,
            'crossover':    self.crossover,
        }
        np.savez(path, **array_tbl)

    def load(self, fname: str) -> None:
        """Load the simulation results from a file"""
        path = data_dir / fname
        data = np.load(path)
        self.n          = data['n']
        self.c0         = data['c0']
        self.flow_rate  = data['flow_rate']
        self.voltage    = data['voltage']
        self.current    = data['current']
        self.curr_dens  = data['curr_dens']
        self.crossover  = data['crossover']
        print(f'Loaded SimulationResults data in {fname:s}.')

# *************************************************************************************************
@dataclass
class PlotOptions:
    """Options for plotting one frame"""
    fname: str
    figsize: tuple[float, float] = (8.0, 6.0)
    title: Optional[str] = None
    xticks: Optional[NumpyFloatArray] = None
    yticks: Optional[NumpyFloatArray] = None
    xticklabels: Optional[list[str]] = None
    yticklabels: Optional[list[str]] = None

# *************************************************************************************************
def calc_eq_conc(Na: float, a0: float, verbose: bool) -> NumpyFloatArray:
    """
    Calculate equilibrium concentrations of all species in a mixture of sodium acetate and acetic acid
    Na: float - total concentration of sodium ions in mM
    a0: float - total concentration of acetate in mM; sum of NaA and HA
        [Na+] = Na
        [A-] + [HA] = a0
    """    
    # Subfunction to calculate the charge given a fraction x of the total acetate that is HA
    def f(t: float) -> float:
        x: float = expit(t)
        HA: float = x * a0
        A: float = (1.0 - x) * a0
        H: float = (Ka * HA) / A
        OH: float = Kw / H
        return (Na + H) - (A + OH)

    # Find t = logit(x) where x is the fraction of acetate that is HA with a root-finding algorithm
    x0: float = 0.0
    rtol: float = 1.0E-15
    sol = root_scalar(f=f, method='newton', x0=x0, rtol=rtol)
    t: float = sol.root
    x: float = expit(t)
    HA: float = x * a0
    A: float = (1.0 - x) * a0
    H: float = (Ka * HA) / A
    OH: float = Kw / H
    # Do a final tweak to impose electroneutrality by adjusting Na
    Na = A + OH - H
    # Wrap up the concentrations in an array
    c0: NumpyFloatArray = np.array([H, Na, HA, A, OH])
    if verbose:
        charge = (Na + H) - (A + OH)
        pH = -np.log10(H * 1.0E-3)
        print('Equilibrium species concentrations:')
        print(f'Na      = {Na:.6f} (total sodium concentration in mM)')
        print(f'a0      = {a0:.6f} (total acetate concentration in mM)')
        print(f'pH      = {pH:.6f}')
        print(f'charge  = {charge:5.3e}')
        species_list = ['H+', 'Na+', 'HA', 'A-', 'OH-']
        for i in range(5):
            print(f'{species_list[i]:5s} = {c0[i]:15.9f}')
        outcome = 'successfully' if sol.converged else 'unsuccessfully'
        print(f'Converged {outcome:s} in {sol.iterations} iterations.\n')
    # Return the concentrations in an array
    return c0

# *************************************************************************************************
def calc_eq_conc_pH(pH: float, a0: float) -> NumpyFloatArray:
    """Calculate equilibrium concentrations of all species given pH and total acetate concentration a0"""
    # [H+] from pH; need to convert to millimolar!
    H: float = pow(10.0, -pH) * 1.0E3
    # [HA] from HA acid equilibrium
    HA: float = (a0 * H ) / (H + Ka)
    # [A-] from HA acid equilibrium
    A = (a0 * Ka) / (H + Ka)
    # [OH-] from H2O equilibrium
    OH: float = Kw / H
    # [Na+] from electroneutrality
    Na: float = A + OH - H
    # Wrap up the concentrations in an array
    return np.array([H, Na, HA, A, OH])

