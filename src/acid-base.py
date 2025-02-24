import numpy as np
import numpy.typing as npt
from scipy.interpolate import pchip_interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from utils import Geometry, Conditions, SolverOptions, SimulationResults
from utils import data_dir, plot_dir, calc_eq_conc
from solver import Solver, load_solver

# Data types
NumpyFloatArray = npt.NDArray[np.float64]

# *************************************************************************************************
# Configure matplotlib environment
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 10
mpl.rcParams['text.usetex'] = True
preamble: str = r'\usepackage{amsmath}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# *************************************************************************************************
def sweep_voltage(g: Geometry, cond: Conditions, opt: SolverOptions, voltage: np.ndarray):
    """Run the simulation on a series of voltages"""
    # Number of voltages to sweep
    n: int = len(voltage)
    # Concentration at inlet
    c0: NumpyFloatArray = np.zeros((n, 5), dtype=np.float64)
    # Flow rate
    flow_rate: NumpyFloatArray = np.full_like(voltage, cond.Q, dtype=np.float64) 
    # Array of currents; in mA
    current: NumpyFloatArray = np.zeros_like(voltage, dtype=np.float64)
    # Array of current densities; in mA / cm^2
    curr_dens: NumpyFloatArray = np.zeros_like(voltage, dtype=np.float64)
    # Array of crossover ratios
    crossover: NumpyFloatArray = np.zeros_like(voltage, dtype=np.float64)

    # Save the original voltage
    V0_orig: float = cond.V0

    # Status
    print('Running voltage sweep for the following voltages:')
    print(voltage)

    # Loop over the voltages
    for i, V0 in enumerate(voltage):
        # Copy the concentrations c0 from the original conditions
        c0[i, :] = cond.c0
        # Update the voltage
        cond.V0 = V0
        # File name
        V0_i: int = int(V0 * 1000)
        fname: str = f'sweep_voltage_v{V0_i:06d}.npz'
        path: Path = Path(data_dir / fname)
        # Load the solver if it's already been solved, otherwise create a new instance and solve it
        if path.exists():
            # Load the solver state from file
            s: Solver = load_solver(fname)
        else:
            # Create a new solver instance
            s: Solver = Solver(g=g, cond=cond, opt=opt)
            # Solve the system with multigrid solver
            s.solve()
            # Save the final state
            s.save(fname=fname)
        # Save the current in mA
        current[i] = s.total_current_mA
        # Save the current density in mA / cm^2
        curr_dens[i] = s.total_curr_dens_mA_cm2
        # Save the crossover ratio
        crossover[i] = s.crossover

    # Restore the original voltage
    cond.V0 = V0_orig

    # Wrap results in a SimulationResults instance
    res: SimulationResults = SimulationResults(n=n, c0=c0, flow_rate=flow_rate, 
        voltage=voltage, current=current, curr_dens=curr_dens, crossover=crossover)
    return res

# *************************************************************************************************
def sweep_flow(g: Geometry, cond: Conditions, opt: SolverOptions, flow_rate: np.ndarray):
    """Run the simulation on a series of flow rates"""
    # Number of flow rates to sweep
    n: int = len(flow_rate)
    # Concentration at inlet
    c0: NumpyFloatArray = np.zeros((n, 5), dtype=np.float64)
    # Voltage
    voltage: NumpyFloatArray = np.full_like(flow_rate, cond.Q, dtype=np.float64) 
    # Array of currents; in mA
    current: NumpyFloatArray = np.zeros_like(flow_rate, dtype=np.float64)
    # Array of current densities; in mA / cm^2
    curr_dens: NumpyFloatArray = np.zeros_like(flow_rate, dtype=np.float64)
    # Array of crossover ratios
    crossover: NumpyFloatArray = np.zeros_like(flow_rate, dtype=np.float64)

    # Save the original flow rate
    Q_orig: float = cond.Q

    # Status
    print('Running flow rate sweep for the following flow rates:')
    print(flow_rate)

    # Loop over the voltages
    for i, Q in enumerate(flow_rate):
        # Copy the concentrations c0 from the original conditions
        c0[i, :] = cond.c0
        # Update the flow rate
        cond.Q = Q
        # File name
        Q_i: int = int(Q * 1000)
        fname: str = f'sweep_flow_q{Q_i:06d}.npz'
        path: Path = Path(data_dir / fname)
        # Load the solver if it's already been solved, otherwise create a new instance and solve it
        if path.exists():
            # Load the solver state from file
            s: Solver = load_solver(fname)
        else:
            # Create a new solver instance
            s: Solver = Solver(g=g, cond=cond, opt=opt)
            # Solve the system
            s.solve()
            # Save the final state
            s.save(fname=fname)
        # Save the current in mA
        current[i] = s.total_current_mA
        # Save the current density in mA / cm^2
        curr_dens[i] = s.total_curr_dens_mA_cm2
        # Save the crossover ratio
        crossover[i] = s.crossover

    # Restore the original flow rate
    cond.Q = Q_orig

    # Wrap results in a SimulationResults instance
    res: SimulationResults = SimulationResults(n=n, c0=c0, flow_rate=flow_rate, 
        voltage=voltage, current=current, curr_dens=curr_dens, crossover=crossover)
    return res

# *************************************************************************************************
def plot_voltage_sweep(res: SimulationResults):
    """Plot the results of a voltage sweep"""
    # Plot settings
    figsize: tuple[int, int] = (12, 8)
    color: str = 'blue'
    marker: str = '.'
    linewidth: int = 2
    markersize: int = 10

    # Plot the current density vs. voltage
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Current vs. Voltage')
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Current (mA)')
    ax.plot(res.voltage, res.current, color=color, linewidth=linewidth, marker=marker, markersize=markersize)
    fpath: Path = plot_dir / '32-current_vs_voltage.png'
    fig.savefig(fpath)

    # Plot the current density vs. voltage
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Current Density vs. Voltage')
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel(r'Current Density ($\text{mA}/\text{cm}^2$)')
    ax.plot(res.voltage, res.curr_dens, color=color, linewidth=linewidth, marker=marker, markersize=markersize)
    fpath: Path = plot_dir / '33-curr_dens_vs_voltage.png'
    fig.savefig(fpath)

    # Plot the crossover vs. current density
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Crossover vs. Current Density')
    ax.set_xlabel(r'Current Density ($\text{mA}/\text{cm}^2$)')
    ax.set_ylabel('Crossover Ratio')
    ax.plot(res.curr_dens, res.crossover, color=color, linewidth=linewidth, marker=marker, markersize=markersize)
    # ax.set_xlim(0, 250)
    fpath: Path = plot_dir / '34-xover_vs_current.png'
    fig.savefig(fpath)

# *************************************************************************************************
def plot_flow_sweep(res: SimulationResults):
    """Plot the results of a flow rate sweep"""
    # Plot settings
    figsize: tuple[int, int] = (12, 8)
    color: str = 'blue'
    marker: str = '.'
    linewidth: int = 2
    markersize: int = 10

    # Plot the crossover vs. flow rate
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Crossover vs. Flow Rate')
    ax.set_xlabel(r'Flow Rate ($\text{mL}/\text{min}$)')
    ax.set_ylabel('Crossover Ratio')
    ax.plot(res.flow_rate, res.crossover, color=color, linewidth=linewidth, marker=marker, markersize=markersize)
    # ax.set_xlim(0, 250)
    fpath: Path = plot_dir / '24-xover_vs_flow_rate.png'
    fig.savefig(fpath)

# *************************************************************************************************
def main():
    """Run the simulation"""
    # Create data and output directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Problem dimensions in cm
    Lx: float = 11.18
    Ly: float = 0.10
    Lz: float = 0.4472

    # Flow rate in mL / minute
    Q: float = 10.40
    # Total concentration of sodium ions in mM
    Na: float = 2000.0
    # Total concentration of acetate in mM
    a0: float = 2000.0
    verbose: bool = False
    # Concentrations of species at equilibrium of a mixture of sodium acetate and acetic acid
    c0: NumpyFloatArray = calc_eq_conc(Na=Na, a0=a0, verbose=verbose)
    # Applied voltage in millivolts
    V0: float = 109.00
    V0_i: int = int(V0)
    # Fraction of positive charge on the anode due to protons
    H_frac: float = 1.0
    # OH concentration in mM in the cathode
    OH_cathode: float = 0.0

    # Number of grid points in x direction
    m: int = 256
    # Number of grid points in y direction
    n: int = 64
    # CFL factor
    cfl: float = 0.500
    # Maximum change in concentration in a time step; in Mol / L
    max_chng: float = 0.25
    # Steady state tolerance
    tol: float = 1.0E-4

    # Impose electro-neutrality?
    do_neutralize: bool = False

    # Maximum number of time steps
    nstep_max: int = 200000
    # Maximum number of stagnant steps before early termination
    nstagnant_max: int = 10000
    # Interval for plots
    plot_int: int = 0
    # Interval for saving checkpoints
    save_int: int = 0
    # Interval for summary to console
    summary_int: int = 10000
    # Interval to print a dot to console
    dot_int: int = 100

    # Build geometry
    g: Geometry = Geometry(Lx=Lx, Ly=Ly, Lz=Lz)

    # Build operating conditions
    cond: Conditions = Conditions(c0=c0, Q=Q, V0=V0, H_frac=H_frac, OH_cathode=OH_cathode)

    # Build solver options
    opt: SolverOptions = SolverOptions(m=m, n=n, cfl=cfl, max_chng=max_chng, 
        nstep_max=nstep_max, nstagnant_max=nstagnant_max, tol=tol, 
        do_neutralize=do_neutralize, plot_int=plot_int, save_int=save_int, summary_int=summary_int, dot_int=dot_int)

    # Initialize solver instance
    s: Solver = Solver(g=g, cond=cond, opt=opt)

    # Name of file to load or solve
    V0_i: int = int(V0)
    fname: str = f'NaA_v{V0_i:03d}.npz'
    path: Path = Path(data_dir / fname)

    # Load solver state from a file if it exists
    if path.exists():
        print(f'Loading solver state from file {path}...')
        s = load_solver(fname)
    # Otherwise, solve the instance and save it
    else:
        print(f'Solver state not found in {path}. Proceeding to solve...')
        s.solve()
        s.save(fname=fname)

    # Print summary of the final state
    s.summarize()

    # Plot the final state - outputs used in publication figures
    add_title: bool = True
    is_final: bool = True
    s.plot_frame(add_title=add_title, is_final=is_final)

# *************************************************************************************************
if __name__ == '__main__':
    main()
