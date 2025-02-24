import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from pathlib import Path
from typing import Optional

from utils import Geometry, Conditions, SolverOptions, PlotOptions
from utils import Ka, Kw, ns, D, charge_num, kappa_fact, F, R, T, \
    perm_HA, crossover_OH_alpha, crossover_OH_beta, \
    data_dir, plot_dir, ndig, mpl_kwargs

# Data types
NumpyFloatArray = npt.NDArray[np.float64]
Float64 = np.float64

# Default figure file type - from mpl_kwargs
ftyp: str = mpl_kwargs['format']

# Default figure size
figsize_dflt: tuple[float, float] = (10.0, 2.0)

# *************************************************************************************************
class Solver:
    """Class to hold the solver state; also includes the geometry and conditions"""

    # *************************************************************************************************
    def __init__(self, g: Geometry, cond: Conditions, opt: SolverOptions):
        """Initialize the solver"""
        # All units in calculations are SI. Quoted units are converted to SI.

        # Conversion factor from cm to m
        cm2m: float = 0.01

        # Copy the geometry and scale from cm to m
        self.Lx: float = g.Lx * cm2m
        self.Ly: float = g.Ly * cm2m
        self.Lz: float = g.Lz * cm2m

        # Copy the operating conditions
        # Concentrations are in mM; no need to convert these
        self.c0: NumpyFloatArray = cond.c0
        # Scale flow rate from mL / minute to m^3 / sec
        self.Q: float = cond.Q * 1.0E-6 / 60.0
        # Potential difference in V; convert from mV
        self.V0: float = cond.V0 * 1.0E-3
        # Electric field in V / m - assuming a uniform field
        self.E0: float = self.V0 / self.Ly
        # Fraction of positive charge on the anode due to protons
        self.H_frac: float = cond.H_frac
        # Concentration of OH- at the cathode; mM
        self.OH_cathode: float = cond.OH_cathode

        # Total concentration of acetate (HA + A-) in millimolar at the inlet
        self.acetate: float = float(np.sum(self.c0[2:4]))
        # The pH of the solution at the inlet; need to convert from mM to M first
        self.pH: float = -np.log10(self.c0[0] * 1.0E-3)
        # The mean concentration at the outlet; calculated in calc_outlet_conc
        self.c1: NumpyFloatArray = np.zeros(ns)

        # Unpack the solver options
        # Dimensionless CFL factor
        self.cfl: float = opt.cfl
        # Maximum concentration change in a time step in mM
        self.max_chng: float = opt.max_chng
        # Maximum number of time steps in simulation; alias for legibility
        self.nstep_max: int = opt.nstep_max
        # Maximum number of stagnant steps
        self.nstagnant_max: int = opt.nstagnant_max
        # Tolerance for convergence
        self.tol: float = opt.tol
        # Impose electroneutrality?
        self.do_neutralize: bool = opt.do_neutralize
        # Interval for plots
        self.plot_int: int = opt.plot_int
        # Interval for saving checkpoints
        self.save_int: int = opt.save_int
        # Interval for summary to console
        self.summary_int: int = opt.summary_int
        # Interval to print a dot to console
        self.dot_int: int = opt.dot_int
        # Maximum number of grid points in the y direction for multigrid solver
        self.n_max: int = opt.n_max

        # Number of grid points in the x direction
        self.m: int = opt.m
        # Number of grid points in the y direction
        self.n: int = opt.n
        # The step size in x; meters
        self.hx: float = self.Lx / self.m
        # The step size in y; meters
        self.hy: float = self.Ly / self.n

        # Calculate the velocity field in m/s
        self.calc_u(verbose=False)

        # Time for fluid to flow through the system; seconds
        volume: float = self.Lx * self.Ly * self.Lz
        self.tau: float = volume / self.Q

        # Shape of the concentration field is (ns, m, n)
        shape_conc: tuple[int, int, int] = (ns, self.m, self.n)

        # Concentration field with all species; mM. Species in order are: H+, Na+, HA, A-, OH-
        self.conc: NumpyFloatArray = np.zeros(shape_conc)
        # Concentration at previous time step
        self.conc_prev: NumpyFloatArray = np.zeros(shape_conc)
        # Initialize concentrations to match the inlet
        self.conc[:,:,:] = self.c0.reshape((ns, 1, 1))
        self.conc_prev[:,:,:] = self.c0.reshape((ns, 1, 1))

        # Shape of 2D fields is (m, n); 2D fields are cell centered by default (e.g. concentration)
        shape_2d: tuple[int, int] = (self.m, self.n)
        # Shape of 2D fields on grid nodes is (m+1, n+1)
        shape_2d_nodal: tuple[int, int] = (self.m+1, self.n+1)

        # Electric potential phi in volts; stored at nodes so size is (m+1, n+1)
        self.phi: NumpyFloatArray = np.zeros(shape_2d_nodal)
        # Initialize phi to a uniform electric field oriented along the y-axis
        self.phi[:,:] = np.linspace(self.V0, 0.0, self.n+1).reshape((1, self.n+1))
        # Charge density in C / m^3 on a nodal grid aligned with phi
        self.Q_dens: NumpyFloatArray = np.zeros(shape_2d_nodal)
        
        # Electric field in V / m along the x axis; cell centered
        self.Ex: NumpyFloatArray = np.zeros(shape_2d)
        # Electric field in V / m along the y axis; cell centered
        self.Ey: NumpyFloatArray = np.zeros(shape_2d)
        # Initialize Ey to the uniform electric field E0 oriented along the y-axis
        self.Ey[:] = self.E0

        # Electrical conductivity kappa in S / m
        self.kappa: NumpyFloatArray = np.zeros(shape_2d)
        # Electrical resistivity rho in Ohm-m
        self.rho: NumpyFloatArray = np.zeros(shape_2d)
        # Current density within in A / m^2
        self.curr_dens: NumpyFloatArray = np.ones(shape_2d)

        # Fraction of total current due to H+ at the top
        self.frac_H_top: NumpyFloatArray = np.zeros(self.m)
        # Fraction of total current due to Na+ at the top
        self.frac_Na_top: NumpyFloatArray = np.zeros(self.m)

        # Advection rate; mM / sec
        self.advection: NumpyFloatArray = np.zeros(shape_conc)
        # Diffusion rate; mM / sec
        self.diffusion: NumpyFloatArray = np.zeros(shape_conc)
        # Electromigration rate; mM / sec
        self.migration: NumpyFloatArray = np.zeros(shape_conc)
        # Total rate of change of concentration in mM / sec
        self.dC_dt: NumpyFloatArray = np.zeros(shape_conc)

        # Positive charge density; C / m^3
        self.Q_pos: NumpyFloatArray = np.zeros(shape_2d)
        # Negative charge density; C / m^3
        self.Q_neg: NumpyFloatArray = np.zeros(shape_2d)
        # The net charge density before neutrality is imposed; C / m^3
        self.Q_net: NumpyFloatArray = np.zeros(shape_2d)
        # Charge density imposed during electroneutrality; C / m^3
        self.Q_abs: NumpyFloatArray = np.zeros(shape_2d)

        # Length of time step in seconds from CFL condition
        u_max: float = np.max(np.abs(self.u))
        self.dt_cfl = (self.hx / u_max) * self.cfl
        # Initial time step in seconds
        self.dt = self.dt_cfl

        # Number of time steps taken
        self.nstep: int = 0
        # Number of stagnant steps taken
        self.nstagnant: int = 0

        # Total elapsed time in the simulation
        self.t_sim: float = 0.0

        # Wall time when simulation started
        self.t0: float = time.time()

        # RMS change in concentration per time tau; full history over simulation steps
        self.rms_chng: NumpyFloatArray = np.zeros(self.nstep_max)
        # RMS of dC_dt; difference between rms_chng and rms_dC_dt is that the former includes 
        # the effects of fast equilibration and electroneutrality.
        self.rms_dC_dt: NumpyFloatArray = np.zeros(self.nstep_max)

        # Total current in of H+ at the bottom
        self.current_bot: float = 0.0
        # Total current of H+ at the top
        self.current_top_H: float = 0.0
        # Total current of Na+ at the top
        self.current_top_Na: float = 0.0
        # Total current at the top (sum of H+ and Na+)
        self.current_top: float = 0.0
        # Total current at the outlet
        self.total_current_out: float = 0.0

        # Best estimate of the current; average of current in and out; in mA
        self.total_current_mA: float = 0.0
        # Best estimate of the current density in mA / cm^2
        self.total_curr_dens_mA_cm2: float = 0.0
        # Total resistance in Ohms
        self.resistance: float = 0.0
        # Area specific resistance in Ohm cm^2
        self.asr: float = 0.0

        # Crossover ratio due to H+ through the top
        self.crossover_H: float = 0.0
        # Crossover ratio due to HA diffusing through the top
        self.crossover_HA: float = 0.0
        # Crossover ratio of OH- transported through the top CEM (diffusion + electromigration)
        self.crossover_OH: float = 0.0
        # Total crossover ratio of H+ to total current out
        self.crossover: float = 0.0

        # Initialize the current density and related fields
        self.calc_current()

    # *************************************************************************************************
    def upsample(self, f: int):
        """Upsample all spatial fields by a factor of s"""
        # All units in calculations are SI. Quoted units are converted to SI.

        # Increase m and n by a factor of f
        self.m *= f
        self.n *= f
        # Decrease the step sizes hx and hy by a factor of f
        self.hx /= f
        self.hy /= f

        # Recalculate the velocity field in m/s
        self.calc_u(verbose=False)

        # Recalculate CFL time step
        u_max: float = np.max(np.abs(self.u))
        self.dt_cfl = (self.hx / u_max) * self.cfl
        self.dt = min(self.dt, self.dt_cfl)

        # Argument to kronecker delta function np.kron used to upsample 2d spatial fields
        kron_2d = np.ones((f, f))
        # Argument to kronecker delta function np.kron used to upsample 3d concentration fields
        kron_3d = np.ones((1, f, f))

        # Concentration fields
        self.conc = np.kron(self.conc, kron_3d)
        self.conc_prev = np.kron(self.conc_prev, kron_3d)

        # Electrical spatial fields
        self.phi = np.kron(self.phi, kron_2d)
        self.kappa = np.kron(self.kappa, kron_2d)
        self.rho = np.kron(self.rho, kron_2d)
        self.curr_dens = np.kron(self.curr_dens, kron_2d)

        # Fraction of current due to H+ and Na at the top
        self.frac_H_top = np.kron(self.frac_H_top, kron_2d)
        self.frac_Na_top = np.kron(self.frac_Na_top, kron_2d)

        # Mass transport components
        self.advection = np.kron(self.advection, kron_3d)
        self.diffusion = np.kron(self.diffusion, kron_3d)
        self.migration = np.kron(self.migration, kron_3d)
        self.dC_dt = np.kron(self.dC_dt, kron_3d)

        # Charge density fields
        self.Q_pos = np.kron(self.Q_pos, kron_2d)
        self.Q_neg = np.kron(self.Q_neg, kron_2d)
        self.Q_net = np.kron(self.Q_net, kron_2d)
        self.Q_abs = np.kron(self.Q_abs, kron_2d)

        # Update the current density and related fields
        self.calc_current()

    # *************************************************************************************************
    def save(self, fname: Optional[str] = None):
        """Save the state of the solver to a file"""
        V0_i: int = int(self.V0 * 1.0E6)
        ac_i: int = int(self.acetate)
        fname_dflt: str = f'sim_v{V0_i:06d}_a{ac_i:03d}_n{self.n:02d}_step{self.nstep:0{ndig}d}.npz'
        fname_: str = fname or fname_dflt
        path: Path = data_dir / fname_
        array_tbl = {
            # The geometry
            'Lx'                : self.Lx,
            'Ly'                : self.Ly,
            'Lz'                : self.Lz,
            # The operating conditions
            'Q'                 : self.Q,
            'pH'                : self.pH,
            'acetate'           : self.acetate,
            'c0'                : self.c0,
            'V0'                : self.V0,
            'H_frac'            : self.H_frac,
            'OH_cathode'        : self.OH_cathode,
            # The solver options
            'm'                 : self.m,
            'n'                 : self.n,
            'cfl'               : self.cfl,
            'max_chng'          : self.max_chng,
            'nstep_max'         : self.nstep_max,
            # Solver internal state
            'conc'              : self.conc,
            'rms_chng'          : self.rms_chng,
            'nstep'             : self.nstep,
            # Key outputs
            'total_current_mA'          : self.total_current_mA,
            'total_curr_dens_mA_cm2'    : self.total_curr_dens_mA_cm2,
            'crossover'                 : self.crossover,
        }
        np.savez(path, **array_tbl)

    # *************************************************************************************************
    def load_state(self, fname: str):
        """Load the internal state of the solver from a file"""
        path = data_dir / fname
        data = np.load(path)
        self.conc  = data['conc']
        self.rms_chng = data['rms_chng']
        # Need to subtract one from nstep to reflect the number of completed steps
        self.nstep = max(data['nstep'] - 1, 0)
        self.total_current_mA = data['total_current_mA']
        self.total_curr_dens_mA_cm2 = data['total_curr_dens_mA_cm2']
        self.crossover = data['crossover']
        print(f'Loaded internal state of {fname}.')

    # *************************************************************************************************
    def wall_time(self, verbose) -> float:
        # The elapsed time
        et: float = time.time() - self.t0
        # Report results if requested
        if verbose:
            print(f'Wall time: {et:5.2f} s')
        return et

    # *************************************************************************************************
    def calc_u(self, verbose: bool):
        """Calculate the velocity field assuming a Poiseuille flow profile in the x direction"""
        # Surface area transverse to flow in m^2; flux in x direction so this is the YZ plane
        A_yz: float = self.Ly * self.Lz

        # Average velocity in m / s
        u_bar: float = self.Q / A_yz

        # y fraction at midpoint
        t = np.linspace(0, 1.0, self.n+1)
        # Powers of yt
        t2 = t * t
        t3 = t2 * t
        # u_int is the integral of the velocity profile
        u_int = 6.0 * self.n * u_bar * (t2 / 2.0 - t3 / 3.0)
        # u is the mean velocity in each cell
        self.u: NumpyFloatArray = np.diff(u_int).reshape((1, self.n))

        # Report results if requested
        if verbose:
            u_bar_cmps = u_bar * 100.0
            u_mean_cmps = np.mean(self.u) * 100.0
            print('Fluid velocity summary:')
            print(f'u_bar   : {u_bar_cmps:5.2e} cm/s')
            print(f'Mean(u) : {u_mean_cmps:5.2e} cm/s\n')

    # *************************************************************************************************
    def calc_advection(self):
        """Calculate the advection term in place"""
        # Alias members for legibility
        m: int = self.m
        n: int = self.n
        hx_inv: float = 1.0 / self.hx

        # Calculate the advection term in mM / s to the right of the inlet
        # Look to the left - need to do upwinding and flow is left to right
        self.advection[0:ns, 1:m, 0:n] = - self.u.reshape(1, 1, n) * hx_inv * \
            np.diff(self.conc[0:ns, 0:m, 0:n], axis=1)

        # Note that advection[:, 0, :] is zero; Dirichlet boundary condition at the inlet

    # *************************************************************************************************
    def calc_diffusion(self):
        """Calculate the advection term in place"""
        # Alias members for legibility
        m: int = self.m
        n: int = self.n
        # Inverse step size and its square in the x direction
        hx_inv: float = 1.0 / self.hx
        hx2_inv: float = hx_inv * hx_inv
        # Inverse step size and its square in the y direction
        hy_inv: float = 1.0 / self.hy
        hy2_inv: float = hy_inv * hy_inv

        # Prefactor for diffusion term - diffusion coefficient divided by step size squared
        # 0D version is has shape (ns)
        D_fac_x_0d: NumpyFloatArray = D * hx2_inv
        D_fac_y_0d: NumpyFloatArray = D * hy2_inv
        # 1D version has shape (ns, 1)
        D_fac_x_1d: NumpyFloatArray = D_fac_x_0d.reshape(ns, 1)
        D_fac_y_1d: NumpyFloatArray = D_fac_y_0d.reshape(ns, 1)
        # 2D version has shape (ns, 1, 1)
        D_fac_x_2d: NumpyFloatArray = D_fac_x_0d.reshape(ns, 1, 1)
        D_fac_y_2d: NumpyFloatArray = D_fac_y_0d.reshape(ns, 1, 1)

        # Diffusion on interior points with a four point stencil
        self.diffusion[:, 1:m-1, 1:n-1] = \
            D_fac_x_2d * (
             self.conc[:, 0:m-2, 1:n-1] + 
             self.conc[:, 2:m-0, 1:n-1] -
             self.conc[:, 1:m-1, 1:n-1] * 2.0) + \
            D_fac_y_2d * (
             self.conc[:, 1:m-1, 0:n-2] + 
             self.conc[:, 1:m-1, 2:n-0] -
             self.conc[:, 1:m-1, 1:n-1] * 2.0)

        # Diffusion on the bottom boundary where j=0
        self.diffusion[:, 1:m-1,  0] = \
            D_fac_x_1d * (
             self.conc[:, 0:m-2,  0] + 
             self.conc[:, 2:m-0,  0] -
             self.conc[:, 1:m-1,  0] * 2.0) + \
            D_fac_y_1d * (
             self.conc[:, 1:m-1,  1] -
             self.conc[:, 1:m-1,  0] )

        # Diffusion on the top boundary where j=n-1
        self.diffusion[:, 1:m-1,n-1] = \
            D_fac_x_1d * (
             self.conc[:, 0:m-2,n-1] + 
             self.conc[:, 2:m-0,n-1] -
             self.conc[:, 1:m-1,n-1] * 2.0) + \
            D_fac_y_1d * (
             self.conc[:, 1:m-1,n-2] -
             self.conc[:, 1:m-1,n-1] )

        # Diffusion on the left boundary where i=0 is always zero because of Dirichlet boundary condition

        # Diffusion on the right boundary where i=m-1
        self.diffusion[:, m-1,  1:n-1] = \
            D_fac_y_1d * (
             self.conc[:, m-1,  0:n-2] + 
             self.conc[:, m-1,  2:n-0] -
             self.conc[:, m-1,  1:n-1] * 2.0) + \
            D_fac_x_1d * (
             self.conc[:, m-2,  1:n-1] - 
             self.conc[:, m-1,  1:n-1] )             

        # Diffusion on the bottom right corner; i=m-1, j=0
        self.diffusion[:, m-1,  0] = \
            D_fac_x_0d * (
             self.conc[:, m-2,  0] -
             self.conc[:, m-1,  0] ) + \
            D_fac_y_0d * (
             self.conc[:, m-1,  1] -
             self.conc[:, m-1,  0] )
                     
        # Diffusion on the top right corner; i=m-1, j=n-1
        self.diffusion[:, m-1, n-1] = \
            D_fac_x_0d * (
                self.conc[:, m-2, n-1] -
                self.conc[:, m-1, n-1] ) + \
            D_fac_y_0d * (
                self.conc[:, m-1, n-2] -
                self.conc[:, m-1, n-1] )

    # *************************************************************************************************
    def calc_migration(self):
        """Calculate the electromigration term in place"""
        # Alias members for legibility
        m: int = self.m
        n: int = self.n
        hy_inv: float = 1.0 / self.hy
        hx_inv: float = 1.0 / self.hx

        # Coefficient for electromigration
        beta: NumpyFloatArray = charge_num * D * (F / (R * T))

        # Prefactor for electromigration 
        M_fac_x: NumpyFloatArray = - beta.reshape(ns, 1, 1) * hx_inv
        M_fac_y: NumpyFloatArray = - beta.reshape(ns, 1, 1) * hy_inv
        # Electromigration on rows above the top row; use upwinding
        # Electromigration is div dot (C * grad phi); grad phi = - (Ex i + Ey j)
        # M_fac_x * np.diff(self.conc[:, :, :] * self.Ex[np.newaxis, :, :], axis=1) +
        self.migration[:, 0:m, 0:n] = 0.0
        self.migration[:, 1:m, 0:n] += M_fac_x * np.diff(self.conc[:, :, :] * self.Ex[np.newaxis, :, :], axis=1)
        self.migration[:, 0:m, 1:n] += M_fac_y * np.diff(self.conc[:, :, :] * self.Ey[np.newaxis, :, :], axis=2)

        # Electromigration on the bottom boundary where j=0. 
        # Model this as replacing Na+ with H+ at the bottom
        # Note that the specific area is 1 / hy

        # Take the NEGATIVE of the migration from the row above - this row is the "source" of the migration
        self.migration[:, :, 0] = -self.migration[:, :, 1]

        # Net charge migration consistent with the current at the bottom (j=0)
        migration_net_bot: NumpyFloatArray = self.curr_dens[:, 0] * (hy_inv / F)
        # Amount of Na+ replaced by H+ at the bottom
        Na_to_H: NumpyFloatArray = migration_net_bot * self.H_frac
        # Replace Na+ with H+ at the bottom depending on the proton fraction at the anode
        self.migration[0, :, 0] += Na_to_H
        self.migration[1, :, 0] -= Na_to_H

    # *************************************************************************************************
    def calc_charge_dens(self):
        """Calculate charge density on the cell-centered grids in C / m^3"""
        # Masks for positive and negative charged species
        mask_pos = (charge_num > 0)
        mask_neg = (charge_num < 0)
        # Charge number for positive and negative species, filtered by mask and reshaped
        charge_num_pos = charge_num[mask_pos].reshape(-1, 1, 1)
        charge_num_neg = charge_num[mask_neg].reshape(-1, 1, 1)
        # The positive charge density in C; the positive species are H+ and Na+
        self.Q_pos[:,:] =  np.sum(self.conc[mask_pos,:,:] * charge_num_pos, axis=0) * F
        # The negative charge in C; the negative species are A- and OH-
        self.Q_neg[:,:] = -np.sum(self.conc[mask_neg,:,:] * charge_num_neg, axis=0) * F
        # The net charge density before neutrality is imposed
        self.Q_net[:,:] = self.Q_pos[:,:] - self.Q_neg[:,:]
        # The absolute charge density imposed by electroneutrality
        self.Q_abs[:,:] = np.sqrt(self.Q_pos[:,:] * self.Q_neg[:,:])

    # *************************************************************************************************
    def neutralize(self):
        """Adjust species concentrations to impose electroneutrality"""
        # Masks for positive and negative charged species
        mask_pos = (charge_num > 0)
        mask_neg = (charge_num < 0)

        # Apply multiplicative factor for positive species
        self.conc[mask_pos, :, :] *= self.Q_abs[:,:] / self.Q_pos[:,:]
        # Apply multiplicative factor for negative species
        self.conc[mask_neg, :, :] *= self.Q_abs[:,:] / self.Q_neg[:,:]

    # *************************************************************************************************
    def equilibrate_impl(self, conc: NumpyFloatArray):
        """Adjust species concentrations to impose fast equilibriria for water and acetic acid"""
        # Solve quadratic equation for how much the following reaction needs to run forward to equilbrium
        # H+ + A- -> HA
        # H = conc[0]
        # A = conc[3]
        # HA = conc[2]
        p: NumpyFloatArray = -(conc[0] + conc[3] + Ka)
        q: NumpyFloatArray = conc[0] * conc[3] - Ka * conc[2]
        # Small root of the quadratic equation; p is negative
        t: NumpyFloatArray = (-p - np.sqrt(p * p - 4.0 * q)) / 2.0
        # Adjust the concentrations to the equilibrium values
        conc[0] -= t
        conc[3] -= t
        conc[2] += t

        # Solve quadratic equation for how much the following reaction needs to run forward to equilbrium
        # H+ + OH- -> H2O
        p = -(conc[0] + conc[4])
        q = conc[0] * conc[4] - Kw
        # Small root of the quadratic equation; p is negative
        t = (-p - np.sqrt(p * p - 4.0 * q)) / 2.0
        # Adjust the concentrations to the equilibrium values
        conc[0] -= t
        conc[4] -= t
        
    # *************************************************************************************************
    def equilibrate(self):
        """Adjust species concentrations to impose fast equilibriria for water and acetic acid"""
        # Delegate to equilibrate_impl for each cell
        self.equilibrate_impl(self.conc)
       
    # *************************************************************************************************
    def calc_current(self):
        """"Calculate electrical terms kappa, rho and curr_dens in place"""
        # Calculate the electrical conductivity kappa
        self.kappa[:,:] = np.sum(self.conc[:,:,:] * kappa_fact.reshape(ns, 1, 1), axis=0)
        # Calculate the electrical resistivity rho
        self.rho[:,:] = 1.0 / self.kappa[:,:]
        # The current density
        self.curr_dens[:,:] = self.E0 * self.kappa[:,:]

    # *************************************************************************************************
    def calc_crossover(self):
        """"Calculate the crossover of H+ at the top and total current at both top and bottom"""
        # Alias members for legibility
        m: int = self.m
        n: int = self.n
        # The cross-sectional area of one cell for the current; current is along y, so this is hx * Lz
        area: float = self.hx * self.Lz
        # The total geometric area in SI units (m^2)
        area_tot: float = self.Lx * self.Lz
        # The total geometric area in cm^2
        area_tot_cm2: float = area_tot * 1.0E4

        # Current due to proton flux in at the bottom; use the row above (j=1) to avoid boundary issues
        self.current_bot = float(np.sum(self.curr_dens[:, 1])) * area

        # Conductivity due to H+ at the top
        kappa_H_top: NumpyFloatArray = kappa_fact[0] * self.conc[0, :, n-1]
        # Conductivity due to Na+ at the top
        kappa_Na_top: NumpyFloatArray = kappa_fact[1] * self.conc[1, :, n-1]
        # Total conductivity due to positive ions at the top
        kappa_tot_top: NumpyFloatArray = kappa_H_top + kappa_Na_top
        # Fraction of total current due to H+ at the top
        self.frac_H_top[:]  = kappa_H_top  / kappa_tot_top
        # Fraction of total current due to Na+ at the top
        self.frac_Na_top[:] = kappa_Na_top / kappa_tot_top

        # Total current at the top; use the second to top row (j=n-2) to avoid boundary issues
        current_top: NumpyFloatArray = self.curr_dens[:, n-2] * area

        # Total current due to proton flux in at the bottom     
        self.current_top_H  = float(np.sum(current_top * self.frac_H_top))
        self.current_top_Na = float(np.sum(current_top * self.frac_Na_top))
        self.current_top = float(np.sum(current_top))

        # Amount of HA diffusing out at the top; in mM / s
        diff_HA_top: float = float(np.sum(self.conc[2, :, n-1] * perm_HA * area))

        # The best estimate of the current in mA; the current at the TOP. 
        # The bottom currrent is low due to Na+ depletion
        self.total_current_mA = self.current_top * 1000.0
        # The best estimate of the current density; in mA / cm^2
        self.total_curr_dens_mA_cm2 = self.total_current_mA / area_tot_cm2

        # The total resistance in Ohms; V = IR so R = V / I. Convert V to mV. I is already in mA.
        self.resistance = (self.V0 * 1.0E3) / self.total_current_mA
        # The area specific resistance in Ohm cm^2. ASR = R * A
        self.asr = self.resistance * area_tot_cm2

        # Mean current density at the top; use the second to top row (j=n-2) to avoid boundary issues
        j_mean: float = float(np.mean(self.curr_dens[:, n-2]))
        # Crossover flux of OH- from the cathode; in mol / m^2 / s
        crossover_flux_OH: float =  crossover_OH_alpha * self.OH_cathode + \
            crossover_OH_beta * self.OH_cathode * j_mean
        # Crossover rate of OH from the cathode; in mol / s
        crossover_rate_OH: float = crossover_flux_OH * area_tot

        # The crossover ratio; should be small 
        xover_den: float = max(self.current_top, 1.0E-12)
        self.crossover_H = self.current_top_H / xover_den
        self.crossover_HA = (diff_HA_top * F) / xover_den
        self.crossover_OH = (crossover_rate_OH * F) / xover_den
        self.crossover = self.crossover_H + self.crossover_HA + self.crossover_OH

    # *************************************************************************************************
    def calc_outlet_conc(self):
        """"Calculate the mean concentration at the outlet"""
        # Alias members for legibility
        m: int = self.m
        n: int = self.n
        # The cross-sectional area of one cell for the outbound flux in m^2; unit face in the YZ plane
        area: float = self.hy * self.Lz
        # Calculate the flux at the outlet in moles / s; concentration x velocity x area is in moles / s
        flux_out: NumpyFloatArray = np.sum(self.conc[:, m-1, :] * self.u.reshape(1, n), axis=1) * area
        # Calculate the total volume out in m^3 / s
        volume_out: float = float(np.sum(self.u)) * area
        # Calculate the mean concentration at the outlet in mM
        self.c1[:] = (1.0 / volume_out) * flux_out
        # Equilibrate this mixture
        self.equilibrate_impl(self.c1)
        # Calculate the total ionic current out in A
        self.total_current_out = float(np.sum(flux_out * charge_num)) * F

    # *************************************************************************************************
    def explicit_time_step(self):
        """Update the concentration field with an explicit time step"""
        # Calculate advection, diffusion and migration terms
        self.calc_advection()
        self.calc_diffusion()
        self.calc_migration()

        # Total rate of change in concentration in mM / sec
        self.dC_dt[:,:,:] = self.advection[:,:,:] + self.diffusion[:,:,:] + self.migration[:,:,:]

        # Calculate time step
        dC_dt_max: float = np.max(np.abs(self.dC_dt))
        dC_dt_max = max(dC_dt_max, 1.0E-6)
        self.dt = min((self.dt_cfl, self.max_chng / dC_dt_max))

        # Copy new to old concentrations
        self.conc_prev[:,:,:] = self.conc[:,:,:]

        # Apply update to concentration from transport equation
        self.conc[:,:,:] += self.dC_dt * self.dt

        # Calculate the charge density
        self.calc_charge_dens()

        # Impose electroneutrality if applicable
        if self.do_neutralize:
            self.neutralize()

        # Impose fast equilibria
        self.equilibrate()

        # Apply Dirichlet boundary condition at the inlet
        self.conc[:, 0, :] = self.c0.reshape(ns, 1)

        # Calculate electrical terms kappa, rho and curr_dens
        self.calc_current()

        # Calculate the crossover and total current
        self.calc_crossover()

        # Calculate dimensionless RMS change in concentration over one step for convergence
        rms_chng_fac = self.tau / (self.dt * self.c0.reshape(ns, 1, 1))
        conc_rel_chng = rms_chng_fac * (self.conc - self.conc_prev)
        self.rms_chng[self.nstep] = np.sqrt(np.mean(np.square(conc_rel_chng)))

        # Calculate dimensionless RMS rate of change in concentrations
        rms_rate_fac = self.tau / self.c0.reshape(ns, 1, 1)
        self.rms_dC_dt[self.nstep] = np.sqrt(np.mean(np.square(self.dC_dt * rms_rate_fac)))
        
    # *************************************************************************************************
    def time_step(self):
        """Update the concentration field with an explicit time step"""
        # Delegate to explicit_time_step
        self.explicit_time_step()
        # Was the last step stagnant?
        if (self.nstep > 1 and self.rms_chng[self.nstep] > self.rms_chng[self.nstep-1]):
            self.nstagnant += 1
        else:
            self.nstagnant = 0
        # Update number of steps and the total elapsed time
        self.nstep += 1
        self.t_sim += self.dt

    # *************************************************************************************************
    def plot_field_impl(self, X: NumpyFloatArray, plot_opt: PlotOptions, imshow_kwargs: dict):
        """Plot a scalar field X with a maximum value vmax and save it to a file called fname"""
        # unpack plot options
        title: Optional[str] = plot_opt.title
        fname: str = plot_opt.fname
        figsize = plot_opt.figsize
        xticks = plot_opt.xticks
        yticks = plot_opt.yticks
        xticklabels = plot_opt.xticklabels
        yticklabels = plot_opt.yticklabels

        # create the plot and save it; X must be transposed for imshow
        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(X.T, **imshow_kwargs)

        # add ticks and labels if applicable
        ax.set_xticks(xticks, xticklabels) # type: ignore
        ax.set_yticks(yticks, yticklabels) # type: ignore

        # build colorbar
        divider = make_axes_locatable(ax)
        size = figsize[0] * 0.040
        pad = figsize[0] * 0.016
        cax = divider.append_axes("right", size=size, pad=pad)
        cbar = fig.colorbar(img, cax=cax)
        cbar.ax.tick_params(labelsize=8)

        # add title if provided
        if title is not None:
            # ax.set_title(title)
            # place title in top left of output rectangle
            tx: float = figsize[0] * 0.05
            ty: float = figsize[1] * 0.95
            ax.text(tx, ty, title, fontsize=10, verticalalignment='top', color='white')

        # save output file to specified location
        fpath = plot_dir / fname
        fig.savefig(fpath, **mpl_kwargs)
        plt.close(fig)

    # *************************************************************************************************
    def plot_field(self, X: NumpyFloatArray, fname: str, figsize: tuple[float, float], title: Optional[str], 
                            vmin: Optional[float], vmax: Optional[float]):
        """Plot a scalar field X with fixed sizing and save it to a file called fname"""
        # Unpack figsize
        figsize_x: float = figsize[0]
        figsize_y: float = figsize[1]
        # The plot extent matches the geometry
        extent = [0.0, figsize_x, 0.0, figsize_y]

        # Tick locations
        xticks: Optional[NumpyFloatArray] = np.array([])
        yticks: Optional[NumpyFloatArray] = np.array([])
        # Tick labels
        xticklabels = []
        yticklabels = []
        # Wrap plot options
        plot_opt = PlotOptions(title=title, fname=fname, figsize=figsize, 
            xticks=xticks, yticks=yticks, xticklabels=xticklabels, yticklabels=yticklabels)

        # Miscellaneous
        origin: str = 'lower'
        interpolation: str = 'bilinear'
        cmap: str = 'jet'
        imshow_kwargs: dict = {
            'extent': extent,
            'origin': origin,
            'interpolation': interpolation,
            'cmap': cmap,
            'vmin': vmin if vmin is not None else np.min(X),
            'vmax': vmax if vmax is not None else np.max(X),
        }

        # Delegate to plot_field_impl
        self.plot_field_impl(X=X, plot_opt=plot_opt, imshow_kwargs=imshow_kwargs)

    # *************************************************************************************************
    def plot_field_native(self, X: NumpyFloatArray, fname: str, title: Optional[str], 
                            vmin: Optional[float], vmax: Optional[float]):
        """Plot a scalar field X with native sizing and save it to a file called fname"""
        # Figure size in inches
        figsize_x = 8.0
        figsize_y = self.Ly / self.Lx * figsize_x
        figsize = (figsize_x, figsize_y)
        # The plot extent matches the geometry
        extent = [0.0, self.Lx, 0.0, self.Ly]

        # Tick locations
        xticks = np.linspace(0.0, self.Lx, 21)
        yticks = np.linspace(0.0, self.Ly, 3)
        # Tick labels
        xticklabels = []
        yticklabels = []
        # Wrap plot options
        plot_opt = PlotOptions(title=title, fname=fname, figsize=figsize, 
            xticks=xticks, yticks=yticks, xticklabels=xticklabels, yticklabels=yticklabels)

        # Miscellaneous
        origin: str = 'lower'
        #interpolation: str = 'bilinear'
        interpolation: str = 'none'
        cmap: str = 'jet'
        imshow_kwargs: dict = {
            'extent': extent,
            'origin': origin,
            'interpolation': interpolation,
            'cmap': cmap,
            'vmin': vmin if vmin is not None else np.min(X),
            'vmax': vmax if vmax is not None else np.max(X),
        }

        # Delegate to plot_field_impl
        self.plot_field_impl(X=X, plot_opt=plot_opt, imshow_kwargs=imshow_kwargs)

    # *************************************************************************************************
    def plot_frame(self, add_title: bool, is_final: bool):
        """Plot the current frame"""
        # Delegate to plot_frame_conc
        self.plot_frame_conc(add_title=add_title, is_final=is_final)
        # Delegate to plot_frame_transport
        self.plot_frame_transport(add_title=add_title, is_final=is_final)

    # *************************************************************************************************
    def plot_frame_conc(self, add_title: bool, is_final: bool):
        """Plot the current frame - concentrations of various species"""
        # Arguments used by plot_field
        X: NumpyFloatArray
        figsize: tuple[float, float] = figsize_dflt
        vmin: Optional[float]
        vmax: Optional[float]
        fname: str
        title: Optional[str]

        # Filename part for the frame number
        frame_str: str = '' if is_final else f'_{self.nstep:0{ndig}d}'
    
        # Units substring for millimolar
        mM: str = r'$\text{mmol} \cdot \text{L}^{-1}$'

        # Plot H concentration
        X: NumpyFloatArray = self.conc[0]
        vmin = 0.0
        vmax = None
        title: Optional[str] = r'$\text{H}^{+}$ Concentration' + f" ({mM})" if add_title else None
        fname: str = f'01_H{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot Na concentration
        X = self.conc[1]
        vmin = None
        vmax = None
        title = r'$\text{Na}^{+}$ Concentration' + f" ({mM})" if add_title else None 
        fname = f'02_Na{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot HA concentration
        X = self.conc[2]
        vmin = None
        vmax = None
        title = r'$\text{HA}$ Concentration' + f" ({mM})" if add_title else None 
        fname = f'03_HA{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot acetate concentration
        X = self.conc[3]
        vmin = None
        vmax = None
        title = r'$\text{A}^{-}$ Concentration' + f" ({mM})" if add_title else None 
        fname = f'04_A{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot OH concentration
        X = self.conc[4]
        vmin = None
        vmax = None
        title = r'$\text{OH}^{-}$ Concentration' + f" ({mM})" if add_title else None 
        fname = f'05_OH{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot pH
        X: NumpyFloatArray = - np.log10(self.conc[0] * 1.0E-3)
        vmin = None
        vmax = None
        title: Optional[str] = 'pH' if add_title else None
        fname: str = f'06_pH{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot total acetate concentration (HA + A-)
        X = self.conc[2] + self.conc[3]
        vmin = None
        vmax = None
        title = r'$\text{HA} + \text{A}^{-}$ Concentration' + f" ({mM})" if add_title else None 
        fname = f'07_A_total{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot sodium fraction ([Na+] / ([Na+] + [H+]))
        X = self.conc[0] / (self.conc[0] + self.conc[1])
        vmin = None
        vmax = None
        title = r'Sodium Fraction (dimensionless)' if add_title else None 
        fname = f'08_Na_frac{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

    # *************************************************************************************************
    def plot_frame_transport(self, add_title: bool, is_final: bool):
        """Plot the current frame - transport quantities"""
        # Arguments used by plot_field
        X: NumpyFloatArray
        figsize: tuple[float, float] = figsize_dflt
        vmin: Optional[float]
        vmax: Optional[float]
        fname: str
        title: Optional[str]

        # Filename part for the frame number
        frame_str: str = '' if is_final else f'_{self.nstep:0{ndig}d}'
        # Quantile range for color scale
        qt_lo: float = 0.001
        qt_hi: float = 0.999
        quantiles: NumpyFloatArray = np.array([qt_lo, qt_hi])

        # Transport of H+: advection, diffusion, electromigration and total

        # Units substring for millimolar per second
        #mMps: str = r'$\text{mM} \cdot \text{s}^{-1}$'
        mMps: str = r'$\text{mmol} \cdot \text{L}^{-1} \cdot \text{s}^{-1}$'        

        # Plot advection of H+
        X = self.advection[0]
        vmin, vmax = np.quantile(X, quantiles)
        title = r'$\text{H}^{+}$ Advection' + f" ({mMps})" if add_title else None 
        fname = f'21_advection_H{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot diffusion of H+
        X = self.diffusion[0]
        vmin, vmax = np.quantile(X[:,1:], quantiles)
        title = r'$\text{H}^{+}$ Diffusion' + f" ({mMps})" if add_title else None 
        fname = f'22_diffusion_H{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot electromigration of H+
        X = self.migration[0]
        # Set scale above the bottom row
        vmin, vmax = np.quantile(X[:,1:], quantiles)
        title = r'$\text{H}^{+}$ Electromigration' + f" ({mMps})" if add_title else None 
        fname = f'23_migration_H{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot rate of change of H+
        X = self.dC_dt[0]
        vmin, vmax = np.quantile(X[:,1:], quantiles)
        title = r'$\text{H}^{+}$ Transport' + f" ({mMps})" if add_title else None 
        fname = f'24_transport_H{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Transport of Na+: advection, diffusion, electromigration and total

        # Plot advection of Na+
        X = self.advection[1]
        vmin, vmax = np.quantile(X, quantiles)
        title = r'$\text{Na}^{+}$ Advection' + f" ({mMps})" if add_title else None 
        fname = f'21_advection_Na{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot diffusion of Na+
        X = self.diffusion[1]
        vmin, vmax = np.quantile(X, quantiles)
        title = r'$\text{Na}^{+}$ Diffusion' + f" ({mMps})" if add_title else None 
        fname = f'22_diffusion_Na{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot electromigration of Na+
        X = self.migration[1]
        vmin, vmax = np.quantile(X, quantiles)
        title = r'$\text{Na}^{+}$ Electromigration' + f" ({mMps})" if add_title else None 
        fname = f'23_migration_Na{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

        # Plot rate of change of Na+
        X = self.dC_dt[1]
        vmin, vmax = np.quantile(X, quantiles)
        title = r'$\text{Na}^{+}$ Transport' + f" ({mMps})" if add_title else None 
        fname = f'24_transport_Na{frame_str}.{ftyp}'
        self.plot_field(X=X, figsize=figsize, vmin=vmin, vmax=vmax, title=title, fname=fname)

    # *************************************************************************************************
    def progress_summary(self):
        """Short summary of progress during solution of the PDEs"""
        # Update current field
        self.calc_current()

        # Simulation and wall time
        wall_time: float = self.wall_time(verbose=False)
        print(f'\nStep {self.nstep:{ndig}d}. t = {self.t_sim:12.6f} s. '
            f'dt = {self.dt:6.3e} s. dt_cfl = {self.dt_cfl:6.3e} s. Wall time = {wall_time:6.0f} s.')

        # Headers for summary
        # print(f"{'quantity':20s} : {'mean':12s} : {'std':12s} : {'min':12s} : {'max':12s} : {'x_mom':12s} : {'y_mom':12s}")
        print(f"{'quantity':20s} : {'mean':12s} : {'std':12s} : {'min':12s} : {'max':12s}")

        # Subfunction - summarize one scalar field
        def summarize_X(X: NumpyFloatArray, name: str, show_moments: bool):
            mean: Float64 = np.mean(X)
            stdev: Float64 = np.std(X)
            min: Float64 = np.min(X)
            max: Float64 = np.max(X)
            # Print summary for this quantity
            print(f"{name:20s} : {mean:12.6f} : {stdev:12.6f}: {min:12.6f} : {max:12.6f}")

        # Summary statistics for H+
        summarize_X(self.conc[0], 'H+', True)

        # Summary statistics for Na+
        summarize_X(self.conc[1], 'Na+', True)

        # Summary statistics for HA 
        summarize_X(self.conc[2], 'HA', True)

        # Summary statistics for A- 
        summarize_X(self.conc[3], 'A-', True)

        # Summary statistics for OH- 
        summarize_X(self.conc[4], 'OH-', True)

        # Summary statistics for electric claculations
        summarize_X(self.rho, 'rho', False)
        summarize_X(self.kappa, 'kappa', False)
        summarize_X(self.curr_dens, 'j', False)

        # Summary statistics for H+ transport
        summarize_X(self.advection[0], 'Advection [H+]', False)
        summarize_X(self.diffusion[0], 'Diffusion [H+]', False)
        summarize_X(self.migration[0], 'Migration [H+]', False)
        summarize_X(self.dC_dt[0], 'd[H+]/dt', False)

        # Summary statistics for sodium transport
        summarize_X(self.advection[1], 'Advection [Na+]', False)
        summarize_X(self.diffusion[1], 'Diffusion [Na+]', False)
        summarize_X(self.migration[1], 'Migration [Na+]', False)
        summarize_X(self.dC_dt[1], 'd[Na+]/dt', False)

        # RMS change this step for convergence testing
        if self.nstep > 0:
            i = self.nstep-1
            print(f'RMS conc change  = {self.rms_chng[i]:12.6e}')
            print(f'RMS dC/dt        = {self.rms_dC_dt[i]:12.6e}')
            print(f'Crossover        = {self.crossover:12.6e}')

    # *************************************************************************************************
    def summarize(self):
        """Print a summary of the solver state"""
        # Dimensions in cm
        Lx_cm: float = self.Lx * 100.0
        Ly_cm: float = self.Ly * 100.0
        Lz_cm: float = self.Lz * 100.0
        # Geometric area in cm^2
        area_cm2: float = Lx_cm * Lz_cm

        # Flow rate in mL / min
        Q_mL_min: float = self.Q * 1.0E6 * 60.0
        # Total concentration of sodium (NaA) in millimolar
        sodium: float = self.c0[1]
        # Total concentration of acetate (HA + A-) in millimolar
        acetate: float = self.acetate
        # The pH at the inlet
        pH_in: float = self.pH
        # Potential difference in millivolts
        V0_mV: float = self.V0 * 1000.0
        # Step size in microns - x
        hx_um: float = self.hx * 1.0E6
        # Step size in microns - y
        hy_um: float = self.hy * 1.0E6

        # Mean velocity quoted in cm / s
        u_bar_cm_s: float = float(np.mean(self.u)) * 100.0
        # Time scale in seconds for fluid to flow through the system
        tau: float = self.tau

        # Calculate concentration at the outlet
        self.calc_outlet_conc()
        pH_out: float = - np.log10(self.c1[0] * 1.0E-3)
        # Calculate net charge density at inlet and outlet in SI units
        charge_dens_in: float = float(np.sum(self.c0 * charge_num)) * F
        charge_dens_out: float = float(np.sum(self.c1 * charge_num)) * F

        # Current entering at the bottom; entirely due to H+; in mA
        current_bot_mA = self.current_bot * 1000.0
        # Current exiting at the top; due to H+ and Na+; in mA
        current_top_mA = self.current_top * 1000.0
        # Current at the outlet
        current_out_mA = self.total_current_out * 1000.0
        # Total inbound current
        current_in_tot_mA = current_bot_mA
        # Total outbound current
        current_out_tot_mA = current_top_mA + current_out_mA
        # Relative mismatch in current; dimensionless
        mismatch = 1.0 - current_in_tot_mA / current_out_tot_mA

        # Whether the solver has converged or terminated early
        termination_type: str
        if self.nstep < self.nstep_max and self.nstagnant < self.nstagnant_max:
            termination_type = 'converged'
        elif self.nstagnant >= self.nstagnant_max:
            termination_type = 'terminated early (stagnant steps)'
        else:
            termination_type = 'terminated early (total steps)'

        print('Geometry:')
        print(f'Lx          : {Lx_cm:8.3f} cm')
        print(f'Ly          : {Ly_cm:8.3f} cm')
        print(f'Lz          : {Lz_cm:8.3f} cm')
        print(f'Area        : {area_cm2:8.3f} cm^2')
        print('\nOperating Conditions:')
        print(f'sodium      : {sodium:8.3f} millimolar')
        print(f'acetate     : {acetate:8.3f} millimolar')
        print(f'Q           : {Q_mL_min:8.3f} mL/minute')
        print(f'V0          : {V0_mV:8.3f} millivolts')
        print('\nConcentration at inlet (mM):')
        print(f'H+          : {self.c0[0]:15.9f}')
        print(f'Na+         : {self.c0[1]:15.9f}')
        print(f'HA          : {self.c0[2]:15.9f}')
        print(f'A-          : {self.c0[3]:15.9f}')
        print(f'OH-         : {self.c0[4]:15.9f}')
        print(f'pH          : {pH_in:8.3f}')
        print(f'charge dens : {charge_dens_in:8.3e} C/m^3')
        print('\nConcentration at outlet (mM):')
        print(f'H+          : {self.c1[0]:15.9f}')
        print(f'Na+         : {self.c1[1]:15.9f}')
        print(f'HA          : {self.c1[2]:15.9f}')
        print(f'A-          : {self.c1[3]:15.9f}')
        print(f'OH-         : {self.c1[4]:15.9f}')
        print(f'pH          : {pH_out:8.3f}')
        print(f'charge dens : {charge_dens_out:8.3e} C/m^3')
        print('\nFlow Calculations:')
        print(f'u_bar       : {u_bar_cm_s:8.3f} cm/s')
        print(f'tau         : {tau:8.3f} s')
        print('\nSolver Parameters:')
        print(f'm           : {self.m}')
        print(f'n           : {self.n}')
        print(f'hx          : {hx_um:5.2e} um')
        print(f'hy          : {hy_um:5.2e} um')
        print(f'dt          : {self.dt:5.2e} s')
        print(f'cfl         : {self.cfl:8.6f} (dimensionless)')
        print(f'max_chng    : {self.max_chng:8.6f} mM')
        print(f'nstep_max   : {self.nstep_max}')
        print('\nSolver Convergence:')
        print(f'Termination Type: {termination_type} at step {self.nstep}.')
        print(f'RMS change      : {self.rms_chng[self.nstep-1]:12.3e} (dimensionless)')
        print(f'RMS transport   : {self.rms_dC_dt[self.nstep-1]:12.3e} (dimensionless)')
        print(f'Current Mismatch: {mismatch:12.3e} (dimensionless)')
        print(f'Current bottom  : {current_bot_mA:12.3f} mA')
        print(f'Current top     : {current_top_mA:12.3f} mA')
        print(f'Current out     : {current_out_mA:12.3f} mA')
        print('\nSimulation Outputs - Electrical:')
        print(f'Current         : {self.total_current_mA:12.3f} mA')
        print(f'Current Density : {self.total_curr_dens_mA_cm2:12.3f} mA/cm^2')
        print(f'Resistance      : {self.resistance:12.3f} Ohms')
        print(f'ASR             : {self.asr:12.3f} Ohm cm^2')
        print('\nSimulation Outputs - Crossover:')
        print(f'Crossover - H   : {self.crossover_H:12.3e} (dimensionless)')
        print(f'Crossover - HA  : {self.crossover_HA:12.3e} (dimensionless)')
        print(f'Crossover - OH  : {self.crossover_OH:12.3e} (dimensionless)')
        print(f'Crossover       : {self.crossover:12.3e} (dimensionless)')
        print('')

    # *************************************************************************************************
    def solve_impl(self, tol: float, plot_int: int, save_int: int, summary_int: int, dot_int: int):
        """Run simulation to steady state"""
        # Don't add titles to plots except at the end
        add_title: bool = False
        # Intermediate plots are not final
        is_final: bool = False

        # Plot the velocity field
        u: NumpyFloatArray = self.u * 100.0
        figsize: tuple[float, float] = figsize_dflt
        vmax: float = float(np.max(u))
        title: str = 'Velocity Field'
        self.plot_field(X=u, figsize=figsize, vmin=0.0, vmax=vmax, title=title, fname=f'00_u.{ftyp}')

        # Main time stepping loop
        while (self.nstep < self.nstep_max):
            # Alias nstep to i for legibility
            i = self.nstep
            # Print a dot to console
            if (i % dot_int == 0 and i > 0):
                print('.', end='', flush=True)
            # Plot a frame every plot_int steps
            if (plot_int > 0 and i % plot_int == 0 and i > 0):
                self.plot_frame(add_title=add_title, is_final=is_final)
            # Save a checkpoint every save_int steps
            if (save_int > 0 and i % save_int == 0 and i > 0):
                self.save()
            # Summarize model every summary_int steps
            if (summary_int > 0 and i % summary_int == 0):
                self.progress_summary()
            # Advance to the next time step; this increments nstep
            self.time_step()
            # Test for convergence
            if (self.rms_chng[i] < tol):
                print(f'\nConverged on step {i:d}.')
                print(f'RMS change = {self.rms_chng[i]:12.6e}.')
                # print(f'RMS rate   = {self.rms_dC_dt[i]:12.6e}.')
                break
            # Test for early termination due to stagnant steps
            if (self.nstagnant > self.nstagnant_max):
                print(f'\nTerminated due to {self.nstagnant:d} stagnant steps.')
                print(f'RMS change = {self.rms_chng[i]:12.6e}.')
                break

        # Always summarize the final state
        self.progress_summary()
        # Plot the final frame if plot interval is nonzero
        if (plot_int > 0):
            add_title = True
            is_final = True
            self.plot_frame(add_title=add_title, is_final=is_final)
        # Save the final state if the save interval is nonzero
        if (save_int > 0):
            self.save()

    # *************************************************************************************************
    def solve(self):
        """Solve at the configured resolution"""
        # Unpack solver options
        tol: float = self.tol
        plot_int: int = self.plot_int
        save_int: int = self.save_int
        summary_int: int = self.summary_int
        dot_int: int = self.dot_int

        # Solve at the base resolution
        self.solve_impl(tol=tol, plot_int=plot_int, save_int=save_int, summary_int=summary_int, dot_int=dot_int)
        
    # *************************************************************************************************
    def solve_multigrid(self):
        """Solve at multiple resolutions"""
        # Unpack solver options
        n_max: int = self.n_max
        tol: float = self.tol

        # Solve at the base resolution
        self.solve()
        # Status
        print(f'solve_multigrid: Completed solve at base resolution with n = {self.n}, tol = {tol:5.3e}.')
        print('********************************************************************************')

        # Upsample as long as the resolution is less than n_max
        while self.n < n_max:
            # Upsample
            self.upsample(f=2)
            # Solve at the upsampled resolution
            self.solve()
            # Status
            print(f'solve_multigrid: Completed solve at resolution with n = {self.n}')
            print('********************************************************************************')

        # Report elapsed time
        self.wall_time(verbose=True)
        print('')

    # *************************************************************************************************
    def calc_eq_miss_HA(self) -> float:
        """Show equilibrium factor miss on HA equilibrium"""
        H = self.conc[0]
        A = self.conc[3]
        HA = self.conc[2]
        eq_factor = (H * A) / (HA * Ka)
        eq_miss = np.abs(eq_factor - 1.0)
        return np.mean(eq_miss)

    # *************************************************************************************************
    def calc_eq_miss_H2O(self) -> float:
        """Show equilibrium factor miss on H2O equilibrium"""
        H = self.conc[0]
        OH = self.conc[4]
        eq_factor = (H * OH) / Kw
        eq_miss = np.abs(eq_factor - 1.0)
        return np.mean(eq_miss)

    # *************************************************************************************************
    def test_equilibrium(self):
        """Test whether equilibrate is working"""
        # Equilibrium miss before calling equilibrate
        eq_miss_HA_0: float = self.calc_eq_miss_HA()
        eq_miss_H2O_0: float = self.calc_eq_miss_H2O()

        # Call equilibrate
        self.equilibrate()

        # Equilibrium miss after calling equilibrate
        eq_miss_HA_1 = self.calc_eq_miss_HA()
        eq_miss_H2O_1 = self.calc_eq_miss_H2O()

        # Report results
        print(f'***** Equilibrium miss before and after')
        print(f'***** HA : {eq_miss_HA_0:8.6f} -> {eq_miss_HA_1:8.6f}')
        print(f'***** H2O: {eq_miss_H2O_0:8.6f} -> {eq_miss_H2O_1:8.6f}')

# *************************************************************************************************
def load_solver(fname: str) -> Solver:
    """Load a solver instance from a file; factory function"""
    path = data_dir / fname
    data = np.load(path)
    # Geometry; convert from meters to cm
    Lx: float   = data['Lx'] * 100.0
    Ly: float   = data['Ly'] * 100.0
    Lz: float   = data['Lz'] * 100.0
    # Operating conditions
    c0: NumpyFloatArray  = data['c0']
    Q: float        = data['Q']
    V0: float       = data['V0']
    H_frac: float   = data['H_frac']
    OH_cathode: float = data['OH_cathode']
    # Solver options
    m: int          = data['m']
    n: int          = data['n']
    cfl: float      = data['cfl']
    max_chng: float = data['max_chng']
    nstep_max: int  = data['nstep_max']
    # Create the geometry instance
    g: Geometry = Geometry(Lx=Lx, Ly=Ly, Lz=Lz)
    # Create the operating conditions instance
    cond: Conditions = Conditions(c0=c0, Q=Q, V0=V0, H_frac=H_frac, OH_cathode=OH_cathode)
    # Create the solver options
    opt: SolverOptions = SolverOptions(m=m, n=n, cfl=cfl, max_chng=max_chng, nstep_max=nstep_max)
    # Create a solver instance
    s: Solver = Solver(g=g, cond=cond, opt=opt)
    # Load the state
    s.load_state(fname=fname)
    # Return the solver instance
    return s
