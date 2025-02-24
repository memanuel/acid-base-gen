import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from typing import Any
NumpyArray = np.ndarray[Any, np.dtype[np.float64]]

# Array of OH concentrations in mM = mol/m^3
conc_OH: NumpyArray = np.array([500.0, 1000.0, 2000.0, 500.0, 1000.0, 2000.0])

# Array of current densities in mA/cm^2
curr_dens: NumpyArray = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])

# Array of crossover fluxes in nM / cm^2 / s
crossover: NumpyArray = np.array([4.0, 12.0, 27.0, 6.0, 18.0, 37.0])

# Conversion factor for current density from mA/cm^2 to A/m^2
curr_dens_to_SI: float = 10.0

# Conversion factor for crossover flux from nM/cm^2/s to mol/m^2/s
crossover_to_SI: float = 1.0E-5

# Directory to save plots
plot_dir: Path = Path('figs', 'analysis')

# Set font family to arial
plt.rcParams["font.family"] = "arial"
# Use LaTeX to format text in plots
# plt.rc('text', usetex=True)

# *************************************************************************************************
def fit_crossover(conc_OH: NumpyArray, curr_dens: NumpyArray, crossover: NumpyArray):
    """Fit a regression model to the crossover data"""

    # Convert current density to SI units of A / m^2
    curr_dens_SI = curr_dens * curr_dens_to_SI

    # Convert crossover flux to SI units of mol / m^2 / s
    crossover_SI = crossover * crossover_to_SI

    # Fit a regression model of the form
    # crossover = alpha * conc_OH + beta * conc_OH * curr_dens_SI

    # Assemble a 6x2 matrix of [conc_OH, curr_dens_SI] for the regression
    X = np.column_stack((conc_OH, conc_OH * curr_dens_SI))

    # Solve the linear regression problem
    coef = np.linalg.lstsq(X, crossover_SI, rcond=None)[0]
    alpha, beta = coef
    # calculate the r squared
    resid = crossover_SI - np.dot(X, coef)
    ss_res = np.sum(np.square(resid))
    ss_tot = np.sum(np.square(crossover_SI))
    r2 = 1.0 - ss_res / ss_tot

    # Return the coefficients and R2
    return alpha, beta, r2

def convert_fit_units(alpha: float, beta: float) -> tuple[float, float]:
    """
    Convert regression coefficients alpha and beta from SI to display units
    INPUTS:
        alpha:  Coefficient alpha in SI units of meter /sec
        beta:   Coefficient beta in SI units of meter^3 / sec / A
    RETURNS:
        alpha_disp: Coefficient alpha in display units of cm / sec
        beta_disp:  Coefficient beta in display units of cm^3 / sec / mA
    """

    # Convert alpha to from m^1 sec^-1 to cm^1 sec^-1
    alpha_disp: float = alpha * 1.0E2

    # Convert beta from SI units of m^3 sec^-1 A^-1 to cm^3 sec^-1 mA^-1
    beta_disp: float = beta * 1.0E6 / 1.0E3

    return alpha_disp, beta_disp

# *************************************************************************************************
def plot_crossover(conc_OH: NumpyArray, curr_dens: NumpyArray, crossover: NumpyArray, 
    alpha: float, beta: float, r2: float):

    # Convert current density to SI units of A / m^2; conversion factor is 10^-5
    curr_dens_SI = curr_dens * curr_dens_to_SI

    # Estimate the crossover flux using the regression model
    crossover_fit_SI = alpha * conc_OH + beta * conc_OH * curr_dens_SI

    # Convert the crossover flux from SI back to nM / cm^2 / s
    crossover_fit = crossover_fit_SI / crossover_to_SI

    # Convert the regression coefficients to display units
    alpha_disp: float
    beta_disp: float
    alpha_disp, beta_disp = convert_fit_units(alpha, beta)

    # Plot the data and the regression model
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_title('Crossover Flux Due to OH - Data vs. Model', fontsize=20)
    ax.set_xlabel(r'Crossover flux - Model ($\text{nmol} \cdot \text{cm}^{-2} \cdot \text{s}^{-1}$)', fontsize=16)
    ax.set_ylabel(r'Crossover flux - Actual ($\text{nmol} \cdot \text{cm}^{-2} \cdot \text{s}^{-1}$)', fontsize=16)
    ax.axis('equal')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.plot(crossover_fit, crossover, 'o', markersize=10, color='red', label='Data')
    ax.plot(crossover_fit, crossover_fit, color='black', label='Model')
    ax.legend(loc='lower right', edgecolor='black', fontsize=14)
    textstr = '\n'.join((
        r'Crossover = $\alpha \cdot \text{[OH]} + \beta \cdot \text{[OH]} \cdot j$',
        r'$\alpha=%5.3e \; \text{cm} \cdot \text{s}^{-1}$ ' % (alpha_disp,),
        r'$\beta=%5.3e \; \text{cm}^3 \cdot \text{s}^{-1} \cdot \text{mA}^{-1}$' % (beta_disp, ),
        r'$R^2=%8.6f$' % (r2, )))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    fig.savefig(plot_dir / '31_crossover_OH.png')

# *************************************************************************************************
def main():
    """Fit crossover data to a regression model"""
    # Fit the regression model
    alpha: float
    beta: float
    r2: float
    alpha, beta, r2 = fit_crossover(conc_OH=conc_OH, curr_dens=curr_dens, crossover=crossover)
    
    # Convert the regression coefficients to display units
    alpha_disp: float
    beta_disp: float
    alpha_disp, beta_disp = convert_fit_units(alpha, beta)
    
    # Report the results in display units
    print(f'crossover = alpha * conc_OH + beta * conc_OH * curr_dens')
    print(f'alpha = {alpha_disp:6.3e} cm / s')
    print(f'beta  = {beta_disp:6.3e} cm^3 / s / mA')
    print(f'R^2   = {r2:6.3f}')

    # Create the plot directory if it does not exist
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot the data and the regression model
    plot_crossover(conc_OH=conc_OH, curr_dens=curr_dens, crossover=crossover, 
                   alpha=alpha, beta=beta, r2=r2)

# *************************************************************************************************
if __name__ == '__main__':
    main()
