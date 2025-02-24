import numpy as np
import matplotlib.pyplot as plt
from utils import calc_eq_conc_pH, calc_eq_conc, kappa_fact, pKa
from pathlib import Path
from typing import Any
NumpyArray = np.ndarray[Any, np.dtype[np.float64]]

# Directory to save plots
plot_dir: Path = Path('figs', 'analysis')

# *************************************************************************************************
def calc_asr(conc: NumpyArray) -> float:
    """Calculate the ASR given the concentration"""
    # The length of the cell in cm
    L: float = 0.1
    # The conductivity in S / m
    kappa: float = float(np.sum(conc * kappa_fact))
    # The resistivity in Ohm-m
    r: float = 1.0 / kappa
    # The area specific resistance in Ohm-cm^2
    # L is already in cm, but need to multiply by 100 to convert from m to cm in resistivity
    return r * L * 100.0

# *************************************************************************************************
def plot_asr_pH(a0: float):
    """Plot the simulated ASR vs. pH"""
    # pH values
    pH_min: float = 0.0
    pH_max: float = 6.0
    pH_step: float = 0.05
    pH: NumpyArray = np.arange(pH_min, pH_max + pH_step * 0.5, pH_step)

    # ASR values
    asr: NumpyArray = np.zeros_like(pH)

    # Calculate the ASR values
    for i, pH_i in enumerate(pH):
        # Calculate the equilibrium concentrations
        conc: NumpyArray = calc_eq_conc_pH(pH=pH_i, a0=a0)
        # Calculate the area-specific resistance
        asr[i] = calc_asr(conc=conc)

    # ASR at the pKa
    conc: NumpyArray = calc_eq_conc_pH(pH=pKa, a0=a0)
    asr_pKa: float = calc_asr(conc=conc)

    # Plot the ASR vs. pH
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_title(r'Area-Specific Resistance vs. pH with $[\text{OAc}^-]$ = 2 M', fontsize=16)
    ax.set_xlabel(r'pH (dimensionless)')
    ax.set_ylabel(r'Area Specific Resistance ($\Omega \text{cm}^{2}$)')
    ax.plot(pH, asr, color='blue')
    ax.plot(pKa, asr_pKa, color='red', marker='o', markersize=5.0)
    # ax.legend(loc='upper right', edgecolor='black', fontsize=14)
    x: float = pKa
    y: float = asr_pKa
    dx: float = 0.15
    dy: float = 1.0
    # arrowprops = dict(facecolor='black', arrowstyle='->')
    ax.annotate(f'pKa = {x:0.2f}\nasr = {y:0.2f}', xy=(x, y), xytext=(x + dx, y + dy))
    fig.savefig(plot_dir / '32_asr_pH.png')

# *************************************************************************************************
def plot_asr_Na(a0: float):
    """Plot the simulated ASR vs. Sodium Fraction"""
    # Sodium fractions of acetate
    Na_step = 0.005
    Na_frac = np.linspace(Na_step / 2, 1.0 - Na_step / 2, 200)

    # ASR values
    asr: NumpyArray = np.zeros_like(Na_frac)

    # Calculate the ASR values
    for i, Na_frac_i in enumerate(Na_frac):
        # The absolute sodium concentration
        Na: float = Na_frac_i * a0
        # Calculate the equilibrium concentrations
        conc: NumpyArray = calc_eq_conc(Na=Na, a0=a0, verbose=False)
        # Calculate the area-specific resistance
        asr[i] = calc_asr(conc=conc)

    # ASR at the pKa
    conc: NumpyArray = calc_eq_conc_pH(pH=pKa, a0=a0)
    Na_frac_pKa: float = conc[1] / a0
    asr_pKa: float = calc_asr(conc=conc)

    # Shared by both plots
    title: str = r'Area-Specific Resistance vs. Na fraction in 2M Acetate'
    xlabel: str = r'Na fraction = $[\text{Na}^{+}] / ([\text{Na}^{+}] + [\text{H}^{+}])$ (dimensionless)'
    ylabel: str = r'Area Specific Resistance ($\Omega \text{cm}^{2}$)'

    # Plot the ASR vs. pH
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(Na_frac, asr, color='blue')
    ax.plot(Na_frac_pKa, asr_pKa, color='red', marker='o', markersize=5.0)
    # ax.legend(loc='upper right', edgecolor='black', fontsize=14)
    x: float = Na_frac_pKa
    y: float = asr_pKa
    dx: float = 0.02
    dy: float = 1.00
    # arrowprops = dict(facecolor='black', arrowstyle='->')
    ax.annotate(f'Na frac = {x:0.2f}\nasr = {y:0.2f}', 
                xy=(x, y), xytext=(x + dx, y + dy))
    fig.savefig(plot_dir / '33_asr_Na_frac.png')

    # Zoom in on Na fraction between 0.4 and 0.6
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.75, 1.50)
    ax.plot(Na_frac, asr, color='blue')
    ax.plot(Na_frac_pKa, asr_pKa, color='red', marker='o', markersize=5.0)
    dx: float = 0.005
    dy: float = 0.02
    ax.annotate(f'Na frac = {x:0.2f}\nasr = {y:0.2f}', 
                xy=(x, y), xytext=(x + dx, y + dy))
    fig.savefig(plot_dir / '34_asr_Na_frac.png')

# *************************************************************************************************
def main():
    """Fit crossover data to a regression model"""
    # The total acetate concentration in mM
    a0: float = 2000.0
    # Create the plot directory if it does not exist
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Plot the ASR vs. pH
    plot_asr_pH(a0=a0)
    # Plot the ASR vs. Na fraction
    plot_asr_Na(a0=a0)

# *************************************************************************************************
if __name__ == '__main__':
    main()