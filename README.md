# Introduction

This GitHub repository contains all the data and source code required to replicate the analysis and figures in our publication *Electrochemical acid-base generators for decoupled carbon management*.
These are the steps to replicate the calculations on your own computer.

## Clone the Repository from GitHub

    git clone https://github.com/memanuel/acid-base-gen.git

## Create an Anaconda environment with the necessary packages

Install Anaconda if it is not already on your system.
See [Anaconda Installation Guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
Then run these commands in a terminal if you are on a Windows platform:

    > cd \your\path\to\acid-base-gen
    > conda env create -f src/abg-win64.yml
    > conda activate abg

This will create a new conda environment on your system named `abg` that includes all the packages required to run the Python programs.
The steps on a Linux platform are similar:

    $ cd /your/path/to/acid-base-gen
    $ conda env create -f src/abg-linux.yml
    $ conda activate abg

It has not been tested on a Mac platform. However, in principle it should be easy to adapt to Mac if required.

## Replicate Generation of the published figures

To replicate the published figures on your computer, run these steps in order:

    (abg) python src/acid-base.py

This calculation loads the state of a numerical simulation from a data file that is included in this repository.
If you wish to generate the full calculation from scratch, first manually delete the data file:

    (abg) del data\NaA_v109.npz

Then run `acid-base.py` as above. This time the program will first run the simulation before saving the data file and generating the plots.
