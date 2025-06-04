# Precheck for the configuration

Do the following tests to determine the parameters in configuration generation: thermalization steps, step size of HMC, save interval and beta.

## 1. Heat balance and acceptance rate

Heat balance should be achieved before collecting the first configuration. Acceptance rate should be about 80%.

The way to check the heat balance is to calculate the plaquette value, to see if it is fluctuating around a value.

## 2. Check the autocorrelation

Aim to get the configurations with small autocorrelation.

The way to check the autocorrelation is to calculate Wilson loops with different shapes, then check the plot of reduced uncertainties v.s. bin size, to see if the reduced uncertainties are stable.

*You can also try to plot the heatmap of correlation matrix.*

## 3. Scale setting

Calculate different Wilson loops with different shapes, determine the linear potential, then fit to estimate the lattice spacing. Check if the lattice spacing extracted from different Lz is roughly consistent.

## 4. Spectrum

Tune a pion mass mpi=400 MeV, then check the proton mass to see if it is around 1.2 GeV. (You can do 1 HYP smearing)