### TL-Fitter by D. M. Esteves (INESC MN/IST-ULisboa)  ###
TL Fitter is an interactive fitting routine for thermoluminescence glow curves (either discrete or continuous density of states) based on the matplotlib library. Please use pip install -r requirements.txt to get the correct version of the necessary packages.

The goal of this simple Python script is to read a raw data file (with the temperature in °C), plot it and fit it with a model considering either N discrete states or a continous density of states approximated by N gaussians. An interactive set of sliders is available on the right-hand side of the window, which allow the parameters to be easily varied and to assess their influence on the glow curve. Once the user is happy with the simulated curve, the given parameters can be used as starting values for a fitting routine aiming to minimise the thermoluminescence figure of merit (FOM).

The input file specifies the mandatory or optional inputs.

The file 'example.dat' contains experimental data that is used as an example for the script. The file 'example_Output_sim.txt' gives the curve obtained manually as a first guess, while 'example_Output_fit.txt' shows the best curve obtained by fitting. The corresponding parameters are printed to 'example_Output_par.txt'.
