import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import Slider, Button, RadioButtons
import seaborn as sns
import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.signal import savgol_filter
from lmfit import Minimizer, Parameters, report_fit
import win32api


# Boltzmann constant in eV/K
kB = 8.617e-5

# Integrated exponential function 
def int_exp(x, T0, E):
    return np.asarray([quad(lambda t: np.exp(-E/(kB*t)), T0, T)[0] for T in x])

# Fitting function
def fitting(params, x):
    model = 0

    En = params['Emin'].value
    dE = (params['Emax'].value-params['Emin'].value)/params['N'].value

    for i in range(int(params['N'].value+1)):
        dos = 0.
        for j in range(1,4):
            dos += params[f'n0{j}']*np.exp(-0.5*(En-params[f'Emed{j}'])*(En-params[f'Emed{j}'])/(params[f'Evar{j}']*params[f'Evar{j}']))

        model += params['S']*dos*np.exp(-En/(kB*np.array(x)))*np.exp(-params['S']/params['ramp']*kB*np.array(x)*np.array(x)/En * np.exp(-En/(kB*np.array(x))) * (1-2*kB*np.array(x)/En))/params['ramp']
        En += dE

    model *= dE

    model += params['A'] + params['B']*np.exp(params['C']*np.array(x)) #background
             
    return model
    

# Objective function to be minimised (FOM)
def residual(params, x, data):
    return np.sqrt(abs(fitting(params, x)-data))

#FOM
def FOM(params, df):
    return(sum(abs(residual(params, df['Temperature'], df['Intensity']))**2)/sum(df['Intensity'])*100)

# Main routine

input_file = sys.argv[1]

# Read input file
dfInput = pd.read_csv(input_file, sep='\t', names=['Name','Init','Min','Max','Vary'], dtype='string', comment='#', keep_default_na=False)

file_name = dfInput[dfInput['Name'] == 'data']['Init'][0]
dfInput = dfInput.drop(dfInput[dfInput['Name'] == 'data'].index)

# Prepare output files
file_base = os.path.splitext(file_name)[0]
outputPar = f'{file_base}_Output_par.txt'
outputFit = f'{file_base}_Output_fit.txt'
outputSim = f'{file_base}_Output_sim.txt'

# Always fixed parameters
dfEmpty = dfInput[(dfInput['Vary'] != 'False') & (dfInput['Vary'] != 'True')].reset_index()

# Fixed
dfFix = dfInput[dfInput['Vary'] == 'False'].reset_index()

# Variable
dfPars = dfInput[dfInput['Vary'] == 'True'].reset_index()
Npars = len(dfPars)

params = Parameters()
for i in range(len(dfEmpty)):
    params.add(dfEmpty['Name'][i], value=float(dfEmpty['Init'][i]), vary=False)

for i in range(len(dfFix)):
    params.add(dfFix['Name'][i], value=float(dfFix['Init'][i]), min=float(dfFix['Min'][i]), max=float(dfFix['Max'][i]), vary=False)

for i in range(Npars):
    params.add(dfPars['Name'][i], value=float(dfPars['Init'][i]), min=float(dfPars['Min'][i]), max=float(dfPars['Max'][i]), vary=True)

# Read data file
df = pd.read_csv(file_name, sep=' ', comment='#', header=None, names=['Temperature','Intensity'], dtype='float')
#print(df)

for i in range(len(df)):
    df['Temperature'][i] += 273.
    if df['Temperature'][i] > params['Tmax'].value:
        df = df.drop(i)

sns.reset_defaults()

sns.set_context("poster")

# Prepare the plot
fig = plt.figure("Fitting")
fig.subplots_adjust(right=0.5) #Espaco para sliders

#axis
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.spines[:].set_visible(True)
plt.xlabel(r'Temperature (K)')
plt.ylabel(r'TL Intensity (arb. un.)')

#plot exp and sim curves
ax.plot(df['Temperature'], df['Intensity'], label=r'Experimental points', marker='o', color='b', linestyle='', markersize=5)
plot, = ax.plot(df['Temperature'], fitting(params, df['Temperature']), label=f'Simulation       FOM: {FOM(params, df):.3f}%', color='black', linestyle='-')

leg = ax.legend(loc=10, bbox_to_anchor=(0.5, 1.05), frameon=False)
leg.set_draggable(True)

#Sliders for variable parameters
spacing = .78/Npars
sliders_axes = [plt.axes([0.6, .85 - n*spacing, 0.2, 0.02]) for n in range (Npars)]
sliders = []

for i in range(Npars):
    sliders.append(Slider(ax=sliders_axes[i], label=dfPars['Name'][i], valmin=float(dfPars['Min'][i]), valmax=float(dfPars['Max'][i]), valinit=float(dfPars['Init'][i])))

for slider in sliders_axes:
    slider.spines[:].set_visible(True)

#Update function
def update(val):
    for i in range(Npars):
        params[dfPars['Name'][i]].set(sliders[i].val)

    plot.set_ydata(fitting(params, df['Temperature']))
    plot.set_label(f'Simulation       FOM: {FOM(params, df):.3f}%')

    leg = ax.legend(loc=10, bbox_to_anchor=(0.5, 1.05), frameon=False)
    leg.set_draggable(True)
    #fig.canvas.draw_idle()  

for slider in sliders:
    slider.on_changed(update)

# Reset button
resetax = plt.axes([0.85, 0.01, 0.1, 0.075])
resetax.spines[:].set_visible(True)
reset_but = Button(resetax, 'Reset')

def reset(event):
    for slider in sliders:
        slider.reset()

    plot.set_visible(True)
    Nlines = len(ax.lines) 
    for i in range(2, Nlines):
        ax.lines.pop(i)

    plot.set_label(f'Simulation       FOM: {FOM(params, df):.3f}%')

    leg = ax.legend(loc=10, bbox_to_anchor=(0.5, 1.05), frameon=False)
    leg.set_draggable(True)
    fig.canvas.draw_idle()  

        
# Associate the reset function with the reset button
reset_but.on_clicked(reset)

# Fit button
fitax = plt.axes([0.55, 0.01, 0.1, 0.075])
fitax.spines[:].set_visible(True)
fit_but = Button(fitax, 'Fit')


def fit(event):
    # Minimise
    minner = Minimizer(residual, params, fcn_args=(df['Temperature'], df['Intensity']),  nan_policy='raise')
    result = minner.minimize()

    # Write error report to cmd
    report_fit(result)

    # Figure of Merit
    FigMerit = FOM(result.params, df)

    print(f'    Figure of Merit: \t{FigMerit}')

    with open(outputPar, "w") as parOut:
        for par in result.params:
            parOut.write(f'{result.params[par].name}\t{result.params[par].value}\n')
        parOut.write(f'FOM\t{FigMerit}\n')

    # Fit result
    final = fitting(result.params, df['Temperature'].to_numpy())

    with open(outputFit, "w") as fitOut:
        for i in range(len(df)):
            fitOut.write(f"{df['Temperature'].to_numpy()[i]}\t{final[i]}\n")
    
    
    plot.set_visible(False)
    plot.set_label(f'_Simulation       FOM: {FOM(params, df):.3f}%')

    # Plot fit
    ax.plot(df['Temperature'], final, label=f'Fit                    FOM: {FigMerit:.3f}%', color='red', linestyle='--')

    fig.canvas.draw_idle()  

    leg = ax.legend(loc=10, bbox_to_anchor=(0.5, 1.05), frameon=False)
    leg.set_draggable(True)


# Associate the fit function with the fit button
fit_but.on_clicked(fit)

# Save simulation button
saveax = plt.axes([0.7, 0.01, 0.1, 0.075])
saveax.spines[:].set_visible(True)
save_but = Button(saveax, 'Save data')

def save(event):
    with open(outputSim, "w") as simOut:
        for i in range(len(df)):
            simOut.write(f"{df['Temperature'].to_numpy()[i]}\t{fitting(params, df['Temperature'].to_numpy())[i]}\n")

# Associate the save function with the save button
save_but.on_clicked(save)

# Starts the program with the window already maximised
figManager = plt.get_current_fig_manager()
figManager.window.state('zoomed')

plt.show()
