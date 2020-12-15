#!/usr/bin/env python
# coding: utf-8

# This code produces all figures as well as summary statistics reported in the main paper and supplement. It has to be run after the analysis code `COVID Regression Vengine v3`, using the same ControlFile and from the same folder; this code will read the output .tab files from the analysis, and will fail if they are not available. See main `Analysis Code` folder for `ReadMe` with explanation of ControlFile format and fields.

# In[ ]:


import os
import re
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.cm as cm

from shutil import copy
from distutils.dir_util import copy_tree
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


##### FUNCTION DEFINITIONS FOR PLOTTING FIGURES FROM RESULTS #####

def plot_intervals(res, a, b, ci=(0.05, 0.95), avg='mode'):
    """Plots parameter `a` and `b` values for each country, with error 
    bars indicating credible interval as specified in `ci`, selecting 
    main data values based on `avg`='mode' or 'median'"""
    # Set results to use as median or default to modal (most likely) value
    if avg == 'median':
        res_a, res_b = res[f'{a}_0.5'], res[f'{b}_0.5']
    else:
        res_a, res_b = res[a], res[b]
        
    # Calculate error bars based on upper and lower bounds in `ci`
    eq_gdn_err = (res_a - res[f'{a}_{ci[0]}'], res[f'{a}_{ci[1]}'] - res_a)
    eq_dpm_err = (np.log10(res_b) - np.log10(res[f'{b}_{ci[0]}']), 
                  np.log10(res[f'{b}_{ci[1]}']) - np.log10(res_b))

    # Create figure and plot points with error bars
    fig0, (ax0, ax1) = plt.subplots(2, 1, figsize=[12, 5], constrained_layout=True)
    ax0.errorbar(res.index, res_a, yerr=eq_gdn_err, fmt='o')
    ax1.errorbar(res.index, np.log10(res_b), yerr=eq_dpm_err, fmt='o')
    
    # Assign titles and X-axis labels
    ax0.set_title('Quasi-equilibrium relative contact rate')
    ax1.set_title('Log quasi-equilibrium death rate')
    ax0.tick_params(axis='x', labelsize=8, labelrotation=90)
    ax1.tick_params(axis='x', labelsize=8, labelrotation=90)
    
    # Turn on Y-axis gridlines
    ax0.grid(True, axis='y')
    ax1.grid(True, axis='y')

    fig0.savefig(f"./{baserunname}_Intervals.jpg")


def plot_chg_dpm(res, a, delta, ci=(0.05, 0.95), threshold=0.005, avg='mode'):
    """Plots end-period change in DPM `a`, averaged over time `delta`, 
    with error bars indicating credible interval as specified in `ci`, 
    for countries with absolute change greater than `threshold`, 
    selecting main data values based on `avg`='mode' or 'median'"""
    
    # Set results to use as median or default to modal (most likely) value
    if avg == 'median':
        sort_res = res.sort_values(f'{a}_0.5', ascending=False)
        sort_res_a = sort_res[f'{a}_0.5']
    else:
        sort_res = res.sort_values(a, ascending=False)
        sort_res_a = sort_res[a]

    # Calculate threshold indices and error bars
    sort_res = sort_res[abs(sort_res_a) > threshold]
    sort_res_h = sort_res_a[sort_res_a > threshold]
    sort_res_l = sort_res_a[sort_res_a < -threshold]
    sort_res_a = sort_res_a[abs(sort_res_a) > threshold]
    chg_dpm_err = (sort_res_a - sort_res[f'{a}_{ci[0]}'], 
                   sort_res[f'{a}_{ci[1]}'] - sort_res_a)

    # Create two copies of plot for use with broken axis
    fig0, (ax0, ax1) = plt.subplots(1, 2, figsize=[12, 4], sharey=True, tight_layout=True)
    ax0.errorbar(sort_res_a.index, sort_res_a, yerr=chg_dpm_err, fmt='o')
    ax1.errorbar(sort_res_a.index, sort_res_a, yerr=chg_dpm_err, fmt='o')

    # Split X-axis and set Y-zoom
    ax0.set_xlim(xmax=sort_res_h.index[-1])
    ax1.set_xlim(xmin=sort_res_l.index[0])
    ax0.set_ylim(-0.1, 0.1)
    ax1.set_ylim(-0.1, 0.1)

    # Remove internal spines and ticks
    ax0.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.yaxis.tick_right()

    # Set up X-axis labels, Y-axis gridlines, and figure title
    ax0.tick_params(axis='x', labelsize=8, labelrotation=90)
    ax1.tick_params(axis='x', labelsize=8, labelrotation=90)
    ax0.grid(True, axis='y')
    ax1.grid(True, axis='y')
    fig0.suptitle(r'Trends in quasi-equilibrium death rates $d^{eq}_{NM}$', fontsize=12)

    # Set up broken axis diagonal slashes
    d = .015 # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False)
    ax0.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
    ax0.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-left diagonal
    kwargs.update(transform=ax1.transAxes) # switch to the bottom axes
    ax1.plot((-d,d),(-d,+d), **kwargs) # top-right diagonal
    ax1.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal
   
    fig0.savefig(f"./{baserunname}_ChgDpm.jpg")

    
def plot_scatter(res, x, y):
    """Plots double-histogram scatterplot of country results for `x` and 
    `y` (by default, g(D) and DPM), with `y` on log-scale"""
    # Create main figure
    fig0, ax0 = plt.subplots(figsize=[12, 12], constrained_layout=True)
    
    area = (1e-03 * np.sqrt(res['population'])) ** 2 # Assign area based on population
    
    # Helper function to drop indices with negative x or y values from dataframe
    def drop_negs(df):
        posdf = df[res[x] > 0]
        posdf = posdf[res[y] > 0]
        return posdf
    
    # Drop negatives from results & area / colour keys  
    res_x, res_y, area, gdp = [drop_negs(df) for df in [res[x], res[y], area, res['gdp_per_cap']]]
    
    ax0.set_yscale('log') # Set Y-axis to log scale
    
    # Plot main scatterplot with GDP color scale and area proportional to population
    ax0.scatter(res_x, res_y, s=area, c=np.log(gdp), alpha=0.5)
    
    # Label points with country abbreviations
    for i in res_x.index:
        ax0.annotate(i, (res_x[i], res_y[i]), fontsize=8)
    
    # Create axes for X- and Y-axis histograms
    divider = make_axes_locatable(ax0)
    ax_histx = divider.append_axes('top', 1.5, pad=0.1, sharex=ax0)
    ax_histy = divider.append_axes('right', 1.5, pad=0.1, sharey=ax0)

    # Remove ticks & labels on axes abutting main plot
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    # Plot X- and Y-axis histograms, correcting Y-axis hist for log scale
    xhist = ax_histx.hist(res_x, bins=20)
    ax_histy.set_yscale('log') # Set Y-axis to log scale
    yhist = np.histogram(res_y, bins=20) # Create temporary histogram to calculate bin limits
    logbins = np.logspace(np.log10(yhist[1][0]),np.log10(yhist[1][-1]),len(yhist[1]))
    yhist = ax_histy.hist(res_y, bins=logbins, orientation='horizontal')

    # Format Y-axis ticks as decimals instead of scientific notation
    ax_histy.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    
    # Set labels and titles
    ax0.set_xlabel(r'Quasi-equilibrium normalized contact rate $g^{eq}$')
    ax0.set_ylabel(r'Quasi-equilibrium death rate per million $d^{eq}_{NM}$')
    fig0.suptitle('Normalized contacts vs. expected deaths', fontsize=14)
    
    fig0.savefig(f"./{baserunname}_Scatter.jpg", bbox_inches='tight')


def plot_scatter_basic(res):
    """Plot scatterplot of log(mean DPM) against average Re"""
    # Create figure and scatterplot
    fig0, ax0 = plt.subplots(figsize=[7, 7], constrained_layout=True)
    ax0.set_yscale('log') # Set Y-axis to log scale
    ax0.scatter(res['avg_Re'], res['mean_dpm'], alpha=0.5)
    
    # Label points with country abbreviations
    for i in res.index:
        ax0.annotate(i, (res['avg_Re'][i], res['mean_dpm'][i]), fontsize=8)
    
    # Format Y-axis ticks without scientific notation
    ax0.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax0.yaxis.get_major_formatter().set_scientific(False)
    
    # Set labels and titles
    ax0.set_xlabel(r'Effective reproduction rate $R_e$')
    ax0.set_ylabel(r'Average death rate per million $d_{NM}$')
    ax0.set_xlim(0.5, 1.5) # Widen X-axis for clearer visual clustering of Re values
    fig0.suptitle('Reported death rates vs. effective reproduction rates', fontsize=14)
    
    fig0.savefig(f"./{baserunname}_Scatter_Basic.jpg", bbox_inches='tight')
    
def plot_country_fits(c_list, c_names):
    """Plot four illustrative country-level graphs for countries in 
    `c_list` showing infections (sim and data) and deaths (data) on one 
    plot, and responsiveness and g(D) on another, for each country"""
    
    res_list = [] # Initialise container for relevant data

    # Read country results files and extract necessary data series
    for c in c_list:
        c_res = pd.read_csv(f'./{baserunname}/{c}/{baserunname}_{c}_MC.tab', sep='\t', 
                            index_col=0, error_bad_lines=False)

        #Change X-axis from numeric to datetime
        dates = pd.to_datetime('2019-12-31') + pd.to_timedelta(c_res.columns.astype(int), unit='D')

        res_list.append([dates, # Dates for X-axis
                         c_res.loc[f'Mu[{c}]'], # Infections (model)
                         c_res.loc[f'DataFlowOverTime[{c}]'], # Infections (data)
                         c_res.loc[f'DeathsOverTime[{c}]'], # Deaths (data)
                         c_res.loc[f'alpha[{c}]'], # Responsiveness
                         c_res.loc[f'g death[{c}]']]) # Contact reduction g(D)

    # Create main figure and axes
    fig0 = plt.figure(figsize=[12, 12], constrained_layout=True)
    gs = fig0.add_gridspec(4, 2, height_ratios=[2.5,1,2.5,1]) # Specify size ratios with gridspec
    (ax0, ax4), (ax1, ax5), (ax2, ax6), (ax3, ax7) = gs.subplots(sharex=True) # Note order of axes

    # Compile axes into list by country, for ease of country-by-country plotting
    ax_list = [(ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7)]
    
    # Set labels for left-hand axes only, and figure title
    ax0.set_ylabel(r'Daily new cases (total) $r_{IM}$ &' + '\n' + r'deaths per million $d_{NM}$')
    ax1.set_ylabel(r'Responsiveness $\alpha$ &' + '\n' + r'relative contact rate $g(D)$')
    ax2.set_ylabel(r'Daily new cases (total) $r_{IM}$ &' + '\n' + r'deaths per million $d_{NM}$')
    ax3.set_ylabel(r'Responsiveness $\alpha$ &' + '\n' + r'relative contact rate $g(D)$')
    fig0.suptitle('Illustrative country fits to data', fontsize=14)

    # Loop through pair of axes for each country
    for i, axs in enumerate(ax_list):
        # Plot infection (model and data) lines
        line0, = axs[0].plot(res_list[i][0], res_list[i][1], color='blue', 
                             label=r'Cases $r_{IM}$ (model)')
        line1, = axs[0].plot(res_list[i][0], res_list[i][2], color='blue', linestyle='dotted', 
                             label=r'Cases $r_{IM}$ (data)')
        axb = axs[0].twinx() # Create twinned axes for deaths
        axc = axs[1].twinx() # Create twinned axes for g(D)
        
        # Plot deaths (data), responsiveness, and g(D) lines
        line2, = axb.plot(res_list[i][0], res_list[i][3], color='purple', linestyle='dotted', 
                          label=r'Deaths $d_{NM}$ (data)')
        line3, = axs[1].plot(res_list[i][0], res_list[i][4], color='red', 
                             label=r'Responsiveness $\alpha$')
        line4, = axc.plot(res_list[i][0], res_list[i][5], color='orange', 
                          label=r'Contacts relative to normal $g(D)$')
        axs[0].set_title(c_names[i], y=0.92) # Label with country full names from `c_names`
        axs[1].set_ylim(0) # Ensure responsiveness & g(D) plot starts at 0

    # Set X-axis tick labels to months for bottom axes
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')
    ax3.xaxis.set_major_locator(months)
    ax3.xaxis.set_major_formatter(months_fmt)
    ax7.xaxis.set_major_locator(months)
    ax7.xaxis.set_major_formatter(months_fmt)

    # Set up and display legend
    handles = [line0, line1, line2, line3, line4]
    labels = [line.get_label() for line in handles]
    fig0.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0), ncol=5)
    
    fig0.savefig(f"./{baserunname}_Country_Fits.jpg", bbox_inches='tight')
    

def plot_regression(baserunname):
    """Plot two-panel figure with death regression coefficient over time 
    and last-day actual deaths against calculated equilibrium deaths"""
    
    # Read in deaths data and regression coefficients from processed results
    dthdf = pd.read_csv(f'{baserunname}_deaths.tab', sep='\t', index_col=['field', 'iso_code'])
    regdf = pd.read_csv(f'{baserunname}_regression.tab', sep='\t', index_col=0)
    
    # Extract and clean last-day death data
    X_final, Y_final = dthdf.loc['eqDeath'].iloc[:, -1], dthdf.loc['dpm'].iloc[:, -1]
    X_final[X_final <= 0] = np.NaN
    Y_final[Y_final <= 0] = np.NaN

    #Change X-axis from numeric to datetime
    dates = pd.to_datetime('2019-12-31') + pd.to_timedelta(regdf.index.astype(int), unit='D')

    # Create figure and plot regression coefficient over time on first axes
    fig0, (ax0, ax1) = plt.subplots(1, 2, figsize=[12, 6], constrained_layout=True)
    ax0.plot(dates, regdf['RLM'])
    
    # Plot log-log scatterplot of deaths on second axes
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.scatter(X_final, Y_final)
    
    # Turn off scientific notation on X- and Y-axis
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.get_major_formatter().set_scientific(False)
    
    # Plot regression line for last day on scatterplot
    linespace = np.linspace(X_final.min(), X_final.max(), 20)
    slope_final = regdf['RLM'].iloc[-1] # Get slops from last day regression coefficient
    ax1.plot(linespace, linespace * slope_final, color='purple', 
             linestyle='dotted', label=f'{slope_final}')

    # Set X-axis tick labels to months
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')
    ax0.xaxis.set_major_locator(months)
    ax0.xaxis.set_major_formatter(months_fmt)

    # Set labels and titles, including annotation showing regression line slope
    ax0.set_ylabel('Regression coefficient')
    ax1.set_xlabel(r'Projected quasi-equilibrium deaths per million $d^{eq}_{NM}$')
    ax1.set_ylabel(r'Actual current deaths per million $d_{NM}$')
    ax0.set_title(r'A) Regression coefficient for $d_{NM}$ vs. $d^{eq}_{NM}$')
    ax1.set_title(r'B) Last day $d_{NM}$ vs. $d^{eq}_{NM}$ with regression line')
    ax1.text(100, 100, f'b = {"{:.3f}".format(slope_final)} ', horizontalalignment='right')

    fig0.savefig(f"./{baserunname}_EqDeath_Reg.jpg", bbox_inches='tight')
    

def plot_fits_full(res, a, b, lab):
    """Plot fits for every country in results dataframe `res` between 
    variables `a` and `b` (usually sim output and data), identified with 
    label `lab` (WARNING: takes a while)"""
    #Change x-axis from numeric to datetime
    dates = pd.to_datetime('2019-12-31') + pd.to_timedelta(res.columns.astype(int), unit='D')
    res.columns = dates

    # Create figure - UPDATE subplot number and size based on no. of countries
    fig0, axs = plt.subplots(12, 10, figsize=[12, 12], sharex=True, constrained_layout=True)

    # Extract list of countries to plot from dataframe
    c_list = res.index.levels[1]

    # Extract relevant results for variables `a` and `b`
    res_a, res_b = res.loc[a], res.loc[b]

    # Delete any excess axes
    for i in range(len(axs.flatten())-len(c_list)):
        fig0.delaxes(axs.flatten()[len(c_list) + i])

    # Loop over list of countries to plot each one
    for i, c in enumerate(c_list):
        ax = axs.flatten()[i] # Identify current axes from flattened list
        
        # Plot each variable and set title
        ax.plot(res_a.loc[c], color='blue')
        ax.plot(res_b.loc[c], color='red', linestyle='dotted')
        ax.set_title(c, fontsize=7)
        
        # Add Y-axis ticks and labels, scaled appropriately
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
        ax.yaxis.offsetText.set_fontsize(6)
        ax.tick_params(axis='y', labelsize=6)

        # Set X-axis tick labels to months
        months = mdates.MonthLocator(interval=3)
        months_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        ax.tick_params(axis='x', labelsize=6, labelrotation=90)

    # Add title based on specified `lab`
    fig0.suptitle(f'Simulated daily {lab}s (blue) and true reported {lab} rate (red)                   \n(7-day rolling average) for all countries', fontsize=14)

    fig0.savefig(f"./{baserunname}_CtyFitsFull.jpg")

    
def plot_sensitivity(baserunname, sens_vars, sens_mults):
    """Plot illustrative run parameter sensitivity analysis results"""
    # Read in base run and sensitivity Re and DPM results
    res = pd.read_csv(f'./{baserunname}_sens_base.tab', sep='\t', index_col=0)
    redf = pd.read_csv(f'./{baserunname}_sens_Re.tab', sep='\t', index_col=0)
    dthdf = pd.read_csv(f'./{baserunname}_sens_Death.tab', sep='\t', index_col=0)

    # Set up run names
    sfxs = [str(mult).replace('.','') for mult in sens_mults]
    colnames = [f'{baserunname}_sens_{sens_vars[0][0]}_{sfx}' for sfx in sfxs]

    #Change X-axis from numeric to datetime, for base and sensitivity runs
    basedates = pd.to_datetime('2019-12-31') + pd.to_timedelta(res.columns.astype(int), unit='D')
    dates = pd.to_datetime('2019-12-31') + pd.to_timedelta(redf.columns.astype(int), unit='D')

    # Create figure and main axes, and twinned axes for each
    fig0, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=[12, 10], constrained_layout=True)
    ax0b, ax1b, ax2b, ax3b = [ax.twinx() for ax in [ax0, ax1, ax2, ax3]]

    # Plot lines for base case
    ax0.plot(basedates, res.loc['Re'], color='blue', label='Re')
    ax0b.plot(basedates, res.loc['Deaths'], color='red', label='Deaths')

    for ax, axb in zip([ax1, ax2, ax3], [ax1b, ax2b, ax3b]):
        # Plot lines for Re
        line0, = ax.plot(dates, redf.loc[f'{colnames[1]}'], color='blue', 
                         label=r'Effective reproduction rate $R_e$ (base)')
        line1, = ax.plot(dates, redf.loc[f'{colnames[0]}'], color='blue', 
                         linestyle='dotted', label=r'$R_e$ (0.5x)')
        line2, = ax.plot(dates, redf.loc[f'{colnames[2]}'], color='blue', 
                         linestyle='dashed', label=r'$R_e$ (2x)')
        # Plot lines for death rate
        line3, = axb.plot(dates, dthdf.loc[f'{colnames[1]}'], color='red', 
                          label=r'Daily death rate per million $d_{NM}$ (base)')
        line4, = axb.plot(dates, dthdf.loc[f'{colnames[0]}'], color='red', 
                          linestyle='dotted', label=r'$d_{NM}$ (0.5x)')
        line5, = axb.plot(dates, dthdf.loc[f'{colnames[2]}'], color='red', 
                          linestyle='dashed', label=r'$d_{NM}$ (2x)')

    # Set axes and figure titles
    ax0.set_title('A) Base run showing quasi-equilibrium phase')
    ax1.set_title(r'B) Base effective contact rate $\beta_0$')
    ax2.set_title(r'C) Responsiveness $\alpha$')
    ax3.set_title(r'D) Time to perceive risk $\lambda$')
    fig0.suptitle('Sensitivity of basic epidemic dynamics to key parameters', fontsize=14)

    # Set X-axis tick labels to years and set Y-axis limits
    years = mdates.YearLocator()
    years0 = mdates.YearLocator() # Secondary locator for two-year window
    years_fmt = mdates.DateFormatter('%Y')
    ax0.xaxis.set_major_locator(years0)
    ax0.xaxis.set_major_formatter(years_fmt)
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.set_ylim(ymin=0, ymax=2.1)

    ax0.set_ylim(ymin=0, ymax=2.1)
    ax1b.set_ylim(ymax=10)
    ax2b.set_ylim(ymax=5)
    ax3b.set_ylim(ymax=5)

    # Set up and display legend
    handles = [line0, line1, line2, line3, line4, line5]
    labels = [line.get_label() for line in handles]
    fig0.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0), ncol=2)

    fig0.savefig(f"./{baserunname}_Sensitivity.jpg", bbox_inches='tight')


# In[ ]:


controlfilename = input("Enter control file name (with extension):")
cf = json.load(open(controlfilename, 'r'))

# Unpack controlfile into variables
for k,v in cf.items():
    exec(k + '=v')

for setting in [datasettings, analysissettings]:
    for k, v in setting.items():
        exec(k + '=v')


# In[ ]:


##### GENERATE OUTPUTS & FIGURES BASED ON MAIN RUN #####

# Initialise and read in base data
os.chdir(f"{baserunname}_IterCal")
table = pd.read_csv('InputData.tab', sep='\t', index_col=[0,1])
countrylist = list(table.index.levels[1])
print(countrylist)

basename = cf['baserunname']
startdate = datetime.date.fromisoformat('2019-12-31')

# Create string form of sens_durs for inclusion in summary
sens_durs_text = ', '.join(str(dur) for dur in sens_durs[:-1]) + f'and {sens_durs[-1]}'

summarytext = [] # Initialise container for summary output

# First run graphing & results summary for main disease duration value
for i in [main_dur]:
    mainrunname = f'{basename}{i}'
    print(mainrunname)
    
    # Read in main results tab file
    results = pd.read_csv(f'{mainrunname}_results.tab', sep='\t', index_col=0)
    
    # Drop countries in nonequilibrium state
    results[results['end_dpm'] <= 0] = np.NaN
    results[results['end_dpm_mdl'] <= 0] = np.NaN 
    trunc_list = results[results['end_dpm'].isna()].index.tolist() # Identify dropped countries
    results = results[results['end_dpm'].notna()]

    # Summarise results by quantiles, mean and stdev
    quants = results.quantile([0.05, 0.5, 0.95]).T
    quants['Mean'], quants['StDev'] = results.mean().T, results.std().T
    
    params = quants.loc[params_list] # Subset quantiles dataframe to just listed parameters
    
    # Calculate mean & median IQRs for listed parameters
    iqr_vars = [f'{var}_iqr' for var in params_list]
    param_iqrs = results[iqr_vars] # Extract parameter IQR columns from main results tabfile
    param_iqrs.columns = params_list # Rename IQR columns with base parameter names
    params['Mean IQR'], params['Median IQR'] = param_iqrs.mean(), param_iqrs.median()
    
    # Export summarised results to quantiles and params files
    results.to_csv(f'{mainrunname}_results_trunc.tab', sep='\t')
    quants.loc[tablekeys].to_csv(f'./{mainrunname}_quantiles.tab', sep='\t')
    results[params_list].to_csv(f'{mainrunname}_params_full.tab', sep='\t')
    params.to_csv(f'{mainrunname}_params_summary.tab', sep='\t')
    
    # Convert specified dates from numeric to datetime
    enddate = startdate + datetime.timedelta(days=len(table.columns)-1)
    eqmdate = startdate + datetime.timedelta(days=eqtime)
    eardate = startdate + datetime.timedelta(days=earlytime)
    
    # Summarise MIQRs for full IQRs list
    miqrs = [str(results[f'{var}_iqr'].mean()) for var in iqr_list]
    miqr_text = '\t'.join(iqr_list) # IQR variable names
    miqr_vals = '\t'.join(miqrs) # IQR variable values
    
    # Calculate correlation & log-correlation for DPM and g(D)
    corr = pearsonr(results['end_gdn'], results['end_dpm'])
    logcorr = pearsonr(results['end_gdn'], np.log10(results['end_dpm']))
    
    # Compile summary output as text list
    summarytext.extend(
        [f"Total countries\t{len(countrylist)}\n", 
         f"Total population\t{results['population'].sum()}\n", 
         f"Nonequilibrium countries\t{len(trunc_list)}\n", 
         f"Equilibrium countries\t{len(results.index)}\n", 
         f"Start date\t{startdate.isoformat()}\t\tMin cumulative cases\t{min_cases}\n", 
         f"End date\t{enddate.isoformat()}\t\tMin datapoints\t{min_datapoints}\n", 
         f"Eqm date\t{eqmdate.isoformat()}\t\tStartpoint cases\t{start_cases}\n", 
         f"Early date\t{eardate.isoformat()}\n", 
         f"Disease duration\t{i}\n", 
         f"Initial responsiveness\t{np.exp(-results['alpha 0'].median())}\n", 
         f"Final responsiveness\t{np.exp(-results['end_alpha'].median())}\n", 
         f"Correlation\t{corr[0]}\t{corr[1]}\n", 
         f"LogCorrelation\t{logcorr[0]}\t{logcorr[1]}\n", 
         f"Historical window\t{hist_window}\n", 
         f"eq_gdn MNIQR\t{results['eq_gdn_niqr'].median()}\n", 
         f"eq_dpm MNIQR\t{results['eq_dpm_niqr'].median()}\n", 
         f"end_gdn MNIQR\t{results['end_gdn_niqr'].median()}\n", 
         f"end_dpm MNIQR\t{results['end_dpm_niqr'].median()}\n", 
         f"mean MAEOM\t{results['maeom'].mean()}\n", 
         f"mean MAPE\t{results['mape'].mean()}\n", 
         f"mean r2\t{results['r2'].mean()}\n", 
         f"\t{miqr_text}\n", 
         f"MIQR\t{miqr_vals}\n", 
         f"List of nonequilibrium countries\t{', '.join(trunc_list)}\n", 
         f"MCLIMIT\t{mcsettings['MCLIMIT']}\n", 
         f"MCBURNIN\t{mcsettings['MCBURNIN']}\n", 
         f"Sens days\t{sens_durs_text}\n"
         ])

    # Plot various figures
    plot_scatter(results, 'end_gdn', 'end_dpm')
    plot_intervals(results, 'end_gdn', 'end_dpm', ci=(0.05, 0.95))
    plot_scatter_basic(results)
    plot_sensitivity(mainrunname, sens_vars, sens_mults)
    plot_country_fits(c_list, c_names)
    plot_regression(mainrunname)
    plot_chg_dpm(results, 'chg_dpm', delta, threshold=0.005, avg='median')

    # Plot all-country infection fits
    infdf = pd.read_csv(f'{mainrunname}_infections.tab', sep='\t', index_col=[0,1])
    plot_fits_full(infdf, 'inf_exp', 'inf_data', 'infection')
    


# In[ ]:


##### GENERATE DISEASE DURATION SENSITIVITY OUTPUTS & FIGURES #####

# Next run graphing & results for each disease duration sensitivity value
for i in sens_durs:
    baserunname = f'{basename}{i}'
    print(baserunname)
    
    results = pd.read_csv(f'{baserunname}_results.tab', sep='\t', index_col=0)
    results[results['end_dpm'] <= 0] = np.NaN
    results[results['end_dpm_mdl'] <= 0] = np.NaN
    trunc_list = results[results['end_dpm'].isna()].index.tolist()
    results = results[results['end_dpm'].notna()]
    
    # Only need mean & median for listed parameters
    quants = results.quantile([0.5]).T
    quants['Mean'] = results.mean().T
    params = quants.loc[params_list]
    
    # Read mean & median values from main run and calculate % changes
    main_params = pd.read_csv(f'{mainrunname}_params_summary.tab', sep='\t', index_col=0)
    params['MeanDelta'] = (params['Mean'] - main_params['Mean'])/main_params['Mean']
    params['MedDelta'] = (params[0.5] - main_params['0.5'])/main_params['0.5']

    params.to_csv(f'{baserunname}_params_summary.tab', sep='\t')
    
    corr = pearsonr(results['end_gdn'], results['end_dpm'])
    logcorr = pearsonr(results['end_gdn'], np.log10(results['end_dpm']))

    # Add key sensitivity results to summary output text list
    summarytext.extend(
        [f"Sens{i} Correlation\t{corr[0]}\t{corr[1]}\n", 
         f"Sens{i} LogCorrelation\t{logcorr[0]}\t{logcorr[1]}\n", 
         f"Sens{i} mean MAEOM\t{results['maeom'].mean()}\n", 
         f"Sens{i} mean MAPE\t{results['mape'].mean()}\n", 
         f"Sens{i} mean r2\t{results['r2'].mean()}\n" 
         ])

    plot_scatter(results, 'end_gdn', 'end_dpm')


# In[ ]:


##### GENERATE BEHAVIOURAL RESPONSE SENSITIVITY OUTPUTS & FIGURES #####

# Next run graphing & results for no-behavioural-response sensitivity run    
for i in ['NBR']:
    baserunname = f'{basename}{i}'
    print(baserunname)
    
    results = pd.read_csv(f'{baserunname}_results.tab', sep='\t', index_col=0)
    infdf = pd.read_csv(f'{baserunname}_infections.tab', sep='\t', index_col=[0,1])
    
    summarytext.extend(
        [f"Sens{i} mean MAEOM\t{results['maeom'].mean()}\n", 
         f"Sens{i} mean MAPE\t{results['mape'].mean()}\n", 
         f"Sens{i} mean r2\t{results['r2'].mean()}\n" 
         ])

    plot_fits_full(infdf, 'inf_exp', 'inf_data', 'infection')

# Write compiled output summary to text
with open(f"{mainrunname}_summary.txt", 'w') as summaryfile:
    summaryfile.writelines(summarytext)


# In[ ]:


000000000000000000000000000000000000000000000000000000000000000000000000
000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

