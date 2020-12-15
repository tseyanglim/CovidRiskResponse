Covid Iterative Calibration (Vengine version)
WARNING: THIS VERSION REQUIRES VENGINE TO RUN!

1) Save the .py file in the same folder as your model and relevant modelling files (.voc, .vpd, .vsc, .vdfx inputs, etc.). The program will copy all relevant files to a new subfolder and work there, as it generates a lot of new files.

2) You need a single 'Control File' as input. This is a JSON-format .txt file which acts as a Python dictionary to control the calibration and analysis. Ensure that all fields are appropriately updated before running the .py file. Note that all strings will need to be "double-quoted". The order of fields does not matter.
	a) baserunname - the base name you want to use for the Vensim runs; also the name of the subfolder that will be created
	b) data_url - the URL from which data updates are pulled (from OWID's GitHub)
	c) simsettings - a Python-format dictionary containing some or all of the following keys:
		i) model - the base .mdl file
		ii) data - a Python-format list of .vdf/.vdfx data files to load
		iii) payoff, sensitivity, optparm, savelist, senssavelist - the relevant Vensim control files (.vpd, .vsc, .voc, .lst, and .lst respectively)
		iv) changes - a Python-format list of changes files to load (e.g. .cin, .out)
	d) vensimpath - filepath to your Vensim .exe - MAKE SURE TO UPDATE THIS
	e) timelimit - maximum amount of time to wait between optimization runs of a single model (i.e. restarts) - if Vensim stalls out, this is how long the script will wait before killing Vensim and starting again
	f) updatedata - switch for controlling whether to update data; to turn off updating, set this to 0, in which case an existing InputData.tab file will be used
	g) mccores - to turn off model calibration, set this to 0; if 1 or more, will run country-by-country calibrations
	h) mcsettings - a Python-format dictionary of Vensim optimization control settings to use for running MCMC. These will be used to modify the .voc file for the MCMC runs. Be sure to set either 'Optimizer' or 'Sensitivity' to turn on MCMC (or just leave as-is); the 'MCLIMIT' setting gives the total number of iterations per MCMC process. Additional MCMC and optimization control settings can be added as desired.
	i) datasettings - a Python-format dictionary of settings for data pre-processing & filtering countries to include, as follows:
		i) min_cases - the inclusion threshold for cumulative reported cases
		ii) min_datapoints - the inclusion threshold for number of datapoints reported after the start threshold is reached
		iii) start_cases - the number of cumulative  reported cases at which to start including data for each country
		iv) droplist - a Python-format list of field codes to exclude, mostly for having no deaths or otherwise causing errors
		v) renames - a Python-format dictionary of field codes to rename to match ISO country codes
	j) analysissettings - a Python-format dictionary of settings for calibration and analysis of results, as follows:
		i) hist_window - the time window preceding the last data point over which to average Re and DPM for comparison
		ii) delta - the time window preceding the last data point over which to average changes in the projected equilibrium death rate
		iii) eqtime - the time point at which to calculate projected equilibrium death and g(D) rates, besides the end of the run
		iv) earlytime - the time point at which to calculate responsiveness and g(D) for purposes of comparison with later in the pandemic (set to moderately early date, but late enough for most countries to have started experiencing outbreaks)
		v) main_dur - the main assumed value of disease duration to use for calibration
		vi) sens_durs - a Python-format list of additional disease duration values to use for sensitivity analysis
		vii) genparams - a Python-format list of strings, used to identify lines in the first changes .out file to keep for initial country calibrations; typically this should be (as the name implies) the names of the general parameters
		viii) means_list - a Python-format list of variables to calculate historical means for based on hist_window
		ix) perc_list - a Python-format list of floats between [0,1] indicating percentile values to report when calculating percentiles from MCMC results
		x) gof_vars - a Python-format list of two variables to compare to calculate goodness-of-fit statistics
		xi) iqr_list - a Python-format list of variables for which to calculate and report interquartile ranges
		xii) params_list - a Python-format list of model parameters to report in summarised form (keep this short, to just key parameters)
		xiii) tablekeys - a Python-format list of variables to save in output results tabfile (this should be longer and more inclusive)
		xiv) sens_vars - a Python-format list of lists, each with parameters to change in illustrative parameter sensitivity analysis
		xv) sens_mults - a Python-format list of floats indicating how much to vary sens_vars parameters for parameter sensitivity analysis
		xvi) c_list - a Python-format list of four country ISO codes indicating which countries to plot illustrative fits for using plot_country_fits in graphing codes
		xvii) c_names - a Python-format list of four strings, corresponding to the preferred display names of the countries in c_list

3) Once the Control File is updated, ensure it is in the same folder as the .py file.

4) Run the .py file. It will prompt you for the name of the Control File, after which everything should run automatically.
	IMPORTANT: Vengine has a warning popup on initialization, which the script should dismiss automatically. There are two known times this may fail - on first running the script, and if your computer suspends or sleeps (even if running on server). For the first issue, on first running the script, if Vengine does not start running the optimization automatically after a few seconds, just manually dismiss the popup. For the second issue, I recommend that you change computer power settings to never sleep/suspend while running this script.

5) All output from the .py script will be written to a .log file under "{baserunname}.log".

6) When updating the Control File, watch out for commas and other punctuation! If you get a JSON decoder error when you input the Control File name, double-check the punctuation in the Control File.

IMPORTANT - note re: timelimit parameter
The timelimit parameter is only supposed to kill and restart Vensim if it is stalled out. As long as optimization is continuing (i.e. the optimization .log is still being written to), even if it the overall process takes longer than the timelimit, it will be allowed to complete - UNLESS a single optimization run does not yield any log file changes for longer than the timelimit. If optimization control settings are high-intensity enough that this happens, you WILL get stuck in an infinite loop - so if doing high-res optimization, adjust this parameter up accordingly. On the other hand, if set too high, more time will be wasted when Vensim does happen to stall out.