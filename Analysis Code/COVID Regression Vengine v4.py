#!/usr/bin/env python
# coding: utf-8

# **Important:** The model estimation code is intended to work with an experimental parallelised Vensim engine. With appropriate modifications to the main function calls (but not the analytical procedure), the same analysis can be run on regular commercially available Vensim DSS, though it will take *much* longer. Please contact [Tom Fiddaman](mailto:tom@ventanasystems.com) for information on the experimental Vensim engine.
# 
# For more information on the model estimation procedure, see S1 of the Supplementary Materials of the paper.
# 
# **Note:** if running in Jupyter, the `keyboard` module may need to be installed directly in the Notebook; see [here](https://stackoverflow.com/questions/38368318/installing-a-pip-package-from-within-a-jupyter-notebook-not-working) for example. The `keyboard` module is *only* used to bypass Vengine error messages; if not running Vengine (e.g. using normal Vensim DSS), it is not necessary, and you can safely remove the import statement and all `press` commands in the code.

# In[ ]:


import os
import subprocess
import re
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from keyboard import press
from shutil import copy
from distutils.dir_util import copy_tree
from mpl_toolkits.axes_grid1 import make_axes_locatable


##### CLASS & FUNCTION DEFINITIONS FOR WORKING WITH VENSIM/VENGINE #####

class Script(object):
    """Master object for holding and modifying .cmd script settings, 
    creating .cmd files, and running them through Vensim/Vengine"""
    def __init__(self, controlfile):
        print("Initialising", self)
        for k, v in controlfile['simsettings'].items():
            self.__setattr__(k, v if isinstance(v, str) else v.copy())
        self.setvals = []
        self.runcmd = "MENU>RUN_OPTIMIZE|o\n"
        self.savecmd = f"MENU>VDF2TAB|!|!|{self.savelist}|\n"
        self.basename = controlfile['baserunname']
        self.cmdtext = []
        
    def copy_model_files(self, dirname):
        """Create subdirectory and copy relevant model files to it,
        then change working directory to subdirectory"""
        os.makedirs(dirname, exist_ok=True)
        os.chdir(f"./{dirname}")

        # Copy needed files from the working directory into the sub-directory
        for s in ['model', 'payoff', 'optparm', 'sensitivity', 'savelist', 'senssavelist']:
            if getattr(self, s):
                copy(f"../{getattr(self, s)}", "./")
        for slist in ['data', 'changes']:
            for file in getattr(self, slist):
                copy(f"../{file}", "./")
            
    def add_suffixes(self, settingsfxs):
        """Cleanly modifies .cmd script settings with specified suffixes"""
        for s, sfx in settingsfxs.items():
            if hasattr(self, s):
                self.__setattr__(s, getattr(self, s)[:-4] + sfx + getattr(self, s)[-4:])
   
    def update_changes(self, chglist, setvals=[]):
        """Reformats chglist as needed to extend changes settings; 
        see compile_script for details"""
        # Combines and flattens list of paired change names & suffixes
        flatlist = [i for s in 
                    [[f"{self.basename}_{n}_{sfx}.out" for n in name] 
                     if isinstance(name, list) else [f"{self.basename}_{name}_{sfx}.out"] 
                     for name, sfx in chglist] for i in s]
        self.changes.extend(flatlist)
        self.setvals = setvals
          
    def write_script(self, scriptname):
        """Compiles and writes actual .cmd script file"""
        self.cmdtext.extend(["SPECIAL>NOINTERACTION\n", 
                             f"SPECIAL>LOADMODEL|{self.model}\n"])
        
        for s in ['payoff', 'sensitivity', 'optparm', 'savelist', 'senssavelist']:
            if hasattr(self, s):
                self.cmdtext.append(f"SIMULATE>{s}|{getattr(self, s)}\n")
        
        if hasattr(self, 'data'):
            datatext = ','.join(self.data)
            self.cmdtext.append(f"SIMULATE>DATA|\"{','.join(self.data)}\"\n")

        if self.changes:
            self.cmdtext.append(f"SIMULATE>READCIN|{self.changes[0]}\n")
            for file in self.changes[1:]:
                self.cmdtext.append(f"SIMULATE>ADDCIN|{file}\n")
        
        self.cmdtext.extend(["\n", f"SIMULATE>RUNNAME|{scriptname}\n", 
                             self.runcmd, self.savecmd, 
                             "SPECIAL>CLEARRUNS\n", "MENU>EXIT\n"])
        
        with open(f"{scriptname}.cmd", 'w') as scriptfile:
            scriptfile.writelines(self.cmdtext)
    
    def run_script(self, scriptname, controlfile, subdir, logfile):
        """Runs .cmd script file using function robust to 
        Vengine errors, and returns payoff value if applicable"""
        return run_vengine_script(scriptname, controlfile['vensimpath'], 
                                  controlfile['timelimit'], '.log', check_opt, logfile)

    
class CtyScript(Script):
    """Script subclass for country optimization runs"""
    def __init__(self, controlfile):
        super().__init__(controlfile)
        self.genparams = controlfile['genparams'].copy()
        
    def prep_subdir(self, scriptname, controlfile, subdir):
        """Creates subdirectory for country-specific files and output"""
        self.copy_model_files(subdir)
        copy(f"../{scriptname}.cmd", "./")
        self.genparams.append(f"[{subdir}]")
        for file in self.changes:
            if 'main' in file:
                clean_outfile(file, self.genparams)
            
    def run_script(self, scriptname, controlfile, subdir, logfile):
        self.prep_subdir(scriptname, controlfile, subdir)
        res = run_vengine_script(scriptname, controlfile['vensimpath'], 
                                 controlfile['timelimit'], '.log', check_opt, logfile)
        copy(f"./{scriptname}.out", "..") # Copy the .out file to parent directory
        os.chdir("..")
        return res


class CtyMCScript(CtyScript):
    """Script subclass for country MCMC optimizations"""
    def run_script(self, scriptname, controlfile, subdir, logfile):
        self.prep_subdir(scriptname, controlfile, subdir)
        res = run_vengine_script(scriptname, controlfile['vensimpath'], 
                                 controlfile['timelimit'], '_MCMC_points.tab', check_MC, logfile)
        os.chdir("..")
        return res

        
class LongScript(Script):
    """Script subclass for long calibration runs e.g. all-params"""
    def run_script(self, scriptname, controlfile, subdir, logfile):
        return run_vengine_script(scriptname, controlfile['vensimpath'], 
                                  controlfile['timelimit']*5, '.log', check_opt, logfile)
        

class MultiScript(Script):
    """Script subclass for running multiple scenarios consecutively 
    using SETVAL and exporting to a single output file"""
    def write_script(self, scriptname):
        self.cmdtext.extend(["SPECIAL>NOINTERACTION\n", 
                             f"SPECIAL>LOADMODEL|{self.model}\n"])
        
        for s in ['payoff', 'sensitivity', 'optparm', 'savelist', 'senssavelist']:
            if hasattr(self, s):
                self.cmdtext.append(f"SIMULATE>{s}|{getattr(self, s)}\n")
        
        if hasattr(self, 'data'):
            datatext = ','.join(self.data)
            self.cmdtext.append(f"SIMULATE>DATA|\"{','.join(self.data)}\"\n")

        for varnames, vals, sfx in self.setvals:
            if hasattr(self, 'changes'):
                self.cmdtext.append(f"\nSIMULATE>READCIN|{self.changes[0]}\n")
                for file in self.changes[1:]:
                    self.cmdtext.append(f"SIMULATE>ADDCIN|{file}\n")
            self.cmdtext.extend(
                ["\n", f"SIMULATE>RUNNAME|{scriptname}_{sfx}\n", 
                 *[f"SIMULATE>SETVAL|{var}={val}\n" for var, val in zip(varnames, vals)], 
                 "MENU>RUN|o\n", f"MENU>VDF2TAB|!|{scriptname}|{self.senssavelist}|+||:!\n"])
        
        self.cmdtext.extend(["\n", "SPECIAL>CLEARRUNS\n", "MENU>EXIT\n"])
        
        with open(f"{scriptname}.cmd", 'w') as scriptfile:
            scriptfile.writelines(self.cmdtext)

    def run_script(self, scriptname, controlfile, subdir, logfile):
        return run_vengine_script(scriptname, controlfile['vensimpath'], 
                                  controlfile['timelimit']/5, '.vdf', check_multi, logfile)


class ScenRunScript(Script):
    """Script subclass for simple single runs (not optimzations), 
    optionally with scenario files"""
    def __init__(self, controlfile):
        super().__init__(controlfile)
        self.runcmd = "MENU>RUN|o\n"
    
    def update_changes(self, chglist, setvals=None):
        scen = []
        while True:
            try:
                if type(chglist[-1]) == str:
                    scen.append(chglist.pop())
                else: break
            except IndexError:
                break
        super().update_changes(chglist, setvals)
        scen.reverse()
        self.changes.extend(scen)
        chglist.extend(scen)
    
    def run_script(self, scriptname, controlfile, subdir, logfile):
        return run_vengine_script(scriptname, controlfile['vensimpath'], 
                                  controlfile['timelimit']/5, '.vdf', check_run, logfile)


def compile_script(controlfile, scriptclass, name, namesfx, settingsfxs, 
                   logfile, chglist=[], setvals=[], subdir=None):
    """Master function for assembling & running .cmd script
    
    Parameters
    ----------
    controlfile : JSON object
        Master control file specifying sim settings, runname, etc.
    scriptclass : Script object
        Type of script object to instantiate, depending on run type
    name : str
    namesfx : str
        Along with `name`, specifies name added to baserunname for run
    settingsfxs : dict of str
        Dict of suffixes to append to filenames in simsettings; use to 
        distinguish versions of e.g. .mdl, .voc, .vpd etc. files
    logfile : str of filename/path
    chglist : list of tuples of (str or list, str)
        Specifies changes files to be used in script; specify as tuples 
        corresponding to `name`, `namesfx` of previous run .out to use; 
        tuples can also take a list of `names` as first element, taking 
        each with the same second element; if used with ScenScript run, 
        `chglist` can also take one non-tuple str as its last element, 
        which will be added directly (e.g. .cin files for scenarios)
    setvals : list of tuples of (str, int or float, <str>)
        Specifies variables and values to change for a given run using 
        Vensim's SETVAL script command; by default all SETVAL commands 
        will be implemented together for main run, but if `scriptclass` 
        is MultiScript, each SETVAL command will be implemented and run 
        separately in sequence; if used with MultiScript, each tuple in 
        `setvals` will require a third str element specifying the suffix 
        with which to save the run
    subdir : str, optional
        Name of subdirectory to create/use for run, if applicable
    
    Returns
    -------
    float
        Payoff value of the script run, if applicable, else 0
    """
    mainscript = scriptclass(controlfile)
    mainscript.add_suffixes(settingsfxs)
    mainscript.update_changes(chglist, setvals)
    scriptname = f"{mainscript.basename}_{name}_{namesfx}"    
    mainscript.write_script(scriptname)
    return mainscript.run_script(scriptname, controlfile, subdir, logfile)


def check_opt(scriptname, logfile):
    """Check function for use with run_vengine_script for optimizations"""
    if check_zeroes(scriptname):
        write_log(f"Help! {scriptname} is being repressed!", logfile)
    return not check_zeroes(scriptname)

def check_MC(scriptname, logfile, threshold=0.01):
    """Check function for run_vengine_script for MCMC"""
    if abs(compare_payoff(scriptname, logfile)) >= threshold:
        write_log(f"{scriptname} is a self-perpetuating autocracy! re-running MC...", logfile)
        return False
    return True

def check_run(scriptname, logfile):
    """Check function for run_vengine_script for normal & sens runs"""
    if not os.path.exists(f"./{scriptname}.vdf"):
        write_log(f"Help! {scriptname} is being repressed!", logfile)
    return os.path.exists(f"./{scriptname}.vdf")

def check_multi(scriptname, logfile):
    """Check function for run_vengine_script for Multiscript runs"""
    if not os.path.exists(f"./{scriptname}.tab"):
        write_log(f"Help! {scriptname} is being repressed!", logfile)
    return os.path.exists(f"./{scriptname}.tab")


def run_vengine_script(scriptname, vensimpath, timelimit, checkfile, check_func, logfile):
    """Call Vensim with command script using subprocess; monitor output 
    file for changes to see if Vensim has stalled out, and restart if 
    it does, or otherwise bugs out; return payoff if applicable"""

    write_log(f"Initialising {scriptname}!", logfile)
    attempts = 0
    while attempts < 30:
        proc = subprocess.Popen(f"{vensimpath} \"./{scriptname}.cmd\"")
        time.sleep(2)
        press('enter') # Necessary to bypass the popup message in Vengine
        while True:
            try:
                # Break out of loop if run completes within specified timelimit
                proc.wait(timeout=timelimit)
                break
            except subprocess.TimeoutExpired:
                try:
                    # If run not complete before timelimit, check to see if still ongoing
                    write_log(f"Checking for {scriptname}{checkfile}...", logfile)
                    timelag = time.time() - os.path.getmtime(f"./{scriptname}{checkfile}")
                    if timelag < (timelimit):
                        write_log(f"At {time.ctime()}, {round(timelag,3)}s since last output, "
                                  "continuing...", logfile)
                        continue
                    else:
                        # If output isn't being written, kill and restart run
                        proc.kill()
                        write_log(f"At {time.ctime()}, {round(timelag,3)}s since last output. "
                                  "Calibration timed out!", logfile)
                        break
                except FileNotFoundError:
                    # If output isn't being written, kill and restart run
                    proc.kill()
                    write_log("Calibration timed out!", logfile)
                    break
        if proc.returncode != 1: # Note that Vengine returns 1 on MENU>EXIT, not 0!
            write_log(f"Return code is {proc.returncode}", logfile)
            write_log("Vensim! Trying again...", logfile)
            continue
        try:
            # Ensure output is not bugged (specifics depend on type of run)
            if check_func(scriptname, logfile):
                break
            else:
                attempts += 1
                continue
        except FileNotFoundError:
            write_log("Outfile not found! That's it, I'm dead.", logfile)
            pass
    else:
        write_log(f"FAILURE! {scriptname} failed to calibrate!", logfile)
        return False
    
    time.sleep(2)

    if os.path.exists(f"./{scriptname}.out"):
        payoffvalue = read_payoff(f"{scriptname}.out")
        write_log(f"Payoff for {scriptname} is {payoffvalue}, calibration complete!", logfile)
        return payoffvalue
    return 0 # Set default payoff value for simtypes that don't generate one


def clean_outfile(outfilename, linekey):
    """Clean an outfile to include only lines containing a string in 
    `linekey`, which should be a list of strings to keep"""
    with open(outfilename,'r') as f:
        filedata = f.readlines()

    newdata = [line for line in filedata if any(k in line for k in linekey)]
    
    with open(outfilename, 'w') as f:
        f.writelines(newdata)


def check_zeroes(scriptname):
    """Check if an .out file has any parameters set to zero (indicates 
    Vengine error), return True if any parameters zeroed OR if # runs = 
    # restarts, and False otherwise"""
    filename = f"{scriptname}.out"
    with open(filename,'r') as f0:
        filedata = f0.readlines()
    
    checklist = []
    for line in filedata:
        if line[0] != ':':
            if ' = 0 ' in line:
                checklist.append(True)
            else:
                checklist.append(False)
        elif ':RESTART_MAX' in line:
            restarts = re.findall(r'\d+', line)[0]
    
    # Ensure number of simulations != number of restarts
    if f"After {restarts} simulations" in filedata[0]:
        checklist.append(True)
        
    # Ensure payoff is not erroneous
    if abs(read_payoff(filename)) == 1.29807e+33:
        checklist.append(True)
    
    return any(checklist)


def write_log(string, logfile):
    """Writes printed script output to a logfile"""
    with open(logfile,'a') as f:
        f.write(string + "\n")
    print(string)

    
def modify_mdl(country, finaltime, modelname, newmodelname):
    """Opens .mdl as text, identifies Rgn subscript, and replaces 
    with appropriate country name"""
    with open(modelname,'r') as f:
        filedata = f.read()
        
    rgnregex = re.compile(r"Rgn(\s)*?:(\n)?[\s\S]*?(\n\t~)")
    timeregex = re.compile(r"FINAL TIME\s*=\s*\d*\n")
    tempdata = rgnregex.sub(f"Rgn:\n\t{country}\n\t~", filedata)
    newdata = timeregex.sub(f"FINAL TIME = {finaltime}\n", tempdata)

    with open(newmodelname,'w') as f:
        f.write(newdata)


def split_voc(vocname, mcsettings):
    """Splits .VOC file into multiple versions, for main, country, 
    initial, full model, general MCMC, and country MCMC calibration"""
    with open(vocname,'r') as f0:
        filedata = f0.readlines()
    
    voccty = [line for line in filedata if line[0] == ':' or '[Rgn]' in line]
    vocfull = filedata.copy()

    # Turn off multiple start for full voc
    for l, line in enumerate(vocfull):
        if ':MULTIPLE_START' in line:
            vocfull[l] = ':MULTIPLE_START=OFF\n'

    # Make necessary substitutions for MCMC settings
    vocctymc = ''.join(voccty)
    for k,v in mcsettings.items():
        vocctymc = re.sub(f":{re.escape(k)}=.*", f":{k}={v}", vocctymc)
        
    # Write various voc versions to separate .voc files
    for fname, suffix in zip([voccty, vocfull, vocctymc], 
                             ['c', 'f', 'cmc']):
        with open(f"{vocname[:-4]}_{suffix}.voc", 'w') as f:
            f.writelines(fname)


def create_mdls(controlfile, countrylist, finaltime, logfile):
    """Creates copies of the base .mdl file for each country in list 
    (and one main copy) and splits .VOC files"""
    model = controlfile['simsettings']['model']
    for c in countrylist:
        newmodel = model[:-4] + f'_{c}.mdl'
        modify_mdl(c, finaltime, model, newmodel)

    mainmodel = model[:-4] + '_main.mdl'
    c_list = [f'{c}\\\n\t\t' if i % 10 == 9 else c for i,c in enumerate(countrylist)]
    countrylist_str = str(c_list)[1:-1].replace("'","")
    modify_mdl(countrylist_str, finaltime, model, mainmodel)
    split_voc(controlfile['simsettings']['optparm'], controlfile['mcsettings'])
    write_log("Files are ready! moving to calibration", logfile)


def read_payoff(outfile, line=1):
    """Identifies payoff value from .OUT or .REP file - 
    use line 1 (default) for .OUT, or use line 0 for .REP"""
    with open(outfile) as f:
        payoffline = f.readlines()[line]
    payoffvalue = [float(s) for s in 
                   re.findall(r'-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', payoffline)][0]
    return payoffvalue


def compare_payoff(scriptname, logfile):
    """Returns the difference in payoffs between .OUT and .REP file, 
    which should be zero in most cases except when MCMC bugs out"""
    try:
        difference = read_payoff(f"{scriptname}.out") - read_payoff(f"{scriptname}.rep", 0)
        write_log(f".OUT and .REP payoff difference is {difference}", logfile)
        return difference
    except IndexError:
        return 1e33


def increment_seed(vocfile, logfile):
    """Increments random number seed in a .VOC file by 1"""
    with open(vocfile, 'r') as f:
        vocdata = f.read()
    seedregex = re.compile(r':SEED=\d+')
    try:
        i = int(re.search(r'\d+', re.search(seedregex, vocdata).group()).group())
        newdata = seedregex.sub(f":SEED={i+1}", vocdata)
        with open(vocfile, 'w') as f:
            f.write(newdata)
    except:
        write_log("No seed found, skipping incrementing.", logfile)    


def read_outvals(outfile):
    """Converts .out file into list of tuples of var names & values"""
    with open(outfile, 'r') as f:
        output = [line for line in f.readlines() if (line[0] != ':')]

    names = [line.split('<=')[1].split('=')[0].strip() for line in output]
    values = [float(line.split('<=')[1].split('=')[1]) for line in output]
    
    return list(zip(names, values))


def read_value(outfile, varname):
    """Reads & returns a specified variable's value from an outfile"""
    with open(outfile, 'r') as f:
        output = [line for line in f.readlines() if (line[0] != ':')]
        
    for line in output:
        if varname in line:
            val = float(line.split('<=')[1].split('=')[1])
    
    return val


##### FUNCTION DEFINITIONS FOR ANALYSIS & DATA PROCESSING #####

def get_first_idx(s, threshold):
    """Return index of first value in series `s` above `threshold`"""
    return (s > threshold).idxmax(skipna=True)


def calc_eq_vals(df, eqtime, colnames=['eq_gdn', 'eq_alpha', 'eq_dpm'], duration=None):
    """Calculate equilibrium gdn, dpm, and alpha values projected for 
    `eqtime` based on `duration`, modifying dataframe in place with 
    additional columns as specified in `colnames` list"""
    # Override duration if specified, else take default from dataframe
    if duration:
        dur = duration
    else:
        dur = df['DiseaseDuration']
        
    # Then calculate projected equilibrium values; see model for details
    df[colnames[0]] = 1 / (df['beta'] * dur)
    df[colnames[1]] = (df['alpha 0'] + 1 / (1 + np.exp(-(eqtime - df['t0'])/df['theta']))
                      * (df['alpha f'] - df['alpha 0']))
    df[colnames[2]] = np.log(df['beta'] * dur) / df[colnames[1]]
    

def calc_end_vals(df, cum_dpm, ifr, endtime, delta=None, cum_dpm_del=None, duration=None):
    """Calculate equilibrium SFrac, gdn, dpm, and alpha values projected 
    at `endtime` of run based on `duration` and `cum_dpm`, modifying 
    dataframe in place; also calculates values for `delta` (int) time 
    units before end time if delta is specified"""
    # Override duration if specified, else take default from dataframe
    if duration:
        dur = duration
    else:
        dur = df['DiseaseDuration']
        
    # Then calculate projected equilibrium values; see model for details
    df['SFrac'] = 1 - (cum_dpm * df['DeathReportingRatio']/ifr)/1e+06
    df['end_alpha'] = (df['alpha 0'] + 1 / (1 + np.exp(-(endtime - df['t0'])/df['theta'])) 
                       * (df['alpha f'] - df['alpha 0']))
    df['end_dpm'] = np.log(df['beta'] * df['SFrac'] * dur) / df['end_alpha']
    df['end_gdn'] = np.exp(-df['end_alpha']*df['end_dpm'])
    
    # If delta time is specified, calculate projected values at `endtime` - `delta`
    if delta:
        df['SFrac_del'] = 1 - (cum_dpm_del * df['DeathReportingRatio']/ifr)/1e+06
        df['del_alpha'] = (df['alpha 0'] + 1 / (1 + np.exp(-(
            (endtime - delta) - df['t0'])/df['theta'])) * (df['alpha f'] - df['alpha 0']))
        df['del_dpm'] = np.log(df['beta'] * df['SFrac_del'] * dur) / df['del_alpha']
        df['del_gdn'] = np.exp(-df['del_alpha']*df['del_dpm'])
        
        # Normalise change in value and average over `delta`
        df['chg_dpm_raw'] = (df['end_dpm'] - df['del_dpm'])/df['end_dpm']
        df['chg_dpm'] = df['chg_dpm_raw']/delta
    

def generate_intervals(scriptname, cum_dpm, ifr, eqtime, endtime, iqr_list, 
                       perc_list, delta=None, cum_dpm_del=None, duration=None):
    """Generate credible intervals and IQRs using percentiles of results 
    from MCMC output, returning IQRs for variables in `iqr_list` and 
    percentiles specified in `perc_list` for hardcoded variables"""
    # Read in MCMC sample output
    samdf = pd.read_csv(f"{scriptname}_MCMC_sample.tab", sep='\t')
    samdf.columns = [col.split('[')[0] for col in samdf.columns]
    
    # Calculate quasi-equilibrium values at specified times for all runs in MCMC sample
    calc_eq_vals(samdf, eqtime, duration=duration)
    calc_end_vals(samdf, cum_dpm, ifr, endtime, delta, cum_dpm_del, duration)
    
    # Identify percentiles for quasi-equilibrum values based on `perc_list`
    percs = [samdf[var].quantile(perc_list) for var in 
             ['eq_gdn', 'eq_dpm', 'end_gdn', 'end_dpm', 'chg_dpm']]
    
    for perc, var in zip(percs, ['eq_gdn', 'eq_dpm', 'end_gdn', 'end_dpm', 'chg_dpm']):
        perc.index = [f'{var}_{i}' for i in perc_list]
    
    # Calculate IQRs for vars in `iqr_list` based on percentiles in MCMC sample
    iqr_vals = [(samdf[var].quantile(0.75) - samdf[var].quantile(0.25)) for var in iqr_list]
    iqrs = pd.Series(iqr_vals, index=[f'{var}_iqr' for var in iqr_list])
    
    return iqrs, percs


def calc_mean(resdf, var, limit=180):
    """Return mean of `var` over historical period given by `limit`"""
    limit = min(len(resdf.loc[var]), limit)
    val = resdf.loc[var][-limit:].mean()
    return val


def calc_gof(resdf, simvar, datavar):
    """Calculate goodness-of-fit measures for given sim & data vars"""
    resdf.loc['error'] = abs(resdf.loc[simvar] - resdf.loc[datavar])
    maeom = resdf.loc['error'].mean()/resdf.loc[datavar].mean()
    mape = (resdf.loc['error']/resdf.loc[datavar]).mean()
    r2 = (resdf.loc[simvar].corr(resdf.loc[datavar])) ** 2
    return maeom, mape, r2


def trunc_log(df):
    """Return log10 of a dataframe, ignoring negative base values"""
    df[df <= 0] = np.NaN
    return np.log10(df)


def process_results(scriptname, eqtime, earlytime, gof_vars, iqr_list=[], 
                    means_list=[], perc_list=[0.05,0.95], delta=None, duration=None):
    """Read single-country calibration results and calculate additional 
    outputs, incl. percentiles and IQRs, returning a compiled pd.Series 
    of processed country results `c_res` and a Dataframe with country 
    time series outputs (deaths & infs) `datdf`"""
    # Read country parameter values from .out file
    outlist = read_outvals(f'{scriptname}.out')
    varnames = [n.split('[')[0] for n in [var[0] for var in outlist]]
    vals = [var[1] for var in outlist]
    c_res = pd.Series(vals, index=varnames)
    
    # Read full country calibration results, extract death & inf data for output
    resdf = pd.read_csv(f'{scriptname}.tab', sep='\t', index_col=0, error_bad_lines=False)
    resdf.index = [n.split('[')[0] for n in resdf.index] # Separate subscripts
    datdf = resdf.loc[['DeathsOverTimeRaw', 'eqDeath', 'DataFlowOverTime', 'inf exp']]
    
    # Pull end-of-run values from full country results
    endtime = len(resdf.columns) - 1
    c_res['cum_dpm'] = resdf.loc['CumulativeDpm'][-1]
    c_res['cum_dpm_del'] = resdf.loc['CumulativeDpm'][-(1 + delta)]
    c_res['IFR'] = resdf.loc['IFR'][0]
    c_res['SFrac_mdl'] = resdf.loc['SFrac'][-1]
    c_res['end_dpm_mdl'] = resdf.loc['eqDeath'][-1]
    c_res['end_alpha_mdl'] = resdf.loc['alpha'][-1]
    c_res['end_gdn_mdl'] = resdf.loc['g death'][-1]
    c_res['chg_dpm_mdl'] = (c_res['end_dpm_mdl'] - resdf.loc['eqDeath'][-2])/c_res['end_dpm_mdl']
    
    # Calculate mean Re and GOF statistics
    for var in means_list:
        c_res[f"avg_{var}"] = calc_mean(resdf, var, limit=hist_window)
    c_res['maeom'], c_res['mape'], c_res['r2'] = calc_gof(resdf, gof_vars[0], gof_vars[1])
    
    # Calculate various projections based on analytical approximation
    calc_eq_vals(c_res, eqtime, duration=duration) # for projected eqtime
    calc_end_vals(c_res, c_res['cum_dpm'], c_res['IFR'], 
                  endtime, delta, c_res['cum_dpm_del'], duration) # for end of run
    calc_eq_vals(c_res, earlytime, colnames=['ear_gdn', 'ear_alpha', 'ear_dpm'], 
                 duration=duration) # for estimate of early responsiveness

    # Calculate IQR and percentile values to append to country results
    iqrs, percs = generate_intervals(scriptname, c_res['cum_dpm'], c_res['IFR'], eqtime, endtime, 
                                     iqr_list, perc_list, delta, c_res['cum_dpm_del'], duration)
    c_res = pd.concat([c_res, iqrs, *percs])
    
    return c_res, datdf


def regress_deaths(dthdf):
    """Read in death and expected equilibrium death data, and regress 
    for all countries day by day, recording regression coefficients"""
    regdf = pd.DataFrame(index=dthdf.columns, columns=['n_R', 'RLM'])

    for i in dthdf.columns:
        # Correct for negative values and take log10
        Y_log = trunc_log(dthdf.loc['dpm'][i])
        X_log = trunc_log(dthdf.loc['eqDeath'][i])

        # If insufficient datapoints for date, skip and record NaN
        if Y_log.count() < 3:
            regdf.loc[i] = np.NaN
        
        # Otherwise run robust linear regression
        else:
            mod_RLM = sm.RLM(Y_log, X_log, missing='drop')
            fit_RLM = mod_RLM.fit()
            
            # Record observations and coefficient
            regdf.loc[i] = [fit_RLM.nobs, fit_RLM.params[0]]
    
    regdf.to_csv(f'./{baserunname}_regression.tab', sep='\t')
    
    return regdf


def compile_senslist(sens_vars, vals_dict, multipliers):
    """Compile setvals list for use with MultiScript for sensitivity 
    analysis, based on specified `multipliers` and parameters to test 
    as listed in `sens_vars`"""
    def lookup_dict(vars_list, vals_list):
        return [type(sub)(vals_list[var] for var in sub) for sub in vars_list]

    def lookup_mult(vars_list, mult):
        return [type(sub)(var * mult for var in sub) for sub in vars_list]

    # Pull corresponding values for sensitivity parameters
    base_vals = lookup_dict(sens_vars, vals_dict)
    
    # Generate suffix strings for runnames
    sfxs = [str(mult).replace('.','') for mult in multipliers]

    # Calculate setval values for sensitivity parameters
    mult_list = [lookup_mult(base_vals, mult) for mult in multipliers]

    # Compile & return list of setval tuples
    sens_list = [[(varnames, mults[i], sfxs[j]) for j, mults in enumerate(mult_list)] 
                 for i, varnames in enumerate(sens_vars)]

    return sens_list


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


# Set up files in run directory and initialise logfile
master = Script(cf)
master.changes.extend(scenariolist)
master.copy_model_files(f"{baserunname}_IterCal")
for f in [f"../{controlfilename}", "../ImportData.cmd", "../CovRegInput.frm"]:
    copy(f, "./")
logfile = f"{os.getcwd()}/{baserunname}.log"
write_log(f"-----\nStarting new log at {time.ctime()}\nReady to work!", logfile)


# In[ ]:


##### THIS CELL IS FOR UPDATING DATA ONLY #####

# Read main and mobility data from URL for raw data CSVs
data = pd.read_csv(data_url)
mobdata = pd.read_csv(mobdata_url)

# Extract dictionary mapping of ISO codes to OWID country names
names = data.filter(['iso_code','location'], axis=1).drop_duplicates()
names.replace({'iso_code': renames}, inplace=True) # Rename unusual ISO codes as needed
c_dict = dict(zip(names['location'], names['iso_code']))

# Subset CSV to relevant data fields
data = data.filter(['iso_code','date', 'total_cases', 'new_cases_smoothed', 
                    'new_deaths_smoothed_per_million', 'population', 'gdp_per_capita'], axis=1)

# Rename fields as needed
data.columns = ['iso_code','date', 'total_cases', 'new_cases', 
                'new_dpm', 'population', 'gdp_per_capita']
 
table = pd.pivot_table(data, values=['total_cases', 'new_cases', 'new_dpm', 'population', 
                                     'gdp_per_capita'], index='date', columns='iso_code')
table = table.T
table.index.names = ['field', 'iso_code']
table.columns = pd.to_datetime(table.columns)

# Drop countries with fewer cases than specified threshold, insufficient datapoints, or zero deaths
dropidx_cases = table.loc['total_cases'].index[table.loc['total_cases'].max(axis=1) < min_cases]
dropidx_deaths = table.loc['new_dpm'].index[table.loc['new_dpm'].max(axis=1) == 0]
first_idxs = (table.loc['total_cases'] > start_cases).idxmax(axis=1)
dropidx_data = table.loc['total_cases'].index[
    (table.columns[-1] - first_idxs).dt.days < min_datapoints]
print(dropidx_cases, dropidx_deaths, dropidx_data)
table.drop(dropidx_cases, level='iso_code', inplace=True, errors='ignore')
table.drop(dropidx_deaths, level='iso_code', inplace=True, errors='ignore')
table.drop(dropidx_data, level='iso_code', inplace=True, errors='ignore')
table.drop(droplist, level='iso_code', inplace=True, errors='ignore')

table = table.rename(index=renames) # Rename any unusual ISO codes as needed

# Separate country statistics columns for later use, then temporarily remove
popn = table.loc['population'].mean(axis=1)
gdppc = table.loc['gdp_per_capita'].mean(axis=1)

table.drop(['population', 'gdp_per_capita'], level='field', inplace=True, errors='ignore')

# Convert column indices to day number since startdate
table.columns = (table.columns - pd.to_datetime('2019-12-31')).days

# Reorder multiindex levels before by-country subsetting
table = table.reorder_levels(['iso_code', 'field']).sort_index()

# Identify first date over infection threshold for each country and subset dataframe accordingly
for i in table.index.levels[0]:
    first_idx = get_first_idx(table.loc[i].loc['total_cases'], start_cases)
    table.loc[i].loc[:, :first_idx] = np.NaN

# Clean infinite values and switch multiindex levels back
table.replace([np.inf, -np.inf], np.NaN, inplace=True)
table = table.reorder_levels(['field', 'iso_code']).sort_index()

# Calculate aggregate dpm data for later use
mean_dpm = table.loc['new_dpm'][-hist_window:].mean(axis=1) # Mean over last `hist_window` days

# Extract mobility change data to pivot table
mobdata['average'] = pd.concat([mobdata['retail_and_recreation'], mobdata['workplaces']], 
                               axis=1).mean(axis=1) # Get average of R&R and workplace values
mobdata.replace({'Country': c_dict}, inplace=True) # Convert country names to ISO codes
mobtable = pd.pivot_table(mobdata, values=['retail_and_recreation', 'workplaces', 'average'], 
                          index='Year', columns='Country')
mobtable = mobtable.T

# Calculate averages over last `hist_window` days & recompile into new dataframe
mobtable = mobtable[mobtable.columns[-hist_window:]]
tbm = mobtable.mean(axis=1)
mobmean = pd.concat([tbm.loc['average'], tbm.loc['retail_and_recreation'], tbm.loc['workplaces']], 
                    keys=['mob_avg', 'mob_rr', 'mob_wk'], axis=1)
display(mobmean)

# Export processed dataframes to .tab and import to VDF, or read in existing .tab
display(table)
if updatedata != 0:
    table.to_csv('./InputData.tab', sep='\t')
    subprocess.run(f"{vensim7path} \"./ImportData.cmd\"", check=True)
    mobmean.to_csv('./MobilityData.tab', sep='\t')
else:
    table = pd.read_csv(f'./InputData.tab', sep='\t', index_col=[0,1])
    mobmean = pd.read_csv('./MobilityData.tab', sep='\t')

# Update FinalTime cin with last day of available data - IMPORTANT! USES LAST FILE IN CHANGES LIST
finaltime = len(table.columns)-1
with open(simsettings['changes'][0], 'w') as f:
    f.write(f"FINAL TIME = {finaltime}")


# In[ ]:


##### MAIN ANALYSIS, DURATION SENSITIVITY & RESULTS-PROCESSING CODE #####

# Pull country list from data table
countrylist = list(table.index.levels[1])
print(countrylist)

basename = cf['baserunname']

# Loop through disease duration values to test, starting with main then sensitivity values
for i in ([main_dur] + sens_durs):
    cf['baserunname'] = f'{basename}{i}'
    baserunname = cf['baserunname']
    print(baserunname)

    # Create script object for given duration, to cleanly create calibration subfolder
    sub = Script(cf)
    sub.changes.extend(scenariolist)
    sub.copy_model_files(baserunname)
    copy(f"../{controlfilename}", "./")
    
    # Overwrite disease duration cin file - IMPORTANT! USES LAST FILE IN CHANGES LIST
    with open(simsettings['changes'][-1], 'w') as f:
        f.write(f"DiseaseDuration = {i}")
        
    dur = i # Assign disease duration variable
    
    # Initialise necessary .mdl and .voc files
    create_mdls(cf, countrylist, finaltime, logfile)
    
    # Run country-by-country calibration process, unless otherwise specified (mccores=0)
    if mccores != 0:
        write_log(f"Initialising MCMC with duration {dur}!", logfile)
        c_list = []
        err_list = []
        for c in countrylist:
            # First run Powell optimization, then MCMC
            res_i = compile_script(cf, CtyScript, c, 'i', {'model': f'_{c}', 'optparm': '_c'}, 
                                   logfile, subdir=c)
            if res_i != False:
                res = compile_script(cf, CtyMCScript, c, 'MC', {'model': f'_{c}', 'optparm': '_cmc'}, 
                                     logfile, chglist=[(c, 'i')], subdir=c)
                if res != False:
                    c_list.append(c) # Compile updated c_list of successful calibrations
                else:
                    err_list.append(c) # Compile error list of failed calibrations
            else:
                err_list.append(c) # Compile error list of failed calibrations
        write_log(f"Calibration complete! Error list is:\n{err_list}", logfile)
    
    # If calibration not needed, default to using country list from data as c_list
    else:
        write_log("Hang on to outdated imperialist dogma! Using previous output...", logfile)
        c_list = countrylist
        err_list = []

    write_log("Processing results!", logfile)
    
    # Initialise containers for processed country results and death data
    res_list = []
    dat_list = []
    
    # Loop through country MCMC outputs, calling master results processing function on each
    for c in c_list:
        try:
            c_res, datdf = process_results(f'./{c}/{baserunname}_{c}_MC', eqtime, earlytime, 
                                           gof_vars, iqr_list, means_list, perc_list, delta, dur)
            res_list.append(c_res)
            dat_list.append(datdf)
        except FileNotFoundError:
            err_list.append(c)
            
    # Compile main results dataframe with processed country results
    results = pd.concat(res_list, axis=1)
    
    # Compile country infection and death outputs over time
    dpm_data, eq_death, inf_data, inf_exp = [
        pd.concat([df.loc[var] for df in dat_list], axis=1) for var in [
            'DeathsOverTimeRaw', 'eqDeath', 'DataFlowOverTime', 'inf exp']]
    
    # Assign results dataframe indices based on c_list
    for df in [results, dpm_data, eq_death, inf_data, inf_exp]:
        df.columns = [c for c in c_list if c not in err_list]
    results, dpm_data, eq_death, inf_data, inf_exp = [
        df.T for df in [results, dpm_data, eq_death, inf_data, inf_exp]]

    # Recompile results dataframe with aggregate data previously separated
    results['mean_dpm'], results['population'], results['gdp_per_cap'] = mean_dpm, popn, gdppc
    
    # Recompile results dataframe with mobility data
    results = results.merge(mobmean, how='left', left_index=True, right_index=True)
        
    # Calculate normalised interquartile ranges (NIQRs)
    for var in ['eq_gdn', 'eq_dpm', 'end_gdn', 'end_dpm']:
        results[f'{var}_niqr'] = abs(results[f'{var}_0.75'] - results[f'{var}_0.25'])/results[var]
    
    display(results)
    
    # Compile infection outputs into multiindex dataframe for later graphing
    inf_results = pd.concat([inf_data, inf_exp], keys=['inf_data', 'inf_exp'])
    inf_results.index.names = ['field', 'iso_code']
    
    display(inf_results)
    
    # Compile death outputs into multiindex dataframe and run death regressions
    dth_results = pd.concat([dpm_data, eq_death], keys=['dpm', 'eqDeath'])
    dth_results.index.names = ['field', 'iso_code']
    
    display(dth_results)
    
    regdf = regress_deaths(dth_results)
    display(regdf)
    
    # Generate main output tab files and copy to root directory for easy access
    results.to_csv(f'./{baserunname}_results.tab', sep='\t')
    dth_results.to_csv(f'./{baserunname}_deaths.tab', sep='\t')
    inf_results.to_csv(f'./{baserunname}_infections.tab', sep='\t')
    
    copy(f'./{baserunname}_results.tab', '../')
    copy(f'./{baserunname}_deaths.tab', '../')
    copy(f'./{baserunname}_infections.tab', '../')
    copy(f'./{baserunname}_regression.tab', '../')
    
    os.chdir("..") # Remember to go back to root directory before next iteration!


# In[ ]:


##### SENSITIVITY ANALYSIS CODE FOR NBR TEST #####

# Prepare files for NBR (no behavioural response) sensitivity run
cf['baserunname'] = f'{basename}NBR'
baserunname = cf['baserunname']
print(baserunname)

# Copy in necessary voc and cin files to override normal parameters
vocname = cf['simsettings']['optparm'][:-4] + '_NBR.voc'
cf['simsettings']['optparm'] = vocname
cf['simsettings']['changes'].append('NBR.cin')
copy(f"../{vocname}", "./")
copy(f"../NBR.cin", "./")

for i in [main_dur]:
    # Create script object to cleanly create NBR subfolder
    sub = Script(cf)
    sub.changes.extend(scenariolist)
    sub.copy_model_files(baserunname)
    copy(f"../{controlfilename}", "./")

    # Overwrite disease duration cin file - IMPORTANT! USES LAST FILE IN CHANGES LIST
    with open('DiseaseDuration.cin', 'w') as f:
        f.write(f"DiseaseDuration = {i}")

    dur = i # Assign disease duration variable

    # Initialise necessary .mdl and .voc files
    create_mdls(cf, countrylist, finaltime, logfile)

    # Run country-by-country calibration process, unless otherwise specified (mccores=0)
    if mccores != 0:
        write_log(f"Initialising MCMC with duration {dur}!", logfile)
        c_list = []
        err_list = []
        for c in countrylist:
            # First run Powell optimization, then MCMC, checking for success each time
            res_i = compile_script(cf, CtyScript, c, 'i', {'model': f'_{c}', 'optparm': '_c'}, 
                                   logfile, subdir=c)
            if res_i != False:
                res = compile_script(cf, CtyMCScript, c, 'MC', {
                    'model': f'_{c}', 'optparm': '_cmc'}, logfile, chglist=[(c, 'i')], subdir=c)
                if res != False:
                    c_list.append(c) # Compile updated c_list of successful calibrations
                else:
                    err_list.append(c) # Compile error list of failed calibrations
            else:
                err_list.append(c) # Compile error list of failed calibrations
        write_log(f"Calibration complete! Error list is:\n{err_list}", logfile)

    # If calibration not needed, default to using country list from data as c_list
    else:
        write_log("Hang on to outdated imperialist dogma! Using previous output...", logfile)
        c_list = countrylist
        err_list = []

    write_log("Processing results!", logfile)

    # Initialise containers for processed country results and death data
    res_list = []
    dat_list = []

    # Loop through country MCMC outputs, pulling time series results & GOF measures from each
    for c in c_list:
        try:
            resdf = pd.read_csv(f'./{c}/{baserunname}_{c}_MC.tab', sep='\t', 
                                index_col=0, error_bad_lines=False)
            resdf.index = [n.split('[')[0] for n in resdf.index] # Separate subscripts
            datdf = resdf.loc[['DataFlowOverTime', 'inf exp']]
            
            c_res = pd.Series()
            c_res['maeom'], c_res['mape'], c_res['r2'] = calc_gof(resdf, gof_vars[0], gof_vars[1])
            
            res_list.append(c_res)
            dat_list.append(datdf)
        except FileNotFoundError:
            err_list.append(c)

    # Compile main results dataframe with country GOF measures
    results = pd.concat(res_list, axis=1)

    # Compile country infection outputs over time
    inf_data = pd.concat([df.loc['DataFlowOverTime'] for df in dat_list], axis=1)
    inf_exp = pd.concat([df.loc['inf exp'] for df in dat_list], axis=1)

    # Assign results dataframe indices based on c_list
    for df in [results, inf_data, inf_exp]:
        df.columns = [c for c in c_list if c not in err_list]
    results, inf_data, inf_exp = [df.T for df in [results, inf_data, inf_exp]]

    display(results)

    # Compile infection outputs into multiindex dataframe for later graphing
    inf_results = pd.concat([inf_data, inf_exp], keys=['inf_data', 'inf_exp'])
    inf_results.index.names = ['field', 'iso_code']

    display(inf_results)

    # Generate main output tab files and copy to root directory for easy access
    results.to_csv(f'./{baserunname}_results.tab', sep='\t')
    inf_results.to_csv(f'./{baserunname}_infections.tab', sep='\t')

    copy(f'./{baserunname}_results.tab', '../')
    copy(f'./{baserunname}_infections.tab', '../')

    os.chdir("..") # Remember to go back to root directory before next iteration!


# In[ ]:


##### PARAMETER SENSITIVITY ANALYSIS WITH ILLUSTRATIVE MODEL #####

# Prepare files for parameter sensitivity run, copying in necessary .mdl
copy('../SensitivitySIR.mdl', './')
smcf = cf.copy()
smcf['simsettings']['model'] = 'SensitivitySIR.mdl'
smcf['simsettings']['savelist'] = smcf['simsettings']['senssavelist']
smcf['simsettings']['changes'] = ['DiseaseDuration.cin', 'BaseValues.cin']

# Create script object to cleanly create Sensitivity subfolder
sub = Script(smcf)
sub.copy_model_files('Sensitivity')
copy(f"../{controlfilename}", "./")

for i in [maindur]:
    smcf['baserunname'] = f'{basename}{i}'
    baserunname = smcf['baserunname']
    print(baserunname)
    
    # Overwrite disease duration cin file
    with open('DiseaseDuration.cin', 'w') as f:
        f.write(f"DiseaseDuration = {i}")
        
    # Read base values from cin file to use to compile setvals list
    with open('BaseValues.cin', 'r') as f:
        lines = f.readlines()
        keys = [line.split('=')[0].strip() for line in lines]
        vals = [float(line.split('=')[1]) for line in lines]
        vals_dict = dict(zip(keys, vals))
    
    # Compile setvals list for sensitivity runs
    sens_list = compile_senslist(sens_vars, vals_dict, sens_mults)

    # Extend time horizon and run base case
    with open('ExtTime.cin', 'w') as f:
        f.write("FINAL TIME = 730")
    compile_script(smcf, ScenRunScript, 'sens', 'base', {}, logfile, chglist=['ExtTime.cin'])

    # Initialise containers for Re and DPM results
    redf_list = []
    dthdf_list = []
    
    # Loop through setvals list, running each parameter's setval scenarios
    for setvals in sens_list:
        try:
            os.remove(f'{baserunname}_sens_{setvals[0][0][0]}.tab') # Clear existing tab files
        except FileNotFoundError:
            pass
        # Run MultiScript sensitivity scenarios and extract Re and DPM results for graphing
        compile_script(smcf, MultiScript, 'sens', setvals[0][0][0], {}, logfile, setvals=setvals)
        df = pd.read_csv(f'{baserunname}_sens_{setvals[0][0][0]}.tab', sep='\t', index_col=[0,1])
        redf, dthdf = df.loc['Re'], df.loc['Deaths']
        redf_list.append(redf)
        dthdf_list.append(dthdf)

    # Compile and export Re and DPM results, clearing any duplicates from incomplete runs
    redf = pd.concat(redf_list)
    redf = redf[~redf.index.duplicated(keep='first')]
    redf.to_csv(f'{baserunname}_sens_Re.tab', sep='\t')
    
    dthdf = pd.concat(dthdf_list)
    dthdf = dthdf[~dthdf.index.duplicated(keep='first')]
    dthdf.to_csv(f'{baserunname}_sens_Death.tab', sep='\t')
    
    # Copy outputs to root directory for easy access
    copy(f'./{baserunname}_sens_base.tab', '../')
    copy(f'./{baserunname}_sens_Re.tab', '../')
    copy(f'./{baserunname}_sens_Death.tab', '../')
    
    os.chdir("..") # Remember to go back to root directory before next iteration!


# In[ ]:





# In[ ]:


000000000000000000000000000000000000000000000000000000000000000000000000
000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

