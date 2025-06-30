import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import lmfit
import traceback
from bindcurve import data
from bindcurve import models




def generate_guess(df, saturation=False):
    
    # Sorting the df
    if saturation:
        df = df.sort_values(by=['c'], ascending=True)
    else:
        df = df.sort_values(by=['c'], ascending=False)
    
    # Defining important points on "median response" axis
    ymin_guess = min(df["median"])
    ymax_guess = max(df["median"])
    y_middle = ymin_guess+(ymax_guess-ymin_guess)/2
    
    # Interpolating to obtain guess for concentration axis
    IC50_guess = np.interp(y_middle, df["median"], df["c"])
    
 
    # This is plotting just for development purposes
    #y_curve = np.linspace(ymin_guess, ymax_guess, 1000)
    #plt.plot(df["median"], df["c"], "o")
    #plt.plot(y_curve, np.interp(y_curve, df["median"], df["c"]))
    #plt.yscale("log")
    #plt.show()
    
    return ymin_guess, ymax_guess, IC50_guess



def define_pars(model, ymin_guess, ymax_guess, IC50_guess, RT=None, LsT=None, Kds=None, Ns=False, N=False, fix_ymin=False, fix_ymax=False, fix_slope=False):
    
        # Initiating Parameters class in lmfit
        pars = lmfit.Parameters()
        
        # Setting ymin and ymax
        if fix_ymin==False:
            pars.add('ymin', value = ymin_guess)
        else:
            pars.add('ymin', value = fix_ymin, vary=False)
            
        if fix_ymax==False:
            pars.add('ymax', value = ymax_guess)
        else:
            pars.add('ymax', value = fix_ymax, vary=False)  
                
        # Setting parameters for the logistic models
        if model in models.get_list_of_models("logistic"):
           
            if not fix_slope:
                pars.add('slope', value = 0)
            else:
                pars.add('slope', value = fix_slope, vary=False) 
                
            if model == "IC50":
                pars.add('IC50', value = IC50_guess, min = 0)
            if model == "logIC50":
                pars.add('logIC50', value = np.log10(IC50_guess))


        # Setting parameters for the direct binding Kd models
        if model in models.get_list_of_models("Kd_direct"):
            # Experimental constants
            pars.add('LsT', value = LsT, vary=False)
            # Parameters to be fitted
            pars.add('Kds', value = IC50_guess/2, min = 0)           
                
            if model == "dir_total":
                pars.add('Ns', value = Ns, vary=False)
        if model == "dir_simple":
            pars.add('Kds', value = IC50_guess/2, min = 0)

        # Setting parameters for the competitive binding Kd models
        if model in models.get_list_of_models("Kd_competition"):
            # Experimental constants
            pars.add('RT', value = RT, vary=False)
            pars.add('LsT', value = LsT, vary=False)
            pars.add('Kds', value = Kds, vary=False) 
        
            # Parameters to be fitted
            pars.add('Kd', value = IC50_guess/2, min=0)
            
            if model in ["comp_3st_total", "comp_4st_total"]:
                pars.add('N', value = N, vary=False)
                
            if model in ["comp_4st_specific", "comp_4st_total"]:
                pars.add('Kd3', value = (IC50_guess/2)*10, min=0)

        
        return pars
    
    
    
    
def fit_50(input_df, model, compound_sel = False, fix_ymin = False, fix_ymax = False, fix_slope = False, ci=True, verbose = False):
    """Function for fitting the `IC50` and `logIC50` models.
    
    Parameters
    ----------
    input_df : DataFrame
        Pandas DataFrame containing the input data.
    model : str
        Name of the model. Options: `IC50`, `logIC50`
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds will be used.
    fix_ymin : float or int
        Lower asymptote of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    fix_ymax : float or int
        Upper asymptote of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    fix_slope : float or int
        Slope of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    ci : bool
        Whether to calculate 95% confidence intervals.
    verbose : bool
        If set to "True", more detailed output is printed. Intended mainly for troubleshooting.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the fit results.
    """
    
    print("Fitting", model, "...")
    
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if not compound_sel:
        compounds = input_df["compound"].unique()
    else:
        compounds = compound_sel
        
    # Initiating empty output_df
    if model == "IC50":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'IC50', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'slope', 'Chi^2', 'R^2' ])
    if model == "logIC50":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'logIC50', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'slope', 'Chi^2', 'R^2' ])
    
    
    for compound in compounds:
        
        df_compound = input_df[input_df["compound"].isin([compound])]
        df_compound_pooled = data.pool_data(df_compound)
        
        # Generating initial guesses
        ymin_guess, ymax_guess, IC50_guess = generate_guess(df_compound)
    

        # Defining x and y
        if model == "IC50":
            x = df_compound_pooled["c"]
        if model == "logIC50":
            x = df_compound_pooled["log c"]
            
        y = df_compound_pooled["response"] 
    

        # Setting up the initial parameter values
        pars = define_pars(model, ymin_guess, ymax_guess, IC50_guess, fix_ymin=fix_ymin, fix_ymax=fix_ymax, fix_slope=fix_slope)

        try:
            # Here is the actual fit in lmfit, the function is called from the "models" module
            if model == "IC50":
                fitter = lmfit.Minimizer(models.IC50_lmfit, pars, fcn_args=(x, y))
            if model == "logIC50":
                fitter = lmfit.Minimizer(models.logIC50_lmfit, pars, fcn_args=(x, y))            

            result = fitter.minimize()
            
            # Getting Chi^2 from result container
            Chi_squared = result.chisqr
            # Calculating R^2
            R_squared = 1 - result.residual.var() / np.var(y)
            
            fitted_parameter = model
            
            # Calculating confidence intervals at 2 sigmas (95%)
            if ci:
                ci = lmfit.conf_interval(fitter, result, p_names = [fitted_parameter], sigmas=[2])
                ci_listoftuples = ci.get(fitted_parameter)
                
                loCL = ci_listoftuples[0][1]    # This is the lower confidence limit at 2 sigmas (95%)
                upCL = ci_listoftuples[2][1]    # This is the upper confidence limit at 2 sigmas (95%)
            else:
                loCL = "nd"
                upCL = "nd"
                    
            
            # Printing verbose output if verbose=True
            if verbose:
                print()
                print("===Compound:", compound)
                print()
                print("Data for compound:\n", df_compound_pooled)
                print()        
                print("---Initial guesses:")
                print("ymin_guess:", ymin_guess)
                print("ymax_guess:", ymax_guess)
                print("IC50_guess:", IC50_guess)
                print()            
                print("---Fitting results:")
                print(lmfit.fit_report(result))
                print()
                print("Chi_squared:", Chi_squared)
                print("R_squared:", R_squared)
                print()            
                if ci:
                    print("---Confidence intervals:")
                    lmfit.printfuncs.report_ci(ci)
                print()

        
            # Creating new row for the output dataframe      
            new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['slope'].value,Chi_squared,R_squared]
                    
                    
            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
         
        except Exception:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
              
    return output_df



def fit_Kd_direct(input_df, model, LsT, Ns=None, compound_sel = False, fix_ymin = False, fix_ymax = False, ci=True, verbose = False):
    """Function for fitting the `dir_simple`, `dir_specific` and `dir_total` models.
    
    Parameters
    ----------
    input_df : DataFrame
        Pandas DataFrame containing the input data.
    model : str
        Name of the model. Options: `dir_simple`, `dir_specific`, `dir_total`
    LsT : float or int
        Total concentration of the labeled ligand.
    Ns : float or int
        Parameter for nonspecific binding of the labeled ligand (needed only for `dir_total` model).
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds will be used.
    fix_ymin : float or int
        Lower asymptote of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    fix_ymax : float or int
        Upper asymptote of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    ci : bool
        Whether to calculate 95% confidence intervals.
    verbose : bool
        If set to "True", more detailed output is printed. Intended mainly for troubleshooting.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the fit results.
    """
    
    print("Fitting", model, "...")
    
    
    # Initial checks
    if fix_ymin and fix_ymax:
        ci=False
        print("Only one parameter is fitted. Confidence intervals will not be calculated.")
        
      
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if not compound_sel:
        compounds = input_df["compound"].unique()
    else:
        compounds = compound_sel


    # Initiating empty output_df
    if model == "dir_simple":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kds', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'Chi^2', 'R^2'])
    if model == "dir_specific":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kds', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'LsT', 'Chi^2', 'R^2'])
    if model == "dir_total":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kds', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'LsT', 'Ns', 'Chi^2', 'R^2'])
  
    for compound in compounds:
        
        df_compound = input_df[input_df["compound"].isin([compound])]
        df_compound_pooled = data.pool_data(df_compound)
        
        # Generating initial guesses
        ymin_guess, ymax_guess, IC50_guess = generate_guess(df_compound, saturation=True)


        # Defining x and y
        x = df_compound_pooled["c"]   
        y = df_compound_pooled["response"] 
    
        if model == "dir_simple":
            LsT=None

        # Setting up the initial parameter values
        pars = define_pars(model, ymin_guess, ymax_guess, IC50_guess, LsT=LsT, Ns=Ns, fix_ymin=fix_ymin, fix_ymax=fix_ymax)
        
        
        try:
            # Here is the actual fit in lmfit, the function is called from the "models" module
            if model == "dir_simple":
                fitter = lmfit.Minimizer(models.dir_simple_lmfit, pars, fcn_args=(x, y))
            if model == "dir_specific":
                fitter = lmfit.Minimizer(models.dir_specific_lmfit, pars, fcn_args=(x, y))
            if model == "dir_total":
                fitter = lmfit.Minimizer(models.dir_total_lmfit, pars, fcn_args=(x, y))

            result = fitter.minimize()

            # Getting Chi^2 from result container
            Chi_squared = result.chisqr
            # Calculating R^2
            R_squared = 1 - result.residual.var() / np.var(y)
            
            fitted_parameter = "Kds"
            
            # Calculating confidence intervals at 2 sigmas (95%)
            if ci:
                ci = lmfit.conf_interval(fitter, result, p_names = [fitted_parameter], sigmas=[2])
                ci_listoftuples = ci.get(fitted_parameter)
                
                loCL = ci_listoftuples[0][1]    # This is the lower confidence limit at 2 sigmas (95%)
                upCL = ci_listoftuples[2][1]    # This is the upper confidence limit at 2 sigmas (95%)
            else:
                loCL = "nd"
                upCL = "nd"
                    
            
            # Printing verbose output if verbose=True
            if verbose:
                print()
                print("===Compound:", compound)
                print()
                print("Data for compound:\n", df_compound_pooled)
                print()        
                print("---Initial guesses:")
                print("ymin_guess:", ymin_guess)
                print("ymax_guess:", ymax_guess)
                print("IC50_guess:", IC50_guess)
                print("Kds_guess:", IC50_guess/2)
                print()            
                print("---Fitting results:")
                print(lmfit.fit_report(result))
                print()
                print("Chi_squared:", Chi_squared)
                print("R_squared:", R_squared)
                print()            
                if ci:
                    print("---Confidence intervals:")
                    lmfit.printfuncs.report_ci(ci)
                print()

            
            # Creating new row for the output dataframe  
            if model == "dir_simple":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, Chi_squared, R_squared]
            if model == "dir_specific":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['LsT'].value, Chi_squared, R_squared]
            if model == "dir_total":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['LsT'].value, result.params['Ns'].value, Chi_squared, R_squared]
        

            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
        
        except Exception:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
                
    return output_df




def fit_Kd_competition(input_df, model, RT, LsT, Kds, N=None, compound_sel = False, fix_ymin = False, fix_ymax = False, ci=True, verbose = False):
    """Function for fitting the `comp_3st_specific`, `comp_3st_total`, `comp_4st_specific` and `comp_4st_total` models.
    
    Parameters
    ----------
    input_df : DataFrame
        Pandas DataFrame containing the input data.
    model : str
        Name of the model. Options: `comp_3st_specific`, `comp_3st_total`, `comp_4st_specific`, `comp_4st_total`
    RT : float or int
        Total concentration of the receptor.
    LsT : float or int
        Total concentration of the labeled ligand.
    Kds : float or int
        Dissociation constant of the labeled ligand.
    N : float or int
        Parameter for nonspecific binding of the unlabeled ligand (needed only for `comp_3st_total` and `comp_4st_total` models).
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds will be used.
    fix_ymin : float or int
        Lower asymptote of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    fix_ymax : float or int
        Upper asymptote of the model will be fixed at the provided value. If set to "False", it will be fitted freely.
    ci : bool
        Whether to calculate 95% confidence intervals.
    verbose : bool
        If set to "True", more detailed output is printed. Intended mainly for troubleshooting.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the fit results.
    """
    
    print("Fitting", model, "...")
    
    # Initial checks
    if fix_ymin and fix_ymax:
        ci=False
        print("Only one parameter is fitted. Confidence intervals will not be calculated.")
        
    
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if not compound_sel:
        compounds = input_df["compound"].unique()
    else:
        compounds = compound_sel
        
    # Initiating empty output_df
    if model == "comp_3st_specific":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'RT', 'LsT', 'Kds', 'Chi^2', 'R^2' ])
    if model == "comp_3st_total":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'RT', 'LsT', 'Kds', 'N', 'Chi^2', 'R^2' ])
    if model == "comp_4st_specific":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'RT', 'LsT', 'Kds', 'Kd3', 'Chi^2', 'R^2' ])
    if model == "comp_4st_total":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'ymin', 'ymax', 'RT', 'LsT', 'Kds', 'Kd3', 'N', 'Chi^2', 'R^2' ])
   
    
    for compound in compounds:
        
        df_compound = input_df[input_df["compound"].isin([compound])]
        df_compound_pooled = data.pool_data(df_compound)
        
        # Generating initial guesses
        ymin_guess, ymax_guess, IC50_guess = generate_guess(df_compound)
    

        # Defining x and y
        x = df_compound_pooled["c"]   
        y = df_compound_pooled["response"] 
    

        # Setting up the initial parameter values
        pars = define_pars(model, ymin_guess, ymax_guess, IC50_guess, RT=RT, LsT=LsT, Kds=Kds, N=N, fix_ymin=fix_ymin, fix_ymax=fix_ymax)
        

        try:
            # Here is the actual fit in lmfit, the function is called from the "models" module
            if model == "comp_3st_specific":
                fitter = lmfit.Minimizer(models.comp_3st_specific_lmfit, pars, fcn_args=(x, y))
            if model == "comp_3st_total":
                fitter = lmfit.Minimizer(models.comp_3st_total_lmfit, pars, fcn_args=(x, y))
            if model == "comp_4st_specific":
                fitter = lmfit.Minimizer(models.comp_4st_specific_lmfit, pars, fcn_args=(x, y))
            if model == "comp_4st_total":
                fitter = lmfit.Minimizer(models.comp_4st_total_lmfit, pars, fcn_args=(x, y))
            
            result = fitter.minimize()


            # Getting Chi^2 from result container
            Chi_squared = result.chisqr
            # Calculating R^2
            R_squared = 1 - result.residual.var() / np.var(y)
            
            fitted_parameter = "Kd"
            
            # Calculating confidence intervals at 2 sigmas (95%)
            if ci:
                ci = lmfit.conf_interval(fitter, result, p_names = [fitted_parameter], sigmas=[2])
                ci_listoftuples = ci.get(fitted_parameter)
                
                loCL = ci_listoftuples[0][1]    # This is the lower confidence limit at 2 sigmas (95%)
                upCL = ci_listoftuples[2][1]    # This is the upper confidence limit at 2 sigmas (95%)
            else:
                loCL = "nd"
                upCL = "nd"
                    
            
            # Printing verbose output if verbose=True
            if verbose:
                print()
                print("===Compound:", compound)
                print()
                print("Data for compound:\n", df_compound_pooled)
                print()        
                print("---Initial guesses:")
                print("ymin_guess:", ymin_guess)
                print("ymax_guess:", ymax_guess)
                print("IC50_guess:", IC50_guess)
                print("Kd_guess:", IC50_guess/2)
                print()            
                print("---Fitting results:")
                print(lmfit.fit_report(result))
                print()
                print("Chi_squared:", Chi_squared)
                print("R_squared:", R_squared)
                print()            
                if ci:
                    print("---Confidence intervals:")
                    lmfit.printfuncs.report_ci(ci)
                print()

        
            # Creating new row for the output dataframe  
            if model == "comp_3st_specific":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, Chi_squared, R_squared]
            if model == "comp_3st_total":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, result.params['N'].value, Chi_squared, R_squared]
            if model == "comp_4st_specific":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, result.params['Kd3'].value, Chi_squared, R_squared]
            if model == "comp_4st_total":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['ymin'].value, result.params['ymax'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, result.params['Kd3'].value, result.params['N'].value, Chi_squared, R_squared]


            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
            
        except Exception:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
                
    return output_df




def convert(IC50_df, model, RT=None, LsT=None, Kds=None, y0=None, compound_sel=False, ci=True, verbose=False):
    """Function for converting IC50 to Kd using `coleska`, `cheng_prusoff` and `cheng_prusoff_corr` models.
    
    Parameters
    ----------
    IC50_df : DataFrame
        Pandas DataFrame containing the fitted IC50 values.
    model : str
        Name of the conversion model. Options: `coleska`, `cheng_prusoff`, `cheng_prusoff_corr`
    RT : float or int
        Total concentration of the receptor.
    LsT : float or int
        Total concentration of the labeled ligand.
    Kds : float or int
        Dissociation constant of the labeled ligand.
    y0 : float or int
        Parameter used in the corrected Cheng-Prusoff model.
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds will be used.
    ci : bool
        Whether to calculate 95% confidence intervals.
    verbose : bool
        If set to "True", more detailed output is printed. Intended mainly for troubleshooting.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the conversion results.
    """
    
    print("Converting IC50 to Kd using", model, "model...")
    
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if not compound_sel:
        compounds = IC50_df["compound"].unique()
    else:
        compounds = compound_sel
        
    if 'IC50' not in IC50_df.columns:
        exit("Provided dataframe does not contain IC50 column. Aborting...")
        
    # If the provided df contains no CL, than only convert means 
    if IC50_df["loCL"].iloc[0] == "nd" and IC50_df["upCL"].iloc[0] == "nd":
        ci=False
        print("Confidence limits not detected in the provided dataframe. Converting only mean values...")
    if not ci:
        loCL = "nd"
        upCL = "nd"

    # Initiating empty output_df
    output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model'])

    for compound in compounds:
        
        df_compound = IC50_df[IC50_df["compound"].isin([compound])]
        
        try:
            # Here are the actual conversions
            if model == "cheng_prusoff":
                Kd = models.cheng_prusoff(LsT, Kds, df_compound["IC50"].iloc[0])
                if ci:
                    loCL = models.cheng_prusoff(LsT, Kds, df_compound["loCL"].iloc[0])
                    upCL = models.cheng_prusoff(LsT, Kds, df_compound["upCL"].iloc[0])
            if model == "cheng_prusoff_corr":
                Kd = models.cheng_prusoff_corr(LsT, Kds, y0, df_compound["IC50"].iloc[0])
                if ci:
                    loCL = models.cheng_prusoff_corr(LsT, Kds, y0, df_compound["loCL"].iloc[0])
                    upCL = models.cheng_prusoff_corr(LsT, Kds, y0, df_compound["upCL"].iloc[0])
            if model == "coleska":
                Kd = models.coleska(RT, LsT, Kds, df_compound["IC50"].iloc[0])
                if ci:
                    loCL = models.coleska(RT, LsT, Kds, df_compound["loCL"].iloc[0])
                    upCL = models.coleska(RT, LsT, Kds, df_compound["upCL"].iloc[0])

            # Creating new row for the output dataframe  
            new_row = [compound, 1, Kd, loCL, upCL, 'nd', model]
        
            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
            
        except Exception:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
    
    return output_df