import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import models
import data
import traceback

def generate_guess(df, saturation=False):
    
    # Sorting the df
    if saturation:
        df = df.sort_values(by=['c'], ascending=True)
    else:
        df = df.sort_values(by=['c'], ascending=False)
    
    # Defining important points on "median response" axis
    min_guess = min(df["median"])
    max_guess = max(df["median"])
    y_middle = min_guess+(max_guess-min_guess)/2
    
    # Interpolating to obtain guess for concentration axis
    IC50_guess = np.interp(y_middle, df["median"], df["c"])
    
 
    # This is plotting just for development purposes
    #y_curve = np.linspace(min_guess, max_guess, 1000)
    #plt.plot(df["median"], df["c"], "o")
    #plt.plot(y_curve, np.interp(y_curve, df["median"], df["c"]))
    #plt.yscale("log")
    #plt.show()
    
    return min_guess, max_guess, IC50_guess



def define_pars(model, min_guess, max_guess, IC50_guess, RT=None, LsT=None, Kds=None, Ns=False, N=False, fix_min=False, fix_max=False, fix_slope=False):
    
        # Initiating Parameters class in lmfit
        pars = lmfit.Parameters()
        
        # Setting min and max
        if fix_min == False:
            pars.add('min', value = min_guess)
        else:
            pars.add('min', value = fix_min, vary=False)
            
        if fix_max == False:
            pars.add('max', value = max_guess)
        else:
            pars.add('max', value = fix_max, vary=False)  
                
        # Setting parameters for the logistic models
        if model in models.get_list_of_models("logistic"):
           
            if fix_slope == False:
                pars.add('slope', value = -1)
            else:
                pars.add('slope', value = fix_slope, vary=False) 
                
            if model == "IC50":
                pars.add('IC50', value = IC50_guess, min = 0)
            if model == "logIC50":
                pars.add('logIC50', value = np.log10(IC50_guess))


        # Setting parameters for the direct binding Kd models
        if model in models.get_list_of_models("Kd_saturation"):
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
    
    
    
    
def fit_50(df, model, compound_sel = False, fix_min = False, fix_max = False, fix_slope = False, ci=True, verbose = False):
    print("Fitting", model, "...")
    
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if compound_sel == False:
        compounds = df["compound"].unique()
    else:
        compounds = compound_sel
        
    # Initiating empty output_df
    if model == "IC50":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'IC50', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'slope', 'Chi^2', 'R^2' ])
    if model == "logIC50":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'logIC50', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'slope', 'Chi^2', 'R^2' ])
    
    
    for compound in compounds:
        
        df_compound = df[df["compound"].isin([compound])]
        df_compound_pooled = data.pool_data(df_compound)
        
        # Generating initial guesses
        min_guess, max_guess, IC50_guess = generate_guess(df_compound)
    

        # Defining x and y
        if model == "IC50":
            x = df_compound_pooled["c"]
        if model == "logIC50":
            x = df_compound_pooled["log c"]
            
        y = df_compound_pooled["response"] 
    

        # Setting up the initial parameter values
        pars = define_pars(model, min_guess, max_guess, IC50_guess, fix_min=fix_min, fix_max=fix_max, fix_slope=fix_slope)

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
                print("min_guess:", min_guess)
                print("max_guess:", max_guess)
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
                        model, result.params['min'].value, result.params['max'].value, result.params['slope'].value,Chi_squared,R_squared]
                    
                    
            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
         
        except:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
              
    return output_df



def fit_Kd_saturation(df, model, LsT, Ns=None, compound_sel = False, fix_min = False, fix_max = False, ci=True, verbose = False):
    print("Fitting", model, "...")
    
    
    # Initial checks
    if fix_min != False and fix_max != False:
        ci=False
        print("Only one parameter is fitted. Confidence intervals will not be calculated.")
        
      
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if compound_sel == False:
        compounds = df["compound"].unique()
    else:
        compounds = compound_sel


    # Initiating empty output_df
    if model == "dir_simple":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kds', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'Chi^2', 'R^2'])
    if model == "dir_specific":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kds', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'LsT', 'Chi^2', 'R^2'])
    if model == "dir_total":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kds', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'LsT', 'Ns', 'Chi^2', 'R^2'])
  
    for compound in compounds:
        
        df_compound = df[df["compound"].isin([compound])]
        df_compound_pooled = data.pool_data(df_compound)
        
        # Generating initial guesses
        min_guess, max_guess, IC50_guess = generate_guess(df_compound, saturation=True)


        # Defining x and y
        x = df_compound_pooled["c"]   
        y = df_compound_pooled["response"] 
    
        if model == "dir_simple":
            LsT=None

        # Setting up the initial parameter values
        pars = define_pars(model, min_guess, max_guess, IC50_guess, LsT=LsT, Ns=Ns, fix_min=fix_min, fix_max=fix_max)
        
        
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
                print("min_guess:", min_guess)
                print("max_guess:", max_guess)
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
                        model, result.params['min'].value, result.params['max'].value, Chi_squared, R_squared]
            if model == "dir_specific":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['min'].value, result.params['max'].value, result.params['LsT'].value, Chi_squared, R_squared]
            if model == "dir_total":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['min'].value, result.params['max'].value, result.params['LsT'].value, result.params['Ns'].value, Chi_squared, R_squared]
        

            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
        
        except:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
                
    return output_df




def fit_Kd(df, model, RT, LsT, Kds, N=None, compound_sel = False, fix_min = False, fix_max = False, ci=True, verbose = False):
    print("Fitting", model, "...")
    
    # Initial checks
    if fix_min != False and fix_max != False:
        ci=False
        print("Only one parameter is fitted. Confidence intervals will not be calculated.")
        
    
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if compound_sel == False:
        compounds = df["compound"].unique()
    else:
        compounds = compound_sel
        
    # Initiating empty output_df
    if model == "comp_3st_specific":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'RT', 'LsT', 'Kds', 'Chi^2', 'R^2' ])
    if model == "comp_3st_total":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'RT', 'LsT', 'Kds', 'N', 'Chi^2', 'R^2' ])
    if model == "comp_4st_specific":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'RT', 'LsT', 'Kds', 'Kd3', 'Chi^2', 'R^2' ])
    if model == "comp_4st_total":
        output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model', 'min', 'max', 'RT', 'LsT', 'Kds', 'Kd3', 'N', 'Chi^2', 'R^2' ])
   
    
    for compound in compounds:
        
        df_compound = df[df["compound"].isin([compound])]
        df_compound_pooled = data.pool_data(df_compound)
        
        # Generating initial guesses
        min_guess, max_guess, IC50_guess = generate_guess(df_compound)
    

        # Defining x and y
        x = df_compound_pooled["c"]   
        y = df_compound_pooled["response"] 
    

        # Setting up the initial parameter values
        pars = define_pars(model, min_guess, max_guess, IC50_guess, RT=RT, LsT=LsT, Kds=Kds, N=N, fix_min=fix_min, fix_max=fix_max)
        

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
                print("min_guess:", min_guess)
                print("max_guess:", max_guess)
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
                        model, result.params['min'].value, result.params['max'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, Chi_squared, R_squared]
            if model == "comp_3st_total":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['min'].value, result.params['max'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, result.params['N'].value, Chi_squared, R_squared]
            if model == "comp_4st_specific":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['min'].value, result.params['max'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, result.params['Kd3'].value, Chi_squared, R_squared]
            if model == "comp_4st_total":    
                new_row = [compound, result.ndata, result.params[fitted_parameter].value, loCL, upCL, result.params[fitted_parameter].stderr, 
                        model, result.params['min'].value, result.params['max'].value, result.params['RT'].value, result.params['LsT'].value, result.params['Kds'].value, result.params['Kd3'].value, result.params['N'].value, Chi_squared, R_squared]


            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
            
        except:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
                
    return output_df




def convert(df, model, RT=None, LsT=None, Kds=None, y0=None, compound_sel=False, ci=True, verbose=False):
    print("Converting IC50 to Kd using", model, "model...")
    
    # In compound selection is provided, than use it, otherwise calculate fit for all compounds
    if compound_sel == False:
        compounds = df["compound"].unique()
    else:
        compounds = compound_sel
        
    if 'IC50' not in df.columns:
        exit("Provided dataframe does not contain IC50 column. Aborting...")
        
    # If the provided df contains no CL, than only convert means 
    if df["loCL"].iloc[0] == "nd" and df["upCL"].iloc[0] == "nd":
        ci=False
        print("Confidence limits not detected in the provided dataframe. Converting only mean values...")
    if ci==False:
        loCL = "nd"
        upCL = "nd"

    # Initiating empty output_df
    output_df = pd.DataFrame(columns=['compound', 'n_points', 'Kd', 'loCL', 'upCL', 'SE', 'model'])

    for compound in compounds:
        
        df_compound = df[df["compound"].isin([compound])]
        
        try:
            # Here are the actual conversions
            if model == "cheng_prusoff":
                Kd = models.cheng_prusoff(LsT, Kds, df_compound["IC50"].iloc[0])
                if ci==True:
                    loCL = models.cheng_prusoff(LsT, Kds, df_compound["loCL"].iloc[0])
                    upCL = models.cheng_prusoff(LsT, Kds, df_compound["upCL"].iloc[0])
            if model == "cheng_prusoff_corrected":
                Kd = models.cheng_prusoff_corrected(LsT, Kds, y0, df_compound["IC50"].iloc[0])
                if ci==True:
                    loCL = models.cheng_prusoff_corrected(LsT, Kds, y0, df_compound["loCL"].iloc[0])
                    upCL = models.cheng_prusoff_corrected(LsT, Kds, y0, df_compound["upCL"].iloc[0])
            if model == "coleska":
                Kd = models.coleska(RT, LsT, Kds, df_compound["IC50"].iloc[0])
                if ci==True:
                    loCL = models.coleska(RT, LsT, Kds, df_compound["loCL"].iloc[0])
                    upCL = models.coleska(RT, LsT, Kds, df_compound["upCL"].iloc[0])

            # Creating new row for the output dataframe  
            new_row = [compound, 1, Kd, loCL, upCL, 'nd', model]
        
            # Adding new row to the output dataframe
            output_df.loc[len(output_df)] = new_row
            
        except:
            print("Calculation for compound " + compound + " failed.")
            if verbose:
                traceback.print_exc()  
    
    return output_df