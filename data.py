import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lmfit
import models


def load_csv(csvfile, c_scale=1):
    print("Loading data from", csvfile)
    
    # Loading input .csv file to pandas
    df = pd.read_csv(csvfile)
    
    # Renaming columns to standard names
    df.columns.values[0] = "compound"
    df.columns.values[1] = "c"
    
    df["c"] = df["c"] * c_scale
    
    # Adding a column log c  
    df.insert(loc=2, column='log c', value=np.log10(df['c']))
    # Adding a column n_replicates 
    df.insert(loc=3, column='n_reps', value=df.count(axis=1, numeric_only=True)-2)
    # Adding median, SD, and SEM columns
    df["median"] = df.iloc[:, 4:].median(numeric_only=True, axis=1)
    df["SD"] = df.iloc[:, 4:-1].std(numeric_only=True, axis=1)   
    df["SEM"] = df.SD/np.sqrt(df.n_reps)
    
    return df


def load_df(df, c_scale=1):
    print("Loading data from", df)
    
    # Renaming columns to standard names
    df.columns.values[0] = "compound"
    df.columns.values[1] = "c"
    
    df["c"] = df["c"] * c_scale
    
    # Adding a column log c  
    df.insert(loc=2, column='log c', value=np.log10(df['c']))
    # Adding a column n_replicates 
    df.insert(loc=3, column='n_reps', value=df.count(axis=1, numeric_only=True)-2)
    # Adding median, SD, and SEM columns
    df["median"] = df.iloc[:, 4:].median(numeric_only=True, axis=1)
    df["SD"] = df.iloc[:, 4:-1].std(numeric_only=True, axis=1)   
    df["SEM"] = df.SD/np.sqrt(df.n_reps)

    return df


def pool_data(df):
      
    # Creating empty list
    pooled_list = []

    # Iterating through df and pooling data into pooled_list
    for index, row in df.iterrows():
        for i in row.iloc[4:-3]:
            pooled_list.append([row.iloc[0], row.iloc[1], row.iloc[2], i])


    # Creating pandas dataframe from pooled_list
    pooled_df = pd.DataFrame(pooled_list, columns=['compound', 'c', 'log c', 'response'])
    
    # Removing rows with NaN 
    pooled_df = pooled_df.dropna(axis=0, how='any')
    
    return pooled_df



def fetch_pars(df, x_curve):
    pars = lmfit.Parameters()   
    
    if df['model'].iloc[0] == "IC50":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('IC50', value = float(df['IC50'].iloc[0]))
        pars.add('slope', value = float(df['slope'].iloc[0]))
        y_curve = models.IC50_lmfit(pars, x_curve)
    
    if df['model'].iloc[0] == "logIC50":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('logIC50', value = float(df['logIC50'].iloc[0]))
        pars.add('slope', value = float(df['slope'].iloc[0]))
        y_curve = models.logIC50_lmfit(pars, x_curve)
        
    if df['model'].iloc[0] == "dir_simple":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        y_curve = models.dir_simple_lmfit(pars, x_curve)
        
    if df['model'].iloc[0] == "dir_specific":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        y_curve = models.dir_specific_lmfit(pars, x_curve)         
        
    if df['model'].iloc[0] == "dir_total":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Ns', value = float(df['Ns'].iloc[0]))
        y_curve = models.dir_total_lmfit(pars, x_curve)         
      
    if df['model'].iloc[0] == "comp_3st_specific":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        y_curve = models.comp_3st_specific_lmfit(pars, x_curve)         
        
    if df['model'].iloc[0] == "comp_3st_total":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        pars.add('N', value = float(df['N'].iloc[0]))
        y_curve = models.comp_3st_total_lmfit(pars, x_curve)               
                     
    if df['model'].iloc[0] == "comp_4st_specific":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        pars.add('Kd3', value = float(df['Kd3'].iloc[0]))
        y_curve = models.comp_4st_specific_lmfit(pars, x_curve)         
        
    if df['model'].iloc[0] == "comp_4st_total":
        pars.add('min', value = float(df['min'].iloc[0]))
        pars.add('max', value = float(df['max'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        pars.add('Kd3', value = float(df['Kd3'].iloc[0]))
        pars.add('N', value = float(df['N'].iloc[0]))
        y_curve = models.comp_4st_total_lmfit(pars, x_curve)              
                       
    return y_curve


def plot(data_df, results_df, compound_sel=False, xmin=False, xmax=False, 
         marker="o", 
         markersize=5, 
         linewidth=1, 
         linestyle="-",
         show_medians=True,
         show_all_data=False, 
         show_errorbars=True, 
         errorbars_kind="SD", 
         errorbar_linewidth = 1, 
         errorbar_capsize=3, 
         cmap="brg_r", 
         cmap_min = 0, 
         cmap_max = 1, 
         custom_colors=False, 
         single_color=False, 
         custom_labels=False,
         single_label=False,
         no_labels=False):
    

    # In compound selection is provided, than use it, otherwise plot all compounds
    if compound_sel == False:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel

    # Setting up colors
    # By default, colors are set up as a cmap
    # cmap options: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colors = plt.colormaps[cmap](np.linspace(cmap_min, cmap_max, len(compounds)))
    if custom_colors:
        colors=custom_colors
    if single_color:
        colors=custom_colors
        colors = [single_color for _ in range(len(compounds))]
        
    # Setting up labels    
    labels = compounds
    if custom_labels:
        labels=custom_labels
    if single_label:
        labels = [single_label for _ in range(len(compounds))]
    if no_labels:
        labels = [None for _ in range(len(compounds))]

    
    # Iterate through compounds and plot them in matplotlib    
    for i, compound in enumerate(compounds):
        

        
        # This is a selection from the dataframe with the experimental data
        sel_data = data_df.loc[data_df['compound'] == compound]
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        

        if sel_results['model'].iloc[0] == "logIC50":
            conc = "log c"
        else:
            conc = "c"
        

        # Turning on/off plotting of errorbars, all data, or medians (defaults is medians with errorbars)
        if show_errorbars:
            plt.errorbar(sel_data[conc], sel_data["median"], yerr=sel_data[errorbars_kind], elinewidth=errorbar_linewidth, capthick=errorbar_linewidth, capsize=errorbar_capsize, linestyle="", marker="none", markersize=markersize, color=colors[i])
        if show_medians:
            plt.plot(sel_data[conc], sel_data["median"], marker=marker, markersize=markersize, linestyle="", color=colors[i])        
        if show_all_data:
            sel_data_pooled = pool_data(sel_data)
            plt.plot(sel_data_pooled[conc], sel_data_pooled["response"], marker=marker, markersize=markersize, linestyle="", color=colors[i])


        # Setting min and max on x axis for the curves
        if xmin:
            min_curve=xmin
        else:
            min_curve = min(sel_data[conc])
        if xmax:
            max_curve=xmax
        else:
            max_curve = max(sel_data[conc]) 

        # Setting up the x values for the curve
        x_curve = np.logspace(np.log10(min_curve), np.log10(max_curve), 1000)
        #x_curve = np.linspace(min_curve, max_curve, int((max_curve-min_curve)*100))

        # Fetching parameters for a given model and getting y values
        y_curve = fetch_pars(sel_results, x_curve)

        # Plotting the curve
        plt.plot(x_curve, y_curve, color=colors[i], linestyle=linestyle, linewidth=linewidth)
        
        # Hidden plots just to make labels for the legend
        if single_label == False:
            if show_medians==True and show_all_data==False:
                plt.plot(sel_data[conc].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])
            if show_all_data==True:
                plt.plot(sel_data_pooled[conc].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])        
            if show_medians==False and show_all_data==False:
                plt.plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=labels[i])
                
    if single_label != False:
        if show_medians==True and show_all_data==False:
            plt.plot(sel_data[conc].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=single_label)
        if show_all_data==True:
            plt.plot(sel_data_pooled[conc].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=single_label)        
        if show_medians==False and show_all_data==False:
            plt.plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=single_label)

            






def plot_old(data_df, results_df, compound_sel=False, xmin=False, xmax=False, 
         marker="o", 
         markersize=5, 
         linewidth=1, 
         linestyle="-",
         show_medians=True,
         show_all_data=False, 
         show_errorbars=True, 
         errorbars_kind="SD", 
         errorbar_linewidth = 1, 
         errorbar_capsize=3, 
         cmap="brg_r", 
         cmap_min = 0, 
         cmap_max = 1, 
         custom_colors=False, 
         single_color=False, 
         custom_labels=False,
         single_label=False,
         no_labels=False):
    

    # In compound selection is provided, than use it, otherwise plot all compounds
    if compound_sel == False:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel

    # Setting up colors
    # By default, colors are set up as a cmap
    # cmap options: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colors = plt.colormaps[cmap](np.linspace(cmap_min, cmap_max, len(compounds)))
    if custom_colors:
        colors=custom_colors
    if single_color:
        colors=custom_colors
        colors = [single_color for _ in range(len(compounds))]
        
    # Setting up labels    
    labels = compounds
    if custom_labels:
        labels=custom_labels
    if single_label:
        labels = [single_label for _ in range(len(compounds))]
    if no_labels:
        labels = [None for _ in range(len(compounds))]

    
    # Iterate through compounds and plot them in matplotlib    
    for i, compound in enumerate(compounds):
        

        
        # This is a selection from the dataframe with the experimental data
        sel_data = data_df.loc[data_df['compound'] == compound]
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        

        # Turning on/off plotting of errorbars, all data, or medians (defaults is medians with errorbars)
        if show_errorbars:
            plt.errorbar(sel_data["c"], sel_data["median"], yerr=sel_data[errorbars_kind], elinewidth=errorbar_linewidth, capthick=errorbar_linewidth, capsize=errorbar_capsize, linestyle="", marker="none", markersize=markersize, color=colors[i])
        if show_medians:
            plt.plot(sel_data["c"], sel_data["median"], marker=marker, markersize=markersize, linestyle="", color=colors[i])        
        if show_all_data:
            sel_data_pooled = pool_data(sel_data)
            plt.plot(sel_data_pooled["c"], sel_data_pooled["response"], marker=marker, markersize=markersize, linestyle="", color=colors[i])


        # Setting min and max on x axis for the curves
        if xmin:
            min_curve=xmin
        else:
            min_curve = min(sel_data["c"])
        if xmax:
            max_curve=xmax
        else:
            max_curve = max(sel_data["c"]) 

        # Setting up the x values for the curve
        x_curve = np.logspace(np.log10(min_curve), np.log10(max_curve), 1000)
        #x_curve = np.linspace(min_curve, max_curve, int((max_curve-min_curve)*100))

        # Fetching parameters for a given model and getting y values
        y_curve = fetch_pars(sel_results, x_curve)

        # Plotting the curve
        plt.plot(x_curve, y_curve, color=colors[i], linestyle=linestyle, linewidth=linewidth)
        
        # Hidden plots just to make labels for the legend
        if single_label == False:
            if show_medians==True and show_all_data==False:
                plt.plot(sel_data['c'].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])
            if show_all_data==True:
                plt.plot(sel_data_pooled['c'].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])        
            if show_medians==False and show_all_data==False:
                plt.plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=labels[i])
                
    if single_label != False:
        if show_medians==True and show_all_data==False:
            plt.plot(sel_data['c'].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=single_label)
        if show_all_data==True:
            plt.plot(sel_data_pooled['c'].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=single_label)        
        if show_medians==False and show_all_data==False:
            plt.plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=single_label)
     
      

      
def plot_grid(data_df, results_df, compound_sel=False, xmin=False, xmax=False, 
         marker="o", 
         markersize=5, 
         linewidth=1, 
         linestyle="-",
         show_medians=True,
         show_all_data=False, 
         show_errorbars=True, 
         errorbars_kind="SD", 
         errorbar_linewidth = 1, 
         errorbar_capsize=3, 
         cmap="brg_r", 
         cmap_min = 0, 
         cmap_max = 1, 
         custom_colors=False, 
         single_color=False, 
         custom_labels=False,
         single_label=False,
         no_labels=False,
         x_logscale=True,
         show_legend=False,
         show_title=True,
         figsize=(7, 5),
         n_cols=3,
         x_label="dose",
         y_label="response",         
         show_inner_ticklabels=False,
         sharex=True,
         sharey=True,
         hspace=0.3,
         wspace=0.3):
    

    # In compound selection is provided, than use it, otherwise plot all compounds
    if compound_sel == False:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel

    # Setting up colors
    # By default, colors are set up as a cmap
    # cmap options: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colors = plt.colormaps[cmap](np.linspace(cmap_min, cmap_max, len(compounds)))
    if custom_colors:
        colors=custom_colors
    if single_color:
        colors=custom_colors
        colors = [single_color for _ in range(len(compounds))]
        
    # Setting up labels    
    labels = compounds
    if custom_labels:
        labels=custom_labels
    if single_label:
        labels = [single_label for _ in range(len(compounds))]
    if no_labels:
        labels = [None for _ in range(len(compounds))]

    
    #fig, axes = plt.subplots(2, 3, figsize=(7, 5))  # 2 rows, 2 columns
    
    # Figure out no of rows
    n_rows = int(len(compounds) / n_cols) + (len(compounds) % n_cols > 0)       # Just a fancy way of rounding up
    
    # Create a grid of subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=hspace, wspace=wspace)
    axes = gs.subplots(sharex=sharex, sharey=sharey)

    
    # Flatten the 2D axes array to iterate over it
    axes = axes.flatten()

    # Remove subplots if they would be empty
    n_remove = n_cols*n_rows - len(compounds)       # No of elements to temove

    for i in range(1, n_remove+1):
        fig.delaxes(axes[-i])  # Remove axes from the figure


    
    # Iterate through compounds and plot them in matplotlib    
    for i, compound in enumerate(compounds):
        
        # This is a selection from the dataframe with the experimental data
        sel_data = data_df.loc[data_df['compound'] == compound]
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]

        # Turning on/off plotting of errorbars, all data, or medians (defaults is medians with errorbars)
        if show_errorbars:
            axes[i].errorbar(sel_data["c"], sel_data["median"], yerr=sel_data[errorbars_kind], elinewidth=errorbar_linewidth, capthick=errorbar_linewidth, capsize=errorbar_capsize, linestyle="", marker="none", markersize=markersize, color=colors[i])
        if show_medians:
            axes[i].plot(sel_data["c"], sel_data["median"], marker=marker, markersize=markersize, linestyle="", color=colors[i])        
        if show_all_data:
            sel_data_pooled = pool_data(sel_data)
            axes[i].plot(sel_data_pooled["c"], sel_data_pooled["response"], marker=marker, markersize=markersize, linestyle="", color=colors[i])


        # Setting min and max on x axis for the curves
        if xmin:
            min_curve=xmin
        else:
            min_curve = min(sel_data["c"])
        if xmax:
            max_curve=xmax
        else:
            max_curve = max(sel_data["c"]) 

        # Setting up the x values for the curve
        x_curve = np.logspace(np.log10(min_curve), np.log10(max_curve), 1000)
        #x_curve = np.linspace(min_curve, max_curve, int((max_curve-min_curve)*100))

        # Fetching parameters for a given model and getting y values
        y_curve = fetch_pars(sel_results, x_curve)


        # Plotting the curve      
        axes[i].plot(x_curve, y_curve, color=colors[i], linestyle=linestyle, linewidth=linewidth)
        
        
        # Setting log scale for x axis
        if x_logscale==True:
            axes[i].set_xscale('log', base=10)
        
        # Show name in the title of subplots
        if show_title==True:
            axes[i].set_title(compound, fontsize=10)


        # Hidden plots just to make labels for the legend
        if show_medians==True and show_all_data==False:
            axes[i].plot(sel_data['c'].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])
        if show_all_data==True:
            axes[i].plot(sel_data_pooled['c'].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])        
        if show_medians==False and show_all_data==False:
            axes[i].plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=labels[i])
            
        # Setting up legend        
        if show_legend==True:        
            axes[i].legend()
                

    # Setting up axis labels 
    for ax in axes:
        ax.set(xlabel=x_label, ylabel=y_label)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes:
        ax.label_outer()
        if show_inner_ticklabels == True:
            ax.xaxis.set_tick_params(labelbottom=True)      # Put back x ticklabels 
            ax.yaxis.set_tick_params(labelbottom=True)      # Put back y ticklabels 
    
    
    
    plt.tight_layout()
    plt.show()






   
def plot_asymptotes(results_df, compound_sel=False, lower=True, upper=True, color="black", linewidth=1, linestyle="--"):

    # If compound selection is provided, than use it, otherwise plot all compounds
    if compound_sel == False:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel
    
    # Iterate through compounds and plot them in matplotlib    
    for compound in compounds:
             
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        if lower:
            plt.axhline(y = float(sel_results['min'].iloc[0]), color=color, linestyle=linestyle, linewidth=linewidth) 
        if upper:
            plt.axhline(y = float(sel_results['max'].iloc[0]), color=color, linestyle=linestyle, linewidth=linewidth)   




def plot_traces(results_df, value, compound_sel=False, kind="full", vtrace=True, htrace=True, color="black", linewidth=1, linestyle="--", label=None):

    # If compound selection is provided, than use it, otherwise plot all compounds
    if compound_sel == False:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel
        

    # Iterate through compounds and plot them in matplotlib    
    for compound in compounds:
        
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        x = float(sel_results[value].iloc[0])
        print("Plotting trace for x value", x)
        
        # Fetching parameters for a given model and getting y values
        y = fetch_pars(sel_results, x)

        
        # Plotting the traces
        if kind == "full":
            if vtrace:
                plt.axvline(x = x, color=color, linestyle=linestyle, linewidth=linewidth) 
            if htrace:
                plt.axhline(y = y, color=color, linestyle=linestyle, linewidth=linewidth)  
        
        if kind == "partial":
            if vtrace:
                plt.vlines(x, ymin=0, ymax=y, color=color, linestyle=linestyle, linewidth=linewidth)
            if htrace:
                plt.hlines(y, xmin=0, xmax=x, color=color, linestyle=linestyle, linewidth=linewidth)

         
        # Hidden plots just to make labels for the legend
        plt.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)





def plot_value(results_df, value, compound_sel=False, marker="o", markersize=5, color="black", label=None, show_annot=True, pre_text="", post_text="", decimals=2, xoffset=50, yoffset=0):
        
    # If compound selection is provided, than use it, otherwise plot all compounds
    if compound_sel == False:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel
        

    # Iterate through compounds and plot them in matplotlib    
    for compound in compounds:
        
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        x = float(sel_results[value].iloc[0])
        print("Plotting marker for x value", x)
        
        # Fetching parameters for a given model and getting y values
        y = fetch_pars(sel_results, x)

        
        # Plotting the marker
        plt.plot(x, y, marker=marker, markersize=markersize, color=color) 

         
        # Hidden plots just to make labels for the legend
        plt.plot(x, y, marker=marker, markersize=markersize, color=color, label=label, linestyle="")
        
        # Show annotation
        #plt.text(x, y, f"{pre_text}{x:.{decimals}f}{post_text}")
        if show_annot==True:
            plt.annotate(f"{pre_text}{x:.{decimals}f}{post_text}", (x, y), xytext=(x+xoffset, y+yoffset), horizontalalignment='center', verticalalignment='center')
        
        




def report(df, decimals=2, p=False):
    
    compounds = df["compound"].unique()

    # Initiating empty output_df
    output_df = pd.DataFrame(columns=['compound', 'Mean (95% CI)', 'Mean \u00B1 SE'])

    
    for compound in compounds:
        
        df_compound = df[df["compound"].isin([compound])]
        
        
        value_compound = df_compound.iloc[0, 2]
        loCL_compound = df_compound.iloc[0, 3]
        upCL_compound = df_compound.iloc[0, 4]
        SE_compound = df_compound.iloc[0, 5]
        
     
                
        # Creating new row for the output dataframe  
        if loCL_compound == "nd" and SE_compound != "nd":
            new_row = [compound, 
                       f"{value_compound:.{decimals}f} ({loCL_compound}, {upCL_compound})", 
                       f"{value_compound:.{decimals}f} \u00B1 {SE_compound:.{decimals}f}"]
        if SE_compound == "nd" and loCL_compound != "nd":
            new_row = [compound, 
                       f"{value_compound:.{decimals}f} ({loCL_compound:.{decimals}f}, {upCL_compound:.{decimals}f})", 
                       f"{value_compound:.{decimals}f} \u00B1 {SE_compound:}"]
        if loCL_compound == "nd" and SE_compound == "nd":
            new_row = [compound, 
                       f"{value_compound:.{decimals}f} ({loCL_compound}, {upCL_compound})", 
                       f"{value_compound:.{decimals}f} \u00B1 {SE_compound}"]            
        if loCL_compound != "nd" and SE_compound != "nd":
            new_row = [compound, 
                        f"{value_compound:.{decimals}f} ({loCL_compound:.{decimals}f}, {upCL_compound:.{decimals}f})", 
                        f"{value_compound:.{decimals}f} \u00B1 {SE_compound:.{decimals}f}"]
            

        # Adding new row to the output dataframe
        output_df.loc[len(output_df)] = new_row
    
    return output_df



    
    