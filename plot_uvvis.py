from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg
import os

def plot_uvvis_data(df: pd.DataFrame, name:str, skip:int = 1):
    """
    First column of dataframe should be Wavelength
    The rest of the columns will be the absorbance value at timestamp
    Plots Absorbance vs. Wavelength, where time is represented 
    through the color of the plot. Time starts from dark line to light purple
    Saves the plot as png
    @param df: dataframe of UV-Vis data
    """

    #########################
    # RAW DATA PLOT
    #########################
    x = df["Wavelength"]
    end_time = int(df.columns[-1])
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(hspace=.5)
    plt.subplot(3, 1, 1)
    for i in range(1, len(df.columns[1:]), skip):
        col = df.columns[i]
        c = int(col)/end_time
        plt.plot(x,df[col], color=(c*.5 + .5, c*.7, c), linewidth=1)
    plt.xlabel("Wavelength")
    plt.ylabel("Absorbance")
    start = name.find("GRAPH_")
    plt.title("Raw UV-VIS Data")

    ##############################
    #   SPECTRAL DECONVOLUTION
    ##############################
    max_iterations = 100
    mcr = McrAR(
    c_regr='OLS',  # Ordinary least squares for concentrations
    st_regr='NNLS',  # Non-negative least squares for spectral components
    c_constraints=[ConstraintNonneg()],
    max_iter=max_iterations
    )
    diff = df.iloc[:,1]-df.iloc[:,-1]
    diff_fixed = np.maximum(diff.to_numpy(), 0)
    init = df.iloc[:,1].to_numpy()
    initial_guess = np.stack([init, diff_fixed], axis=0)
    # print(initial_guess.shape)
    # print(df.iloc[:,1:].to_numpy().T.shape)

    # mcr on all data
    mcr.fit(df.iloc[:,1:].to_numpy().T, ST=initial_guess)

    # Extract results
    spectral_components = mcr.ST_.T  # Final spectral components
    concentrations = mcr.C_  # Contributions of components over time
    reconstructed_data = mcr.D_  # Reconstructed data
    extracted_signal = mcr.ST_opt_

    # print("SPECTRAL", spectral_components.shape)
    # print(spectral_components)
    # print("CONCENTRATion", concentrations.shape)
    # print(concentrations)
    # print("RECONSTRUCTED DATA", reconstructed_data.shape)
    # print(reconstructed_data)
    # print("RESULT", result.shape)
    # print(result)
    # print("C_OPT", mcr.C_opt_.shape)
    # print(mcr.C_opt_)
    # print("D_OPT", mcr.D_opt_.shape)
    # print(mcr.D_opt_)
    # print("DIFF")
    # print(diff)
    # exit()

    # Plot Spectral Deconvolution Results
    plt.subplot(3, 1, 2)
    plt.plot(df["Wavelength"], extracted_signal[0], 'r--', label='Pristine Component (Concentration)')
    plt.plot(df["Wavelength"], extracted_signal[1], 'b--', label='Degraded Component (Concentration)')
    plt.legend()
    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('Spectral Deconvolution of UV-Vis Data')

    ##############################
    # DEGRADATION RATE REGRESSION
    ##############################

    # fitting regression
    y_axis = mcr.C_opt_[:,0]
    negative_mask = np.where(y_axis < 0, False, True)
    x_axis = df.columns[1:].astype(int)[negative_mask]
    y_axis = y_axis[negative_mask]
    min_fitting_points = 5
    slope_uncertainty = 1e12

    result = {"fit_limit": 0,
                  "rmse_transformed": 1e12,
                  "intercept": 0,
                  "slope": "Insufficient points for fitting",
                  "x_data": [],
                  "y_data": [],
                  "slope uncertainty": 1e12,
                  "MCR_iter": f"{mcr.n_iter}/{max_iterations}",
                  }
    
    for i in range(min_fitting_points, len(x_axis)):
        # Extract subset of data to fit
        x_data_raw = x_axis[1:i]  # [t, ]
        y_data_raw = y_axis[1:i]  # [t, ]
        # Normalize and take ln
        lny_data_raw = np.log(y_data_raw / y_axis[0])
        # Expunge NaN values from x and y
        nan_mask = np.where(np.isnan(lny_data_raw), False, True)
        x_data_clipped = x_data_raw[nan_mask]
        y_data_clipped = lny_data_raw[nan_mask]

        fitting_x_data = x_data_clipped
        fitting_y_data = y_data_clipped

        if fitting_x_data.size == 0:
            continue

        fit_info = np.polyfit(fitting_x_data, fitting_y_data, deg=1, full=True)
        beta = fit_info[0][0]
        alpha = fit_info[0][1]
        n = len(fitting_x_data)
        rmse = np.sqrt(fit_info[1][0] / n)
        r_var_time = np.sqrt(np.nanvar(fitting_x_data))

        if len(fitting_x_data) > 2:
            sigma_m = ((rmse / r_var_time) / np.sqrt(n - 2))
            # sigma = sqrt( 1/(n-2) * Sum(e^2) // Sum((x - <x>)^2) )
            #       = sqrt( n/(n-2) * (1/n) * Sum(e^2) // Sum((x - <x>)^2) )
            #       = [RMSE] * sqrt( n/(n-2) // Sum((x - <x>)^2) )
            #       = [RMSE] * sqrt( 1/(n-2) // (Sum((x - <x>)^2)/n) )
            #       = [RMSE] * sqrt( 1/(n-2) // Var(x))
            #       = ([RMSE]/sqrt(Var(x))) * sqrt(1/(n-2))
        else:
            sigma_m = 1e13  # must be larger than 1e12 (default value) or the "dry" runs will overwrite defaults

        if sigma_m <= result['slope uncertainty']:
            result.update({"fit_limit": i,
                            "rmse_transformed": rmse,
                            "intercept": alpha,
                            "slope": beta,
                            "x_data": x_axis,
                            "y_data": y_axis,
                            "slope uncertainty": sigma_m})
    
    # graph the degradation rate plot if there is one
    if not isinstance(result["slope"], str):
        x = fitting_x_data[:result['fit_limit']]
        y = fitting_y_data[:result['fit_limit']]
        x_min = x.min()
        x_max = x.max()
        graph_x = np.linspace(x_min, x_max, 100)
        graph_y = graph_x*result["slope"] + result["intercept"]

        plt.subplot(3, 1, 3)

        plt.scatter(x, y, c="blue", s=10)
        plt.plot(graph_x, graph_y, 'black')
        text_str = f"Slope: {result['slope']:.2e}\nIntercept: {result['intercept']:.2e}\nRMSE: {result['rmse_transformed']:.2e}\nSlope Uncertainty: {result['slope uncertainty']:.2e}"
        plt.text(.68, .95, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        plt.xlabel('Exposure Time')
        plt.ylabel('Ln (Signal)')
        plt.title('Degradation Rate')
    else:
        plt.subplot(3, 1, 3)
        plt.text(.5,.5, "Unable to calculate degradation rate", transform=plt.gca().transAxes, fontsize=20, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    # plt.show()
    plt.savefig(name)

if __name__ == "__main__":

    directory = "ZZData/uvvis/"
    for filename in os.listdir(directory):
        if ".csv" in filename:
            df = pd.read_csv(directory+filename, comment="#").iloc[:,1:]
            o = df.iloc[:,-1]-df.iloc[:,1]
            plot_uvvis_data(df, directory + "GRAPH_" + filename[:-4], 40)
            # exit()



"""

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y, color=(1,1,1), linewidth=2)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')

# Show the plot
plt.show()

# """