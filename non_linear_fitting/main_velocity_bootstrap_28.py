import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import glob
import re
import numpy as np
from scipy.optimize import minimize
pen_vel = 1
pen_dens = 1
pen_int  = 1
temp = "normal"
cut_length   = 5
cut_width_length = 10
# Correct way to insert variable into a string using f-strings
target_file = f"experiment_files/{temp}_temp3.txt"
target_file_v = f"experiment_files/{temp}_velocity_data_all_points.txt"

def optimize_scaling_factor(data, data_density,data_v,target_data,target_data_density,target_data_v):
    # Objective function to minimize the MSE by scaling the x-axis
    def objective(scale_factor):
        scaled_x         = data['x'] * scale_factor
        scaled_x_density = data_density['x'] * scale_factor
        scaled_x_v       = data_v['x']*scale_factor

        scaled_v1  = data_v['v1']*cut_length/scale_factor
        scaled_v2  = data_v['v2']*cut_length/scale_factor
        scaled_v3  = data_v['v3']*cut_length/scale_factor
        scaled_v4  = data_v['v4']*cut_length/scale_factor
        scaled_v5  = data_v['v5']*cut_length/scale_factor
        y_interpolated         = np.interp(target_data['x'],scaled_x,  data['y']*cut_width_length)
        y_density_interpolated = np.interp(target_data_density['x'],scaled_x_density, data_density['y'])
        y_v1_interpolated      = np.interp(target_data_v['x'],scaled_x_v, scaled_v1)
        y_v2_interpolated      = np.interp(target_data_v['x'],scaled_x_v, scaled_v2)
        y_v3_interpolated      = np.interp(target_data_v['x'],scaled_x_v, scaled_v3)
        y_v4_interpolated      = np.interp(target_data_v['x'],scaled_x_v, scaled_v4)
        y_v5_interpolated      = np.interp(target_data_v['x'],scaled_x_v, scaled_v5)
        size_ =  1*(y_density_interpolated - target_data_density['y']).size /(5*((y_v2_interpolated - target_data_v['v2']).size))

        # print((data_v['v1']).size)171 4.0 0.5 1.0 7.89268333903069

        mse = np.mean( 
            pen_int * (y_interpolated - target_data['y']) ** 2 +
            pen_dens *(y_density_interpolated - target_data_density['y']) ** 2
        )   +  np.mean(
            size_*pen_vel * (y_v1_interpolated - target_data_v['v1']) ** 2 +
            size_*pen_vel * (y_v2_interpolated - target_data_v['v2']) ** 2 + 
            size_*pen_vel * (y_v3_interpolated - target_data_v['v3']) ** 2 +
            size_*pen_vel * (y_v4_interpolated - target_data_v['v4']) ** 2 +
            size_*pen_vel * (y_v5_interpolated - target_data_v['v5']) ** 2  
        )


        return mse
    
    # Use scipy's minimize function to find the optimal scaling factor
    result = minimize(objective, x0=1.0, bounds=[(0.001, np.inf)],method='L-BFGS-B')

    y_interpolated = np.interp(target_data['x'],data['x'] * result.x,  data['y']*cut_length)
    y_density_interpolated = np.interp(target_data['x'],data_density['x'] * result.x,  data_density['y'])
    y_v1_interpolated = np.interp(target_data_v['x'],data_v['x']* result.x ,  data_v['v1']*cut_length/ result.x)
    y_v2_interpolated = np.interp(target_data_v['x'],data_v['x']* result.x,  data_v['v2']*cut_length/ result.x)   
    y_v3_interpolated = np.interp(target_data_v['x'],data_v['x']* result.x ,  data_v['v3']*cut_length/ result.x)  
    y_v4_interpolated = np.interp(target_data_v['x'],data_v['x']* result.x,  data_v['v4']*cut_length/ result.x)
    y_v5_interpolated = np.interp(target_data_v['x'],data_v['x']* result.x,  data_v['v5']*cut_length/ result.x)
    
    residuals = target_data['y']-y_interpolated
    residuals_density = target_data_density['y']-y_density_interpolated
    residual_v1    = target_data_v['v1']-y_v1_interpolated
    residual_v2    = target_data_v['v2']-y_v2_interpolated
    residual_v3    = target_data_v['v3']-y_v3_interpolated  
    residual_v4    = target_data_v['v4']-y_v4_interpolated  
    residual_v5    = target_data_v['v5']-y_v5_interpolated
    return result.x, result.fun, residuals, residuals_density,residual_v1,residual_v2,residual_v3,residual_v4,residual_v5, y_interpolated, y_density_interpolated,y_v1_interpolated, y_v2_interpolated,y_v3_interpolated,y_v4_interpolated,y_v5_interpolated

def find_optimum(start, end, target_data, target_data_density,target_data_v):
    min_mse = float('inf')
    second_min_mse = float('inf')
    N = 0
    for i in range(start, end + 1):
        file_pattern = f"simulation_files_180/output_test__{i}_param1_*.txt"
        file_pattern_density = f"simulation_files_180/output_rho_test__{i}_param1_*.txt"
        file_pattern_v   =  f"simulation_files_180/output_hydro_test__{i}_param1_*.txt"

       
        matching_files = glob.glob(file_pattern)
        matching_files_density = glob.glob(file_pattern_density)
        matching_files_v = glob.glob(file_pattern_v)

        if matching_files and matching_files_density and matching_files_v :
            txt_file = matching_files[0]  # Take the first match (assuming only one file matches)
            txt_file_density = matching_files_density[0]  # Take the first match (assuming only one file matches)
            txt_file_v = matching_files_v[0]  # Take the first match (assuming only one file matches)   
            # Check if the file is empty
            if os.path.getsize(txt_file) == 0:
                print(f"Text file {txt_file} is empty, skipping...")
                continue
    
            # Extract param1, param2, and param3 from the filename using regex
            match     = re.search(r'param1_(\d+\.\d+)_param2_(\d+\.\d+)_param3_(\d+\.\d+)', txt_file)
            match_num = i
            if match:
                param1 = float(match.group(1))
                param2 = float(match.group(2))
                param3 = float(match.group(3))
                # print(f"Test {i}: param1 = {param1}, param2 = {param2}, param3 = {param3}")

            # Load the data from the text file
            data = pd.read_csv(txt_file, header=None, names=['x', 'y'], sep=',')
            data_density  = pd.read_csv(txt_file_density, header=None, names=['x', 'y'], sep=',')
            column_names = ['x'] + [f'v{i}' for i in range(1, 101)] 
            data_v = pd.read_csv(txt_file_v, header=None, names=column_names, sep=',')

            last_value = data_density.iloc[-1, 1]
            if (last_value < 0.9 or last_value > 1.1):
               N += 1
               #print("Fail ",i," ",param1," ",param3," ",param2, " ",N)
               continue        


            # Find the optimal scaling factor to minimize the error
            optimal_scaling_factor, mse, residuals, residuals_density, residuals_v1, residuals_v2, \
            residuals_v3, residuals_v4, residuals_v5, y_interpolated, y_density_interpolated, \
            y_v1_interpolated, y_v2_interpolated, y_v3_interpolated, y_v4_interpolated, y_v5_interpolated = \
            optimize_scaling_factor(
                data, 
                data_density, 
                data_v, 
                target_data, 
                target_data_density, 
                target_data_v
            )

            
            # Check if this is the minimum MSE so far
            if  mse < min_mse:
                min_mse = mse
                min_error_scaling_factor = optimal_scaling_factor
                fit_param1 = param1
                fit_param2 = param2
                fit_param3 = param3
                fit_residuals = np.copy(residuals)
                fit_residuals_density = np.copy(residuals_density)
                fit_residuals_v1  = np.copy(residuals_v1)
                fit_residuals_v2  = np.copy(residuals_v2)
                fit_residuals_v3  = np.copy(residuals_v3)
                fit_residuals_v4  = np.copy(residuals_v4)
                fit_residuals_v5  = np.copy(residuals_v5)                               
                fit_y_interpolated = np.copy(y_interpolated)
                fit_y_density_interpolated = np.copy(y_density_interpolated)
                fit_y_v1_interpolated  = np.copy(y_v1_interpolated)
                fit_y_v2_interpolated  = np.copy(y_v2_interpolated)
                fit_y_v3_interpolated  = np.copy(y_v3_interpolated)
                fit_y_v4_interpolated  = np.copy(y_v4_interpolated)
                fit_y_v5_interpolated  = np.copy(y_v5_interpolated)   
                fit_match_num    = i

        #print(i, " ",mse," ",param1," ",param3," ",param2)                            
        # else:
        #     print(f"Text file matching pattern {file_pattern} not found, skipping...")

    # print(N,N/(end-start))
    return fit_match_num,fit_param1, fit_param2, fit_param3, min_error_scaling_factor, \
           fit_residuals, fit_residuals_density, \
           fit_residuals_v1,fit_residuals_v2,fit_residuals_v3,fit_residuals_v4,fit_residuals_v5, \
           fit_y_interpolated, fit_y_density_interpolated,\
           fit_y_v1_interpolated,fit_y_v2_interpolated,fit_y_v3_interpolated,fit_y_v4_interpolated,fit_y_v5_interpolated


if __name__ == "__main__":
    start, end = 1,3000
    n_bootstrap = 100


    data = pd.read_csv(target_file, header=None, sep=',')
    data_v = pd.read_csv(target_file_v, header=None, sep=',')

    target_data = data[[0, 1]]  # Selecting first and second columns
    target_data.columns = ['x', 'y']  # Renaming columns to 'x' and 'y'
    #target_data.iloc[:,1] /= 10.0  # Rescale only the target y-axis by dividing by 10

    target_data_density = data[[0, 2]]  # Selecting first and third columns
    target_data_density.columns = ['x', 'y']  # Renaming columns to 'x' and 'y'

    target_data_v = data_v.iloc[:, 0:6]  # Select first 11 columns
    target_data_v.columns = ['x', 'v1', 'v2', 'v3', 'v4', 'v5']  # Rename columns




    fit_match_num, fit_param1, fit_param2, fit_param3, min_error_scaling_factor, fit_residuals, fit_residuals_density, \
    fit_residuals_v1,fit_residuals_v2,fit_residuals_v3,fit_residuals_v4,fit_residuals_v5, fit_y_interpolated, fit_y_density_interpolated, \
    fit_y_v1_interpolated,fit_y_v2_interpolated,fit_y_v3_interpolated,fit_y_v4_interpolated,fit_y_v5_interpolated= find_optimum(start, end, target_data,target_data_density,target_data_v)



# Create subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    # Corrected plotting function for density data

    axes[0, 0].scatter(target_data_density['x'], target_data_density['y'], label='Original Density Data', marker='o')
    axes[0, 0].plot(target_data_density['x'], fit_y_density_interpolated, label='Fitted Density Data', marker='o',color='black')
    axes[0, 0].set_ylabel('density')
    axes[0, 0].set_xlabel('time')
    axes[0, 0].grid(True)


    # Corrected plotting function for temperature data

    axes[0, 1].scatter(target_data_v['x'], target_data_v['v1'], label='Original Temperature Data', marker='o')
    axes[0, 1].plot(target_data_v['x'], fit_y_v1_interpolated, label='Fitted Temperature Data', marker='o',color='black')
    axes[0, 1].set_ylabel('v1')
    axes[0, 1].set_xlabel('time')
    axes[0, 1].grid(True)
    axes[0,1].set_ylim(-0.1,2)
    # Corrected plotting function for temperature data

    axes[0, 2].scatter(target_data_v['x'], target_data_v['v2'], label='Original Temperature Data', marker='o')
    axes[0, 2].plot(target_data_v['x'], fit_y_v2_interpolated, label='Fitted Temperature Data', marker='o',color='black')
    axes[0, 2].set_ylabel('v2')
    axes[0, 2].set_xlabel('time')
    axes[0, 2].grid(True)
    axes[0,2].set_ylim(-0.1,2)
    # Corrected plotting function for temperature data

    axes[1, 0].scatter(target_data_v['x'], target_data_v['v3'], label='Original Temperature Data', marker='o')
    axes[1, 0].plot(target_data_v['x'], fit_y_v3_interpolated, label='Fitted Temperature Data', marker='o',color='black')
    axes[1, 0].set_ylabel('v3')
    axes[1, 0].set_xlabel('time')
    axes[1, 0].grid(True)
    axes[1,0].set_ylim(-0.1,2)
    # Corrected plotting function for temperature data

    axes[1, 1].scatter(target_data_v['x'], target_data_v['v4'], label='Original Temperature Data', marker='o')
    axes[1, 1].plot(target_data_v['x'], fit_y_v4_interpolated, label='Fitted Temperature Data', marker='o',color='black')
    axes[1, 1].set_ylabel('v4')
    axes[1, 1].set_xlabel('time')
    axes[1, 1].grid(True)
    axes[1,1].set_ylim(-0.1,2)
    # Corrected plotting function for temperature data

    axes[1, 2].scatter(target_data_v['x'], target_data_v['v5'], label='Original Temperature Data', marker='o')
    axes[1, 2].plot(target_data_v['x'], fit_y_v5_interpolated, label='Fitted Temperature Data', marker='o',color='black')
    axes[1, 2].set_ylabel('v5')
    axes[1, 2].set_xlabel('time')
    axes[1, 2].grid(True)
    axes[1,2].set_ylim(-0.1,2)
    plt.show()

    list_param1 = [fit_param1]
    list_param2 = [fit_param2]
    list_param3 = [fit_param3]
    list_tau = [min_error_scaling_factor]
    print(fit_match_num,fit_param1,fit_param3,fit_param2,min_error_scaling_factor[0])

    for n in np.arange(n_bootstrap):

        print("Bootstrap number ", n)

        target_data.iloc[:,1] = fit_y_interpolated + 1*np.random.choice(fit_residuals)        
        target_data_density.iloc[:,1] = fit_y_density_interpolated + 1*np.random.choice(fit_residuals_density)
        target_data_v.iloc[:,1] = fit_y_v1_interpolated + 1.0*np.random.choice(fit_residuals_v1)
        target_data_v.iloc[:,2] = fit_y_v2_interpolated + 1.0*np.random.choice(fit_residuals_v2)
        target_data_v.iloc[:,3] = fit_y_v3_interpolated + 1.0*np.random.choice(fit_residuals_v3)  
        target_data_v.iloc[:,4] = fit_y_v4_interpolated + 1.0*np.random.choice(fit_residuals_v4)    
        target_data_v.iloc[:,5] = fit_y_v5_interpolated + 1.0*np.random.choice(fit_residuals_v5)    

        fit_match_num, fit_param1, fit_param2, fit_param3, min_error_scaling_factor, fit_residuals_, fit_residuals_density_, \
        fit_residuals_v1_,fit_residuals_v2_,fit_residuals_v3_,fit_residuals_v4_,fit_residuals_v5_, fit_y_interpolated_, fit_y_density_interpolated_, \
        fit_y_v1_interpolated_,fit_y_v2_interpolated_,fit_y_v3_interpolated_,fit_y_v4_interpolated_,fit_y_v5_interpolated_= find_optimum(start, end, target_data,target_data_density,target_data_v)

        list_param1.append(fit_param1)
        list_param2.append(fit_param2)
        list_param3.append(fit_param3)
        list_tau.append(min_error_scaling_factor[0])
        print(fit_match_num, fit_param1,fit_param3,fit_param2,min_error_scaling_factor[0])



fig, axs = plt.subplots(1, 4, figsize=(12, 4))
# Convert all elements to float
list_tau_cleaned = [float(tau) if isinstance(tau, np.ndarray) else tau for tau in list_tau]


# Create box plots instead of histograms
axs[0].boxplot(list_param1, vert=True, patch_artist=True)
axs[0].set_title('Param1')

axs[1].boxplot(list_param2, vert=True, patch_artist=True)
axs[1].set_title('Param2')

axs[2].boxplot(list_param3, vert=True, patch_artist=True)
axs[2].set_title('Param3')

axs[3].boxplot(list_tau_cleaned, vert=True, patch_artist=True)
axs[3].set_title('Tau')

# Improve layout for better visualization
plt.tight_layout()
plt.show()


#3.1 1.5 1.1 57.65356551991345
#2.1 1.2 1.1 53.27979511887196

#0.72,0.31513434556937703,0.6575671727846885,1.0,0.9055619320540211,0.8111238641080422,0.23595120482069282

#5.1 0.0 6.1 100.0
#5.1 0.0 1.1 8.125186333078895



#
# 2.1 0.0 0.1 6.9935301589270695
# 5.1 0.0 6.1 72.39526443076161
