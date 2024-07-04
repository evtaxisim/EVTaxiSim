################################ Read Me #######################################
"""
This program creates the results after sim_charge has completed its simulations.
It does it for the scenario where Home Charging is false and for the scenario
where Home Charging is true. It needs to be done seperately as the results are
completely different.
    False   - A histogram is plotted for the total number of iterations it took
            for the simulation to reach steady state. The number of iteratioons
            is placed on the x-axis and the count is placed on the y-axis. The 
            total is also printed out. The positive steady state is given above
            the x-axis and the results where the steady state was zero are given
            below the x-axis
            - The second plot distinguishes the distances that resulted in a 
            positive steady and a steady state that was zero
"""


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

#source_folder = 'D:/Masters/Simulations/Simulation_4/Outputs/EV-Fleet-Sim_Results/' 
#source_folder = 'D:/Masters/Simulations/Simulation_4/Outputs/ICE_Data_Results/' 
source_folder = 'D:/Masters/Simulations/Simulation_4/Outputs/'
num_vehicles = 17
save_common = 'Day_'
days = [str(num).zfill(2) for num in range(0, 32)]  # Days in the month

plt.rcParams['figure.dpi'] = 600

positive_vehicle_days = []
battery_vehicle_days = []
chargers_vehicle_days = []
positive_vehicle_days_true = []
battery_vehicle_days_true = []
chargers_vehicle_days_true = []

positive_vehicle_count = {
    "Vehicle_1": 0,
    "Vehicle_2": 0,
    "Vehicle_3": 0,
    "Vehicle_4": 0,
    "Vehicle_5": 0,
    "Vehicle_6": 0,
    "Vehicle_8": 0,
    "Vehicle_9": 0,
    "Vehicle_10": 0,
    "Vehicle_11": 0,
    "Vehicle_12": 0,
    "Vehicle_13": 0,
    "Vehicle_14": 0,
    "Vehicle_15": 0,
    "Vehicle_16": 0,
    "Vehicle_17": 0,
}

total_vehicle_count = {
    "Vehicle_1": 0,
    "Vehicle_2": 0,
    "Vehicle_3": 0,
    "Vehicle_4": 0,
    "Vehicle_5": 0,
    "Vehicle_6": 0,
    "Vehicle_8": 0,
    "Vehicle_9": 0,
    "Vehicle_10": 0,
    "Vehicle_11": 0,
    "Vehicle_12": 0,
    "Vehicle_13": 0,
    "Vehicle_14": 0,
    "Vehicle_15": 0,
    "Vehicle_16": 0,
    "Vehicle_17": 0,
}

scheduled_vehicle_success = {
    "Vehicle_1": 100,
    "Vehicle_2": 100,
    "Vehicle_3": 100,
    "Vehicle_4": 100,
    "Vehicle_5": 100,
    "Vehicle_6": 100,
    "Vehicle_8": 100,
    "Vehicle_9": 100,
    "Vehicle_10": 100,
    "Vehicle_11": 100,
    "Vehicle_12": 100,
}



def create_checkerboard(values_lists):
    # Determine the number of rows and columns based on the length of the input lists
    rows = len(values_lists)
    cols = max(len(values) for values in values_lists)

    # Set the figure size based on the number of rows and columns
    fig_width = cols
    fig_height = rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Normalize values to create a gradient colormap
    max_value = 183
    norm_values_lists = [np.array(values) / max_value for values in values_lists]

    # Plot the checkerboard pattern with color gradients
    for row, norm_values in enumerate(norm_values_lists):
        for col, norm_value in enumerate(norm_values):
            # Determine the color based on the normalized values and group
            if col < len(values_lists[row]):

                if row in [0, 4]:
                    color = plt.cm.YlGn(norm_value)
                else:
                    color = plt.cm.YlOrRd(norm_value)

                value = values_lists[row][col]
                percentage = round((value / max_value) * 100)

                # Add a rectangle to represent the box
                rectangle = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rectangle)

                # Add text with values and percentages in the center of each box
                plt.text(col + 0.5, row + 0.5, f'{value}\n({percentage}%)', color='black',
                        ha='center', va='center', fontsize=10)

    # Label the x-axis with indices N1 to Nx in the middle of the blocks
    ax.set_xticks([i + 0.5 for i in range(cols)])
    ax.set_xticklabels([f'{i+1}' for i in range(cols)])

    # Set axis heading
    ax.set_xlabel('Number of Depot Chargers')

    # Remove axis labels and ticks on the left y-axis
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Positive', 'Zero - Chargers', 'Zero - Battery'], rotation=45)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Add a secondary y-axis on the right side with custom labels
    ax2 = ax.twinx()
    ax2.set_yticks([1.5])
    ax2.set_yticklabels(['Depot Charging'])
    ax2.set_ylim(ax.get_ylim())

    ax2.set_yticklabels(['Depot Charging'], rotation=90, va='center')

    # Remove axis labels and ticks on the right y-axis
    ax2.yaxis.tick_right()
    ax2.yaxis.set_tick_params(width=0)

    plt.tight_layout()

    print('Saving Results')

    save_path = source_folder + '/Results_Plot.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = source_folder + '/Results_Plot.svg'
    plt.savefig(save_path, format='svg')
    # Save the plot to a specific location as a svg
    save_path = source_folder + '/Results_Plot.pdf'
    plt.savefig(save_path, format='pdf')

    plt.close()

def plot_percentage_bar_graph(total_values, positive_values, additional_percentages, folder):
    # Calculate percentages
    percentages = {}
    for key in total_values.keys():
        total = total_values[key]
        positive = positive_values.get(key, 0)
        percentage_positive = (positive / total) * 100 if total != 0 else 0
        percentages[key] = percentage_positive

    # Sort the dictionary by key
    sorted_percentages = dict(sorted(percentages.items(), key=lambda item: int(item[0].split('_')[1])))
    sorted_additional_percentages = dict(sorted(additional_percentages.items(), key=lambda item: int(item[0].split('_')[1])))

    # Separate keys and values
    keys = list(sorted_percentages.keys())
    values = list(sorted_percentages.values())
    additional_values = [sorted_additional_percentages.get(key, 0) for key in keys]  # Align keys and get values, default to 0 if key not present


    bar_width = 0.35

    # Calculate remainder percentages
    remainder_values = [100 - value for value in values]

    # Plot the bar graph
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(keys)) + bar_width + 0.1, values, bar_width, color='#fac50a', label='Simulated Success Rate')
    ax.bar(np.arange(len(keys)) + bar_width + 0.1, remainder_values, bar_width, bottom=values, color='#ADD8E6')

    ax.bar(np.arange(len(keys)) + 2 * (bar_width + 0.1), additional_values, bar_width, color='#04b321', label='Scheduled Success Rate', alpha=0.5)

    ax.set_xlabel('Vehicle')
    ax.set_ylabel('Percentage (%)')

    numbers = list(range(1, 17)) 

    # Set the ticks and labels
    ax.set_xticks(np.arange(len(keys)) + 1.5 * (bar_width + 0.1))  # Set ticks at positions 0, 1, 2, ...
    ax.set_xticklabels(numbers)  # Set corresponding labels

    # Create a legend
    ax.legend()

    # Set y-axis limits
    ax.set_ylim(0, 120)

    # Show plot
    plt.tight_layout()

    save_path = folder + '/Vehicle_Success_Rate.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = folder + '/Vehicle_Success_Rate.svg'
    plt.savefig(save_path, format='svg')
    # Save the plot to a specific location as a svg
    save_path = folder + '/Vehicle_Success_Rate.pdf'
    plt.savefig(save_path, format='pdf')

    plt.close()

#################################################################################################################
################################## Distance and Steady State Reached ############################################
#################################################################################################################

print('Results - No Home Charging')

sce_folders = glob.glob(os.path.join(source_folder, 'SCE*'))

### For each SCE folder that has HC = False
for sce_folder in sce_folders:

    positive_steady_state = 0
    positive_steady_state_oscilation = 0
    zero_steady_state_battery = 0
    zero_steady_state_chargers = 0

    postive_distances = []
    postive_distances_oscilation = []
    zero_distances_battery = []
    zero_distances_chargers = []

    total_vehicle_days = 0

    day_subfolders = glob.glob(os.path.join(sce_folder, 'Day*'))
    sce_folder_name = os.path.basename(sce_folder)

    print(sce_folder_name)

    min_non_zero_values = []
    final_non_zero_values = []

    ### For each day folder in the SCE folder
    for day_folder in day_subfolders:

        day_folder_name = os.path.basename(day_folder)
        iteration_subfolders = sorted(glob.glob(os.path.join(day_folder, 'Iteration*')), key=lambda x: int(x.split('_')[-1]))

        vehicle_steady_state = {'Vehicle_' + str(i): False for i in range(1, num_vehicles + 1)}
        vehicle_zero_steady_state = {'Vehicle_' + str(i): False for i in range(1, num_vehicles + 1)}
        vehicle_iteration_steady_state = {'Vehicle_' + str(i): False for i in range(1, num_vehicles + 1)}
        
        ### For each iteration in the day folder
        for iteration_folder in iteration_subfolders:

            iteration_folder_name = os.path.basename(iteration_folder)
            iteration_number = int(iteration_folder_name.split('_')[-1])
            
            soc_file_path = os.path.join(iteration_folder, 'soc.csv')
            dis_file_path = os.path.join(iteration_folder, 'distance.csv')

            soc_dataframe = pd.read_csv(soc_file_path)
            dis_dataframe = pd.read_csv(dis_file_path)


            ### Determine if steady state is reached
            if iteration_number == 1:
                steady_state_soc = pd.DataFrame(100, index = [1], columns=soc_dataframe.columns)
            if iteration_number == 2:
                columns_with_zero = soc_dataframe.columns[(soc_dataframe.iloc[0] == 0) & (soc_dataframe.iloc[-1] == 0)].tolist()

                for col in columns_with_zero:
                    vehicle_zero_steady_state[col] = True

            steady_state_soc = pd.concat([steady_state_soc, soc_dataframe.iloc[[-1]]], ignore_index=True)

            last_iteration_folder = iteration_folder

        ### Gather information regarding the SOC behaviour of the vehicles
        if last_iteration_folder:
            soc_file_path_2 = os.path.join(last_iteration_folder, 'soc.csv')
            soc_dataframe_2 = pd.read_csv(soc_file_path_2)

            for column in soc_dataframe_2.columns:
                non_zero_values = soc_dataframe[column][soc_dataframe[column] != 0]

                ### Add to the total active vehicles
                total_vehicle_count[column] += 1

                # Check if there are non-zero values in the column
                if not non_zero_values.empty:
                    # Find minimum non-zero value and add it to min_non_zero_values list
                    min_non_zero_value = non_zero_values.min()
                    min_non_zero_values.append(min_non_zero_value)

                    # Find final non-zero value and add it to final_non_zero_values list
                    final_non_zero_value = non_zero_values.iloc[-1]  # Get the last non-zero value
                    final_non_zero_values.append(final_non_zero_value)



        # Only evaluating on the last iteration because that is when everything reached steady state
        total_length = len(steady_state_soc)

        max_mask = iteration_number // 2

        if max_mask != 0:

            # check for each size of the mask. The max mask size can only be half of the total number of rows
            for k in range(0, max_mask + 1):

                if k != 0:

                    begin_start_row = total_length - k
                    begin_end_row = total_length - 1

                    end_start_row = total_length - 2*k 
                    end_end_row = total_length - k - 1

                    # cycle through each vehicle and check
                    for vehicle_name in steady_state_soc.columns:

                        if vehicle_steady_state[vehicle_name] == False:

                            first_values = np.array(steady_state_soc.loc[begin_start_row:begin_end_row, vehicle_name].to_list())

                            end_values = np.array(steady_state_soc.loc[end_start_row:end_end_row, vehicle_name].to_list())

                            differences = np.abs(first_values - end_values)

                            if np.all(differences <= 1):
                                vehicle_steady_state[vehicle_name] = True
                                distance_sum = dis_dataframe[vehicle_name].sum() / 1000

                                if all(val == 0 for val in first_values) and all(val == 0 for val in end_values):
                                    ### Add steady state success to output
                                    if vehicle_zero_steady_state[vehicle_name] == True:
                                        zero_distances_battery.append(distance_sum)
                                        zero_steady_state_battery = zero_steady_state_battery + 1
                                    else:
                                        zero_distances_chargers.append(distance_sum)
                                        zero_steady_state_chargers = zero_steady_state_chargers + 1

                                ### If positive steady state has been reached        
                                else:
                                    # See which number steady state was reached
                                    if k <= 1:
                                        postive_distances.append(distance_sum)
                                        positive_steady_state = positive_steady_state + 1

                                        # Add for each vehicle
                                        positive_vehicle_count[vehicle_name] += 1
                                    else:
                                        postive_distances_oscilation.append(distance_sum)
                                        positive_steady_state_oscilation = positive_steady_state_oscilation + 1
                                        

            vehicle_steady_state = {'Vehicle_' + str(i): False for i in range(1, num_vehicles + 1)}
                
        total_vehicle_days = total_vehicle_days + len(soc_dataframe.columns)

    print('Vehicle - Total (Positive)')
    for key in total_vehicle_count:
        
        print(f'{key}: {total_vehicle_count[key]} ({positive_vehicle_count[key]})')
    

    # Plot the success rate per vehicle
    plot_percentage_bar_graph(total_vehicle_count, positive_vehicle_count, scheduled_vehicle_success, sce_folder)

    positive_vehicle_count = {
        "Vehicle_1": 0,
        "Vehicle_2": 0,
        "Vehicle_3": 0,
        "Vehicle_4": 0,
        "Vehicle_5": 0,
        "Vehicle_6": 0,
        "Vehicle_8": 0,
        "Vehicle_9": 0,
        "Vehicle_10": 0,
        "Vehicle_11": 0,
        "Vehicle_12": 0,
        "Vehicle_13": 0,
        "Vehicle_14": 0,
        "Vehicle_15": 0,
        "Vehicle_16": 0,
        "Vehicle_17": 0,
    }

    total_vehicle_count = {
        "Vehicle_1": 0,
        "Vehicle_2": 0,
        "Vehicle_3": 0,
        "Vehicle_4": 0,
        "Vehicle_5": 0,
        "Vehicle_6": 0,
        "Vehicle_8": 0,
        "Vehicle_9": 0,
        "Vehicle_10": 0,
        "Vehicle_11": 0,
        "Vehicle_12": 0,
        "Vehicle_13": 0,
        "Vehicle_14": 0,
        "Vehicle_15": 0,
        "Vehicle_16": 0,
        "Vehicle_17": 0,
    }



    ################################### Vehicle-day success rate ###########################################
    print(f"Total Vehicle Days: {total_vehicle_days}")

    print(f"Total Positive Steady State: {positive_steady_state}")
    #print(f"Total Positive Steady State - Oscilations: {positive_steady_state_oscilation}")
    print(f"Total Zero Steady State - Battery: {zero_steady_state_battery}")
    print(f"Total Zero Steady State - Charging: {zero_steady_state_chargers}")           

    print("Saving Graphs")

    ### Append to the list if home charging is true or false
    last_word = sce_folder_name.split("_")[-1]
    if last_word == 'True':
        positive_vehicle_days_true.append(positive_steady_state + positive_steady_state_oscilation)
        chargers_vehicle_days_true.append(zero_steady_state_chargers)
        battery_vehicle_days_true.append(zero_steady_state_battery)
    elif last_word == 'False':
        positive_vehicle_days.append(positive_steady_state + positive_steady_state_oscilation)
        chargers_vehicle_days.append(zero_steady_state_chargers)
        battery_vehicle_days.append(zero_steady_state_battery)
    

    #################################### Distance distribution ##############################################
    # Calculate the range of the data for the non-empty sequences
    data_range_1 = max(postive_distances) - min(postive_distances) if postive_distances else 0
    data_range_2 = max(zero_distances_battery) - min(zero_distances_battery) if zero_distances_battery else 0
    data_range_3 = max(zero_distances_chargers) - min(zero_distances_chargers) if zero_distances_chargers else 0
    data_range_4 = max(postive_distances_oscilation) - min(postive_distances_oscilation) if postive_distances_oscilation else 0

    desired_bin_width = 5  # Adjust as needed

    # Calculate the number of bins based on the desired bin width
    num_bins_1 = max(1, int(np.ceil(data_range_1 / desired_bin_width)))
    num_bins_2 = max(1, int(np.ceil(data_range_2 / desired_bin_width)))
    num_bins_3 = max(1, int(np.ceil(data_range_3 / desired_bin_width)))
    num_bins_4 = max(1, int(np.ceil(data_range_4 / desired_bin_width)))


    ### Plot the distance distribution for zero and positive steady state
    plt.figure()
    # Create a histogram for the list above x-axis
    plt.hist(postive_distances, bins=num_bins_1, color='#FFA500', alpha=0.7, label='Positive Steady State')
    #plt.hist(postive_distances_oscilation, bins=num_bins_4, color='#FFE6B9', alpha=0.7, label='Positive Steady State - Oscilation')

    # Create a histogram for the list below x-axis
    plt.hist(zero_distances_battery, bins=num_bins_2, color='#2D71E6', alpha=0.7, label='0% Steady State - Battery')
    plt.hist(zero_distances_chargers, bins=num_bins_3, color='#ADD8E6', alpha=0.7, label='0% Steady State - Charging')

    #plt.title("Distance Distribution for Steady State")
    plt.xlabel("Distance per Vehicle-day [km]", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.legend(fontsize=16)

    plt.tight_layout()

    save_path = sce_folder + '/Distance_Histogram.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Distance_Histogram.svg'
    plt.savefig(save_path, format = 'svg')
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Distance_Histogram.pdf'
    plt.savefig(save_path, format = 'pdf')

    plt.close()

    ### Plot distance bars

    match = re.search(r'N(\d+)', sce_folder_name)
    extracted_part = match.group(1)

    last_word = sce_folder_name.split("_")[-1]

    plt.figure()

    # Create a histogram for the list above x-axis
    plt.figure(figsize=(10, 4))

    # Create a histogram for the list below x-axis
    plt.vlines(x=zero_distances_battery, ymin=0, ymax=1, colors='#2D71E6', label='0% Steady State - Battery')
    plt.vlines(x=zero_distances_chargers, ymin=0, ymax=1, colors='#ADD8E6', label='0% Steady State - Charging')
    plt.vlines(x=postive_distances, ymin=0, ymax=1, colors='#FFA500', label='Positive Steady State')

    if  zero_distances_battery:
        plt.fill_betweenx(y=[0, 1], x1=min(zero_distances_battery), x2=max(zero_distances_battery), color='#2D71E6', alpha=0.3)
    if  zero_distances_chargers:
        plt.fill_betweenx(y=[0, 1], x1=min(zero_distances_chargers), x2=max(zero_distances_chargers), color='#ADD8E6', alpha=0.3)
    if  postive_distances:
        plt.fill_betweenx(y=[0, 1], x1=min(postive_distances), x2=max(postive_distances), color='#FFA500', alpha=0.3)

    plt.yticks([])

    plt.ylabel(f"N{extracted_part} {last_word}", fontsize=16)

    plt.xlabel("Distance per Vehicle-day [km]", fontsize=16)

    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol = 2)

    plt.tight_layout()
    
    
    save_path = sce_folder + '/Distance_Bar.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Distance_Bar.svg'
    plt.savefig(save_path, format = 'svg', bbox_inches='tight', pad_inches=0.1)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Distance_Bar.pdf'
    plt.savefig(save_path, format = 'pdf', bbox_inches='tight', pad_inches=0.1)

    plt.close()


    ##################################### SOC distribution ########################################

    combined_data = np.vstack([min_non_zero_values, final_non_zero_values])
    kde = gaussian_kde(combined_data, bw_method=2)

    x_range = np.linspace(0, 100, 1000)
    y_range = np.linspace(0, 100, 1000)

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    points = np.vstack([x_grid.ravel(), y_grid.ravel()])

    probabilities = kde(points)

    # Reshape the probabilities to match the grid shape
    probabilities_grid = probabilities.reshape(x_grid.shape)

    # Plot the heatmap
    plt.imshow(probabilities_grid, cmap='YlGnBu', origin='lower', extent=[0, 100, 0, 100])
    plt.colorbar()

    plt.scatter(min_non_zero_values, final_non_zero_values, color='black', label='Vehicle-days', s=5)

    # Add labels and title
    plt.xlabel('Minimum SOC')
    plt.ylabel('Final SOC')
    plt.title('Available Battery Energy')
    plt.tight_layout()

    save_path = sce_folder + '/SOC_Distribution.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/SOC_Distribution.svg'
    plt.savefig(save_path, format = 'svg')
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/SOC_Distribution.pdf'
    plt.savefig(save_path, format = 'pdf')

    plt.close()

############################# Results Plot ################################

#create_checkerboard([positive_vehicle_days_true, chargers_vehicle_days_true, battery_vehicle_days_true, positive_vehicle_days, chargers_vehicle_days, battery_vehicle_days])
create_checkerboard([positive_vehicle_days, chargers_vehicle_days, battery_vehicle_days])


vehicle_days_results = {
    'Positive_False': positive_vehicle_days,
    'Chargers_False': chargers_vehicle_days,
    'Battery_False': battery_vehicle_days,
    'Positive_True': positive_vehicle_days_true,
    'Chargers_True': chargers_vehicle_days_true,
    'Battery_True': battery_vehicle_days_true
}

print(vehicle_days_results)
vehicle_days_results = pd.DataFrame(vehicle_days_results)

save_path = source_folder + 'results_vehicle_days.csv'
vehicle_days_results.to_csv(save_path, index=False)




"""
#################################################################################################################
################################# Scenario for Home Charging = True #############################################
#################################################################################################################

print('Results - Home Charging')

### Home charging scenario
sce_folders = glob.glob(os.path.join(source_folder, 'SCE*True'))

total_vehicle_days = 0

### For each SCE folder that has HC = False
for sce_folder in sce_folders:

    day_exists = {save_common + day: False for day in days}

    day_subfolders = glob.glob(os.path.join(sce_folder, 'Day*'))
    sce_folder_name = os.path.basename(sce_folder)

    vehicle_total_trips = {'Vehicle_' + str(i): 0 for i in range(1, num_vehicles + 1)}
    vehicle_completed_trips = {'Vehicle_' + str(i): 0 for i in range(1, num_vehicles + 1)}

    day_total_trips = {save_common + day: 0 for day in days}
    day_completed_trips = {save_common + day: 0 for day in days}

    vehicle_end_soc = {'Vehicle_' + str(i): 0 for i in range(1, num_vehicles + 1)}
    day_end_soc = {save_common + day: 0 for day in days}

    hundred_distances = []
    all_positive_distances = []
    zero_distances_chargers = []
    zero_distances_battery = []

    hundred_steady_state = 0
    all_positive_steady_state = 0
    zero_steady_state_chargers = 0
    zero_steady_state_battery = 0

    print(sce_folder_name)

    ### For each day folder in the SCE folder
    for day_folder in day_subfolders:

        day_folder_name = os.path.basename(day_folder)
        day_exists[day_folder_name] = True

        soc_file_path = os.path.join(day_folder, 'soc.csv')
        dis_file_path = os.path.join(day_folder, 'distance.csv')

        soc_dataframe = pd.read_csv(soc_file_path)
        dis_dataframe = pd.read_csv(dis_file_path)

        first_row = soc_dataframe.iloc[0]
        last_row = soc_dataframe.iloc[-1]

        day_total_trips[day_folder_name] = len(soc_dataframe.columns)

        total_vehicle_days = total_vehicle_days + len(soc_dataframe.columns)

        for column_name in soc_dataframe.columns:

            distance_sum = dis_dataframe[column_name].sum() / 1000 # change meters to kilometers

            vehicle_total_trips[column_name] = vehicle_total_trips[column_name] + 1

            if (soc_dataframe[column_name] > 0).all():
                vehicle_completed_trips[column_name] = vehicle_completed_trips[column_name] + 1
                day_completed_trips[day_folder_name] = day_completed_trips[day_folder_name] + 1

                all_positive_distances.append(distance_sum)
                all_positive_steady_state = all_positive_steady_state + 1

            if last_row[column_name] >= 98:
                vehicle_end_soc[column_name] = vehicle_end_soc[column_name] + 1
                day_end_soc[day_folder_name] = day_end_soc[day_folder_name] + 1

                hundred_distances.append(distance_sum)
                hundred_steady_state = hundred_steady_state + 1
            else:
                if (soc_dataframe[column_name] <= 0).any():
                    zero_distances_battery.append(distance_sum)
                    zero_steady_state_battery = zero_steady_state_battery + 1
                else:
                    zero_distances_chargers.append(distance_sum)
                    zero_steady_state_chargers = zero_steady_state_chargers + 1


    print(f"Total Vehicle Days: {total_vehicle_days}")
    print(f"Total Postive Days: {all_positive_steady_state}")
    print(f"Total 100% Steady State: {hundred_steady_state}")
    print(f"Total Zero - Chargers: {zero_steady_state_chargers}")
    print(f"Total Zero - Battery: {zero_steady_state_battery}")



    print("Saving Graphs")

    # Calculate the range of the data
    data_range_1 = max(hundred_distances) - min(hundred_distances)
    data_range_2 = max(zero_distances_chargers) - min(zero_distances_chargers)
    data_range_3 = max(zero_distances_battery) - min(zero_distances_battery)

    desired_bin_width = 5  # Adjust as needed

    # Calculate the number of bins based on the desired bin width
    num_bins_1 = int(np.ceil(data_range_1 / desired_bin_width))
    num_bins_2 = int(np.ceil(data_range_2 / desired_bin_width))
    num_bins_3 = int(np.ceil(data_range_3 / desired_bin_width))

    ### Plot the distance distribution for zero and positive steady state
    plt.figure()
    # Create a histogram for the list above x-axis
    plt.hist(hundred_distances, bins=num_bins_1, color='#FFA500', alpha=0.7, label='100% Steady State')

    # Create a histogram for the list below x-axis
    plt.hist(zero_distances_battery, bins=num_bins_3, color='#2D71E6', alpha=0.7, label='Non Steady State - Battery')
    plt.hist(zero_distances_chargers, bins=num_bins_2, color='#ADD8E6', alpha=0.7, label='Non Steady State - Charging')

    plt.title("Distance Distribution for Steady State")
    plt.xlabel("Distance per Day [km]")
    plt.ylabel("Frequency")
    plt.legend()

    save_path = sce_folder + '/Distance_Histogram.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Distance_Histogram.svg'
    plt.savefig(save_path, format = 'svg')

    plt.close()

    ### Vehicle Succesful Trips for day - was it able to stay above 0%
    # Calculate completion and uncompletion percentages
    completion_percentages = []
    for vehicle in vehicle_total_trips:
        if vehicle_total_trips[vehicle] != 0:
            completion_percentage = (vehicle_completed_trips[vehicle] / vehicle_total_trips[vehicle]) * 100
        else:
            completion_percentage = 0
        completion_percentages.append(completion_percentage)

    uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

    # Create the figure and axis objects
    fig, ax = plt.subplots()
    x = np.arange(len(vehicle_total_trips)) * 1.7
    bar_width = 1
    bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Completed Trips', color = '#FFA500')
    bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Uncompleted Trips', color = '#ADD8E6')

    for rect, completion_percentage in zip(bar1 + bar2, completion_percentages):
        height = rect.get_height()
        if completion_percentage > 0:
                if completion_percentage < 10:
                    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
                else:
                    ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)

    ax.set_xticks(x)
    ax.set_xticklabels(range(1, num_vehicles + 1), fontsize = 6)

    ax.set_ylabel('Percentage [%]')
    ax.set_ylim(0, 115)

    ax.set_title('Vehicle_Day Completion Rate')
    ax.set_xlabel('Vehicle')
    plt.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()

    save_path = sce_folder + '/Vehicle_Day_Trip_Completion.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Vehicle_Day_Trip_Completion.svg'
    plt.savefig(save_path, format = 'svg')



    ### Succesful Day Trips - did all the vehicles of that day stay above 0%
    # Calculate completion and uncompletion percentages
    completion_percentages = []
    for vehicle in day_total_trips:
        if day_total_trips[vehicle] != 0:
            completion_percentage = (day_completed_trips[vehicle] / day_total_trips[vehicle]) * 100
        else:
            completion_percentage = 0
        completion_percentages.append(completion_percentage)
    uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

    # Create the figure and axis objects
    fig, ax = plt.subplots()
    x = np.arange(len(day_total_trips)) * 3
    bar_width = 2
    bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Completed Trips', color = '#FFA500')
    bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Uncompleted Trips', color = '#ADD8E6')

    for rect, completion_percentage, vehicle_name in zip(bar1 + bar2, completion_percentages, day_total_trips.keys()):
        if day_exists[vehicle_name]:
            height = rect.get_height()
            if completion_percentage > 0:
                if completion_percentage < 10:
                    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
                else:
                    ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)


    for i, exists in enumerate(day_exists.values()):
        if not exists:
            bar1[i].set_height(0)
            bar2[i].set_height(0)

    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(days) + 1), fontsize = 6)

    ax.set_ylabel('Percentage [%]')
    ax.set_xlabel('Day')
    ax.set_ylim(0, 115)

    ax.set_title('Daily Completion Rate')
    plt.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()

    save_path = sce_folder + '/Daily_Valid_Trip_Completion.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Daily_Valid_Trip_Completion.svg'
    plt.savefig(save_path, format = 'svg')
                


    ### Vehicle Valid Trips for next day - was it able to get back to 0%
    # Calculate completion and uncompletion percentages
    completion_percentages = []
    for vehicle in vehicle_total_trips:
        if vehicle_total_trips[vehicle] != 0:
            completion_percentage = (vehicle_completed_trips[vehicle] / vehicle_total_trips[vehicle]) * 100
        else:
            completion_percentage = 0
        completion_percentages.append(completion_percentage)

    uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

    # Create the figure and axis objects
    fig, ax = plt.subplots()
    x = np.arange(len(vehicle_total_trips)) * 1.7
    bar_width = 1
    bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Valid', color = '#FFA500')
    bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Invalid', color = '#ADD8E6')

    for rect, completion_percentage in zip(bar1 + bar2, completion_percentages):
        height = rect.get_height()
        if completion_percentage > 0:
                if completion_percentage < 10:
                    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
                else:
                    ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)

    ax.set_xticks(x)
    ax.set_xticklabels(range(1, num_vehicles + 1), fontsize = 6)

    ax.set_ylabel('Percentage [%]')
    ax.set_ylim(0, 115)

    ax.set_title('Vehicle_Day Valid Completion Rate')
    ax.set_xlabel('Vehicle')
    plt.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()

    save_path = sce_folder + '/Vehicle_Day_Valid_Completion.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Vehicle_Day_Valid_Completion.svg'
    plt.savefig(save_path, format = 'svg')



    ### Valid for next Day Trips - did all the vehicles manage to get to 100% SOC at the end of the day
    # Calculate completion and uncompletion percentages
    completion_percentages = []
    for vehicle in day_total_trips:
        if day_total_trips[vehicle] != 0:
            completion_percentage = (day_completed_trips[vehicle] / day_total_trips[vehicle]) * 100
        else:
            completion_percentage = 0
        completion_percentages.append(completion_percentage)
    uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

    # Create the figure and axis objects
    fig, ax = plt.subplots()
    x = np.arange(len(day_total_trips)) * 3
    bar_width = 2
    bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Validity', color = '#FFA500')
    bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Invalidity', color = '#ADD8E6')

    for rect, completion_percentage, vehicle_name in zip(bar1 + bar2, completion_percentages, day_total_trips.keys()):
        if day_exists[vehicle_name]:
            height = rect.get_height()
            if completion_percentage > 0:
                if completion_percentage < 10:
                    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
                else:
                    ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)


    for i, exists in enumerate(day_exists.values()):
        if not exists:
            bar1[i].set_height(0)
            bar2[i].set_height(0)

    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(days) + 1), fontsize = 6)

    ax.set_ylabel('Percentage [%]')
    ax.set_xlabel('Day')
    ax.set_ylim(0, 115)

    ax.set_title('Daily Validity Rate')
    plt.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()

    save_path = sce_folder + '/Daily_Valid_Next_Trip.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = sce_folder + '/Daily_Valid_Next_Trip.svg'
    plt.savefig(save_path, format = 'svg')
    
    """