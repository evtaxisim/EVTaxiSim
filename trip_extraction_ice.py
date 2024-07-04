################################################## READ ME ##################################################
"""
    Once the stop_locations have been found, this programme is responsible for using that information as a
    mask in order to obtain the trips that taxis have done.

    The program works by taking the mask and placing the stops according to the 20 minute stop criteria. This
    is then used to split the according to trips and voyages
    - voyages are from energy source to energy source
    - trips are from any 20-minute stop to the next

"""


import os
import pandas as pd
from tqdm import tqdm
import folium
from folium.plugins import HeatMap
from branca.colormap import StepColormap
from folium import plugins
from haversine import haversine
import re
import numpy as np


source_folder = 'D:/Masters/Simulations/Simulation_4/Usable_Data/'
source_folder_2 = 'D:/Masters/Simulations/Simulation_4/Trip_Data/'
save_folder_1 = 'D:/Masters/Simulations/Simulation_4/Trip_Data/Trips'


total_files = len([file for root, dirs, files in os.walk(source_folder) for file in files if file == 'vehicle_day_min.csv']) + 17
filtered_locations_name = 'filtered_locations.csv'
allowable_radius = 150 #m
allowable_energy_expenditure = 55 # kWh

file_path = os.path.join(source_folder_2, filtered_locations_name)
filtered_locations = pd.read_csv(file_path)

filtered_voyages = filtered_locations[~filtered_locations['Name'].str.startswith('S')]
filtered_voyages = filtered_voyages.reset_index(drop=True)

vehicle_trips = pd.DataFrame(columns=['Day', 'Start_Time', 'End_Time', 'Start_Location', 'End_Location', 'Distance_Travelled', 'Energy_Consumed'])
vehicle_voyages = pd.DataFrame(columns=vehicle_trips.columns)

##################################################################################################
######################################## Main Code ###############################################
######################### Vehicle trips with large expenditure ###################################
##################################################################################################

print('Large Expenditure')

for root, dirs, files in tqdm(os.walk(source_folder), total=total_files, desc='Processing Files'):

    for file in files:

        if file == 'vehicle_day_min.csv':

            match = re.search(r'Vehicle_(\d{1,2})_(\d+)', root)
            vehicle_number = int(match.group(1))
            day_number = int(match.group(2))

            file_path = os.path.join(root, file)

            # Read the CSV file using pandas and append it to the csv_data list
            vehicle_day = pd.read_csv(file_path)
            
            
            # Create groups of consecutive True values
            vehicle_day['Group'] = (vehicle_day['20_Min_Stop'] != vehicle_day['20_Min_Stop'].shift(1)).cumsum()


            ### For vehicle_trips
            result_df = vehicle_day[vehicle_day['20_Min_Stop']].groupby('Group').apply(lambda group_df: group_df.iloc[[0]].apply(
                lambda row: pd.Series({
                    'start_index': group_df.index.min(),
                    'end_index': group_df.index.max(),
                    'min_distance_name': filtered_locations.iloc[np.argmin([
                        haversine((lat, lon), (row['Latitude'], row['Longitude']), unit='m')
                        for lat, lon in zip(filtered_locations['Latitude'], filtered_locations['Longitude'])
                    ])].get('Name')
                }), axis=1)).reset_index(drop=True)
            

            max_index = result_df.index.max()

            for index, row in result_df.iterrows():

                # If it is the last index, do not perform any calculations
                if index == max_index:
                    break
                else:

                    first_index = result_df.loc[index, 'end_index'] + 1
                    last_index = result_df.loc[index + 1, 'start_index'] - 1


                    # Get energy consumption for trip
                    trip_energy = vehicle_day.loc[first_index:last_index, 'Energy_Consumption'].sum()
                    trip_distance = vehicle_day.loc[first_index:last_index, 'Distance'].sum()

                    vehicle_trips = vehicle_trips.append({
                        'Day': day_number,
                        'Start_Time': vehicle_day.loc[first_index, 'Time_of_Day'],
                        'End_Time': vehicle_day.loc[last_index, 'Time_of_Day'],
                        'Start_Location': result_df.loc[index, 'min_distance_name'],
                        'End_Location': result_df.loc[index + 1, 'min_distance_name'],
                        'Distance_Travelled': trip_distance,
                        'Energy_Consumed': trip_energy
                    }, ignore_index=True)


            ### For vehicle_voyages
            result_voyages = result_df[~result_df['min_distance_name'].str.startswith('S')]
            result_voyages = result_voyages.reset_index(drop=True)
            
            max_index = result_voyages.index.max()

            for index, row in result_voyages.iterrows():

                # If it is the last index, do not perform any calculations
                if index == max_index:
                    break
                else:

                    first_index = result_voyages.loc[index, 'end_index'] + 1
                    last_index = result_voyages.loc[index + 1, 'start_index'] - 1


                    # Get energy consumption for trip
                    trip_energy = vehicle_day.loc[first_index:last_index, 'Energy_Consumption'].sum()
                    trip_distance = vehicle_day.loc[first_index:last_index, 'Distance'].sum()

                    vehicle_voyages = vehicle_voyages.append({
                        'Day': day_number,
                        'Start_Time': vehicle_day.loc[first_index, 'Time_of_Day'],
                        'End_Time': vehicle_day.loc[last_index, 'Time_of_Day'],
                        'Start_Location': result_voyages.loc[index, 'min_distance_name'],
                        'End_Location': result_voyages.loc[index + 1, 'min_distance_name'],
                        'Distance_Travelled': trip_distance,
                        'Energy_Consumed': trip_energy
                    }, ignore_index=True)

length_filtered_energy_trips = len(vehicle_trips[vehicle_trips['Energy_Consumed'] >= allowable_energy_expenditure*1000])

print(vehicle_trips[vehicle_trips['Energy_Consumed'] >= allowable_energy_expenditure*1000])

print(f'Number of trips > {allowable_energy_expenditure}kWh: {length_filtered_energy_trips}')
print(f'Number of vehicle-trips: {len(vehicle_trips)}')

### Filter the trips to only have trips that are greater than 150m
vehicle_trips = vehicle_trips[vehicle_trips['Distance_Travelled'] >= allowable_radius]
vehicle_trips = vehicle_trips.reset_index(drop=True)

### Filter these trips to create voyages
vehicle_voyages = vehicle_voyages[vehicle_voyages['Distance_Travelled'] >= allowable_radius]
vehicle_voyages = vehicle_voyages.reset_index(drop=True)

print(f'Number of vehicle-trips after {allowable_radius}m: {len(vehicle_trips)}')

#print(vehicle_trips)
#print(vehicle_voyages)



### group the data according days

grouped_trips = vehicle_trips.groupby('Day')

for group_name, group_df in grouped_trips:
    # Customize the file name based on the group name
    file_name = f"Day_{group_name}_Vehicle_Trips.csv"

    file_path = os.path.join(save_folder_1, file_name)

    # Save the group as a CSV file
    group_df.to_csv(file_path, index=False)

    print(f"Saved {file_name}")


grouped_voyages = vehicle_voyages.groupby('Day')

"""
for group_name, group_df in grouped_voyages:
    # Customize the file name based on the group name
    file_name = f"Day_{group_name}_Vehicle_Voyages.csv"

    file_path = os.path.join(save_folder_1, file_name)

    # Save the group as a CSV file
    group_df.to_csv(file_path, index=False)

    print(f"Saved {file_name}")


"""

