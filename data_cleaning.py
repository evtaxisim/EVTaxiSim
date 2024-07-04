################################################## READ ME ##################################################
"""
This program starts by extracting the format required for sim_charge and reframes it according to the vehicle-
day. Furthermore, this is saved to Usable Data to be used in sim_charge. Important to note that this software 
uses a fixed kWh/km value to calculate the energy requirements instead of using ev-fleet-sim.

This is the format required in minutely format

Time_of_Day | Energy_Consumption    | Latitude  | Longitude | Stop      | 20_Min_Stop   | Hub_Location  | Available_Charging    | HC_Location   | Home_Charging |   Distance     
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(string)    | (float)               | (float)   | (float)   | (boolean) | (boolean)     | (boolean)     | (boolean)             | (boolean)     | (boolean)     |   (float)
[YYYY/MM/DD | [Wh/s]                | [degres]  | [degrees] |           |               |               |                       |               |               |   [meters]
    HH:MM:SS]



"""





import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from haversine import haversine
from scipy.stats import mode
import re



source_folder = "D:/Masters/Simulations/Simulation_4/Input/Mobility_Data/"
source_folder_stop = "D:/Masters/Simulations/Simulation_4/Input/Stop_Data/"
destination_folder = "D:/Masters/Simulations/Simulation_4/Usable_Data/"


vehicle_data_dict = {   "1": ["01", "02", "03", "04", "08", "09", "10", "11", "16", "17", "18", "29"],
                        "2": ["01", "02", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "28", "29"],
                        "3": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "28", "29"],
                        "4": ["03", "07", "11", "14", "16", "17", "18", "28", "29"],
                        "5": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "17", "29"],
                        "6": ["03", "04", "10", "11", "17", "18", "29"],
                        "8": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "28", "29"],
                        "9": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "28", "29"],
                        "10": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "28", "29"],
                        "11": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "29"],
                        "12": ["03", "04", "07", "08", "09", "10", "11", "14", "16", "18", "28"],
                        "13": ["17", "29"],
                        "14": ["01", "02", "03", "04", "07", "08", "09", "10", "11", "14", "16", "17", "18", "28"],
                        "15": ["02", "04", "07", "08", "09", "10", "11", "14", "16", "17", "29"],
                        "16": ["02", "04", "08", "10", "14", "16", "18", "28"],
                        "17": ["01", "02", "03", "07", "08", "09", "10", "11", "14"],
}

vehicle_date_change = { "28": "00",
                        "29": "15"

}



folder_prefix = "Vehicle_" 
csv_save_name_1 = 'filled_vehicle_data.csv'
csv_save_name_3 = 'vehicle_day_min.csv'

vehicle_efficiency_055 = 0.55 #kWh/km
vehicle_efficiency_052 = 0.52 #kWh/km
vehicle_efficiency_049 = 0.49 #kWh/km
vehicle_efficiency_040 = 0.4 #kWh/km
vehicle_efficiency_045 = 0.45 #kWh/km
vehicle_efficiency_050 = 0.5 #kWh/km
vehicle_efficiency_060 = 0.6 #kWh/km


vehicle_efficiency = vehicle_efficiency_060


### Box coordinates - Stellenbosch Taxi Rank
stop_location = [
    (-33.932359, 18.857750),  
    (-33.932359, 18.859046),       
    (-33.933172, 18.859046),      
    (-33.933172, 18.857750)       
]

def count_folders_with_prefix(directory_path, prefix):
    folder_count = 0
    for folder_name in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, folder_name)) and folder_name.startswith(prefix):
            folder_count += 1
    return folder_count

def get_last_two_values_as_strings(directory_path):
    folder_names = os.listdir(directory_path)
    last_two_values_array = []
    for folder_name in folder_names:
        if os.path.isdir(os.path.join(directory_path, folder_name)):
            last_two_values = folder_name[-2:]
            last_two_values_array.append(str(last_two_values))
    
    return last_two_values_array


### General Functions
def is_point_in_stop(point):
    lat, lon = point
    latitudes = [coord[0] for coord in stop_location]
    longitudes = [coord[1] for coord in stop_location]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    
    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
        return True
    else:
        return False
    

def is_point_at_home(row, most_common):
    target_latitude = most_common[0]
    target_longitude = most_common[1]
    point_latitude = row['Latitude']
    point_longitude = row['Longitude']
    distance = haversine((target_latitude, target_longitude), (point_latitude, point_longitude), unit = 'm')
    return distance <= 150  


def count_20_minutes(column_stop):
    empty_stop = np.empty_like(column_stop)
    first_true_flag = 1
    counter_true = 0

    ### Check if vehicle has stopped for more than 20 minutes
    for j in range(0, len(column_stop) - 1):

        ### Counts the number of true values starting at the specific index 
        if column_stop[j] == True:
            if first_true_flag == 1:
                current_index = j
                first_true_flag = 0
            elif first_true_flag == 0:
                counter_true = counter_true + 1

        ### If false, 20_min_stop is automatically false
        if column_stop[j] == False:
            #merged_data.loc[j, '20_Min_Stop'] = False
            empty_stop[j] = False
            counter_true = 0

        if column_stop[j] != column_stop[j + 1]:
            first_true_flag = 1
            if counter_true >= 20:
                for m in range(current_index, current_index + counter_true + 1):
                    #merged_data.loc[m, '20_Min_Stop'] = True
                    empty_stop[m] = True
                counter_true = 0
            elif counter_true < 20 and column_stop[j + 1] == False:
                for g in range(current_index, current_index + counter_true + 1):
                    #merged_data.loc[g, '20_Min_Stop'] = Fals
                    empty_stop[g] = False
                counter_true = 0            
    else:
        if counter_true >= 20:
            for m in range(current_index, current_index + counter_true + 2):
                #merged_data.loc[m, '20_Min_Stop'] = True
                empty_stop[m] = True

    return empty_stop


def parse_latitude_longitude(value):
    # Check if the string contains numerical characters or symbols representative of degrees, minutes, and seconds
    contains_numerical_characters = any(char.isdigit() for char in value)
    contains_degrees_symbols = any(char in value for char in ['°', "'", '"', '’', '?', '”'])
    
    if contains_numerical_characters and contains_degrees_symbols:
        # Regular expression to extract numbers
        pattern = r'(-?\d+)[°\?]?(\d+)?[\'\’]?(\d+)?["\”]?([NSWE]?)'
        
        # Search for pattern in the string
        match = re.search(pattern, value)
        
        if match:
            # Extract groups from the match
            degrees = float(match.group(1))
            minutes = float(match.group(2) or 0)
            seconds = float(match.group(3) or 0)
            direction = match.group(4)
            
            # Convert to decimal degrees
            decimal_degrees = degrees + minutes / 60 + seconds / 3600
            
            # Adjust for direction
            if direction in ['S']:
                decimal_degrees *= -1
            
            return decimal_degrees
        else:
            return None
    else:
        # Assume the input string is already in decimal format
        return float(value)





#########################################################################################
###################################### Data cleaning ####################################
#########################################################################################

### Create output folders to save everything to
os.makedirs(destination_folder, exist_ok=True)  # Create the destination folder if it doesn't exist


for filename in os.listdir(source_folder):

    if filename.endswith('.csv'):  # Make sure it's a CSV file

        filepath = os.path.join(source_folder, filename)

        # Extract vehicle number from filename
        vehicle_number = filename.split('_')[1]

        filepath_stop = os.path.join(source_folder_stop, f"stop_labels_Vehicle_{vehicle_number}.csv")

        if vehicle_number in vehicle_data_dict:
            vehicle_days = vehicle_data_dict[vehicle_number]
            
            vehicle_data = pd.read_csv(filepath)
            vehicle_data['date'] = pd.to_datetime(vehicle_data['date'])
            vehicle_data['day'] = vehicle_data['date'].dt.strftime('%d')

            stop_data = pd.read_csv(filepath_stop)
            stop_data['Time'] = pd.to_datetime(stop_data['Time'])
            stop_data['day'] = stop_data['Time'].dt.strftime('%d')
            
            for day in vehicle_days:
                vehicle_cleaned_data = pd.DataFrame()
                

                grouped_data = vehicle_data[vehicle_data['day'] == day].reset_index(drop=True)
                grouped_data_stop = stop_data[stop_data['day'] == day].reset_index(drop=True)

                if day in vehicle_date_change:
                    day_save = vehicle_date_change[day]
                else:
                    day_save = day
                
                if not grouped_data.empty:
                    print(f'Vehicle {vehicle_number} - Day {day_save}')
                    output_folder = os.path.join(destination_folder, f"Vehicle_{vehicle_number}", f"Vehicle_{vehicle_number}_{day_save}")
                    os.makedirs(output_folder, exist_ok=True)

                    ### Change to format below

                    vehicle_cleaned_data['Time_of_Day'] = grouped_data['date']
                    vehicle_cleaned_data['Latitude'] = grouped_data['lat']
                    vehicle_cleaned_data['Longitude'] = grouped_data['lon']
                    vehicle_cleaned_data['Distance'] = grouped_data['odometer']
                    vehicle_cleaned_data['Stop'] = grouped_data_stop['Stopped']

                    

                    # Remove equal sign and inverted commas from latitude and longitude columns
                    vehicle_cleaned_data['Latitude'] = vehicle_cleaned_data['Latitude'].str.replace('="', '').str.replace('"', '')
                    vehicle_cleaned_data['Longitude'] = vehicle_cleaned_data['Longitude'].str.replace('="', '').str.replace('"', '')

                    vehicle_cleaned_data['Latitude'] = vehicle_cleaned_data['Latitude'].apply(parse_latitude_longitude)
                    vehicle_cleaned_data['Longitude'] = vehicle_cleaned_data['Longitude'].apply(parse_latitude_longitude)


                    output_filename = os.path.join(output_folder, 'vehicle_cleaned_data.csv')
                    vehicle_cleaned_data.to_csv(output_filename, index=False)
                    

                    ### Create the minutely data

                    # Combine latitude and longitude
                    vehicle_cleaned_data['lat_lon'] = vehicle_cleaned_data['Latitude'].astype(str) + '_' + vehicle_cleaned_data['Longitude'].astype(str)
                    vehicle_cleaned_data['lat_lon'] = vehicle_cleaned_data['lat_lon'].astype(str)
                    
                    # Group by minute intervals
                    vehicle_cleaned_data['minute'] = vehicle_cleaned_data['Time_of_Day'].dt.strftime('%Y-%m-%d %H:%M')
                    vehicle_cleaned_data = vehicle_cleaned_data.groupby('minute').agg({
                        'lat_lon': 'last',
                        'Distance': 'last',
                        'Stop': lambda x: x.mode().iloc[0] 
                    }).reset_index()

                    vehicle_cleaned_data['Distance_diff'] = vehicle_cleaned_data['Distance'].diff()

                    # Set the first value to zero
                    vehicle_cleaned_data.loc[0, 'Distance_diff'] = 0

                    # If there are any missing values (NaN) due to the diff operation on the first row, fill them with 0
                    vehicle_cleaned_data['Distance_diff'] = vehicle_cleaned_data['Distance_diff'].fillna(0)
                    vehicle_cleaned_data['Distance_diff'] = vehicle_cleaned_data['Distance_diff']*1000
                    
                    # Split back into latitude and longitude
                    vehicle_cleaned_data[['Latitude', 'Longitude']] = vehicle_cleaned_data['lat_lon'].str.split('_', expand=True)
                    vehicle_cleaned_data.drop(columns=['lat_lon'], inplace=True)
                    vehicle_cleaned_data['Time_of_Day'] = vehicle_cleaned_data['minute']
                    vehicle_cleaned_data.drop(columns=['minute'], inplace=True)

                    vehicle_cleaned_data.drop(columns=['Distance'], inplace=True)
                    vehicle_cleaned_data['Distance'] = vehicle_cleaned_data['Distance_diff']
                    vehicle_cleaned_data.drop(columns=['Distance_diff'], inplace=True)


                    output_filename = os.path.join(output_folder, 'vehicle_cleaned_data_min.csv')
                    vehicle_cleaned_data.to_csv(output_filename, index=False)


                    ### Fill in the missing data

                    vehicle_cleaned_data['Time_of_Day'] = pd.to_datetime(vehicle_cleaned_data['Time_of_Day'])

                    # Fill in the missing data before the first row
                    start_time_1 = pd.to_datetime(f'2022-11-{day} 00:00:00', format='%Y-%m-%d %H:%M:%S')
                    end_time_1 = pd.to_datetime(vehicle_cleaned_data['Time_of_Day'].iloc[0]) - pd.Timedelta(minutes=1)


                    time_range_1 = pd.date_range(start=start_time_1, end=end_time_1, freq='min')

                    new_data_1 = pd.DataFrame({'Time_of_Day': time_range_1})
                    new_data_1['Distance'] = 0
                    new_data_1['Latitude'] = vehicle_cleaned_data['Latitude'].iloc[0]
                    new_data_1['Longitude'] = vehicle_cleaned_data['Longitude'].iloc[0]
                    new_data_1['Stop'] = True

                    # Fill in the missing data after the last row
                    start_time_2 = vehicle_cleaned_data['Time_of_Day'].iloc[-1] + pd.Timedelta(minutes=1)
                    end_time_2 = pd.to_datetime(f'2022-11-{day} 23:59:00', format='%Y-%m-%d %H:%M:%S')

                    time_range_2 = pd.date_range(start=start_time_2, end=end_time_2, freq='min')

                    new_data_2 = pd.DataFrame({'Time_of_Day': time_range_2})
                    new_data_2['Distance'] = 0
                    new_data_2['Latitude'] = vehicle_cleaned_data['Latitude'].iloc[-1]
                    new_data_2['Longitude'] = vehicle_cleaned_data['Longitude'].iloc[-1]
                    new_data_2['Stop'] = True

                    # Fill forward any of the missing data
                    vehicle_cleaned_data['Time_of_Day'] = pd.to_datetime(vehicle_cleaned_data['Time_of_Day'])

                    # Create a DateTimeIndex with all the minutes between the first and last time
                    full_index = pd.date_range(start=vehicle_cleaned_data['Time_of_Day'].min(), end=vehicle_cleaned_data['Time_of_Day'].max(), freq='min')

                    # Reindex the DataFrame with the full DateTimeIndex
                    vehicle_cleaned_data = vehicle_cleaned_data.set_index('Time_of_Day').reindex(full_index)

                    # Fill forward any missing data for 'Stop', 'Latitude', and 'Longitude' columns
                    vehicle_cleaned_data[['Stop', 'Latitude', 'Longitude']] = vehicle_cleaned_data[['Stop', 'Latitude', 'Longitude']].fillna(method='ffill')

                    # Fill the 'Distance' column with zeros
                    vehicle_cleaned_data['Distance'].fillna(0, inplace=True)

                    # Reset the index to include the 'Time_of_Day' column
                    vehicle_cleaned_data.reset_index(inplace=True)

                    # Assign the index values to the 'Time_of_Day' column
                    vehicle_cleaned_data['Time_of_Day'] = vehicle_cleaned_data['index']

                    # Drop the 'index' column
                    vehicle_cleaned_data.drop(columns=['index'], inplace=True)


                    # Merge the bottom, original, and top data
                    merged_data = pd.concat([new_data_1, vehicle_cleaned_data, new_data_2])
                    merged_data.reset_index(drop=True, inplace=True)

                    output_filename = os.path.join(output_folder, 'filled_vehicle_data.csv')
                    merged_data.to_csv(output_filename, index=False)




#################################################################################################
################ Reframe each vehicle to start from 04:00:00 an end at 03:59:59 #################
#################################################################################################


print('Reframing data points')

        
days = [str(num).zfill(2) for num in range(0, 32)]  # Days in the month


total_shift = 4 * 60

### Get the number of folders within the directory
### Change based on simulation
num_folders = count_folders_with_prefix(destination_folder, folder_prefix)

### Iterate through each vehicle in directory
for i in range(1, num_folders + 1):

    ### Get the number of days within that folder
    ### No need to change for different smulation
    new_folder = destination_folder + folder_prefix + str(i) + '/'
    new_folder_prefix = folder_prefix + str(i) + '_' # Vehicle_i_
    num_days = count_folders_with_prefix(new_folder, new_folder_prefix)

    day_num_array = get_last_two_values_as_strings(new_folder)

    ### Iterate through each day in vehicle
    for k in range(0, len(days)):

        ### Read the file from Inputs
        file_folder_1 = new_folder + new_folder_prefix + days[k] + '/' # Vehicle_i_k

        # If that day exists
        if os.path.exists(file_folder_1):
            print(f'{file_folder_1} exists: Reframing')

            # Create path to read file
            full_path = file_folder_1 + csv_save_name_1
            # File path exists, read the file
            day_1 = pd.read_csv(full_path)

            file_folder_2 = new_folder + new_folder_prefix + days[k + 1] + '/' # Vehicle_i_k+1

            # If day following starting day exists, use that dataset
            if os.path.exists(file_folder_2):
                # Create path to read file
                full_path = file_folder_2 + csv_save_name_1
                # File path exists, read the file
                day_2 = pd.read_csv(full_path)

                day_1['Time_of_Day'] = pd.to_datetime(day_1['Time_of_Day'])
                day_2['Time_of_Day'] = pd.to_datetime(day_2['Time_of_Day'])

                day_1_filtered = day_1[day_1['Time_of_Day'].dt.time >= pd.to_datetime('04:00:00').time()]
                day_2_filtered = day_2[day_2['Time_of_Day'].dt.time <= pd.to_datetime('03:59:00').time()]

            # just use ending points
            else:
                day_1['Time_of_Day'] = pd.to_datetime(day_1['Time_of_Day'])

                day_1_filtered = day_1[day_1['Time_of_Day'].dt.time >= pd.to_datetime('04:00:00').time()]

                first_date = day_1_filtered['Time_of_Day'].iloc[0].date()

                last_values = day_1_filtered.iloc[-1]
                day_2_filtered = pd.DataFrame([last_values] * total_shift)

                first_date += timedelta(days=1)
                

                # Add the time column to the filled DataFrame starting at the specified time
                day_2_filtered['Time_of_Day'] = pd.date_range(start = '00:00:00', periods = total_shift, freq = 'min')
                day_2_filtered['Time_of_Day'] = day_2_filtered['Time_of_Day'].apply(lambda x: x.replace(year=first_date.year, month=first_date.month, day=first_date.day))

            
            ### Create the new vehiclee day
            vehicle_day = pd.concat([day_1_filtered, day_2_filtered])

            vehicle_day['Time_of_Day'] = pd.to_datetime(vehicle_day['Time_of_Day'])

            home_charging_location = vehicle_day[(vehicle_day['Time_of_Day'].dt.time >= pd.to_datetime('20:00:00').time()) |
                 (vehicle_day['Time_of_Day'].dt.time <= pd.to_datetime('03:59:00').time())]

            most_common_combination = home_charging_location.groupby(['Latitude', 'Longitude']).size().idxmax()

            
            vehicle_day = vehicle_day.reset_index(drop=True)

            x = vehicle_day['Stop']
            ### Determine if vehicle is avaliable to charge
            # Determine 20 minute stop classification
            vehicle_day['20_Min_Stop'] = count_20_minutes(vehicle_day['Stop'])

            # Check to see if vehicle has stopped in specified location
            vehicle_day['Hub_Location'] = vehicle_day[['Latitude', 'Longitude']].apply(lambda x: is_point_in_stop(x), axis=1)

            # Create Available_Charging column - is the vehicle able to charge
            vehicle_day['Available_Charging'] = vehicle_day['Hub_Location'] & vehicle_day['20_Min_Stop']

            vehicle_day['HC_Location'] = vehicle_day[['Latitude', 'Longitude']].apply(lambda x: is_point_at_home(x, most_common_combination), axis=1)

            vehicle_day['Home_Charging'] = vehicle_day['HC_Location'] & vehicle_day['20_Min_Stop']

            vehicle_day['Energy_Consumption'] = vehicle_day['Distance'] * vehicle_efficiency

            save_path = file_folder_1 + csv_save_name_3

            print('Saving data')

            vehicle_day.to_csv(save_path, index=False)


        else:
            print(f'{file_folder_1} does not exist')

                    
        