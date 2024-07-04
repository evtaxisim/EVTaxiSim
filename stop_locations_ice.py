################################################## READ ME ##################################################
"""
    This programme finds the stop locations and creates a dataframe with the first column being the location 
    in gps co-ordinates of the stop and the second being the time interval that it was stopped for. This is 
    then plotted on a map to visualise where the taxis stopped and for how long. Furthermore, this is split 
    for the GoMetro data and the MixTelematics data, as well as being split even further for home location 
    and general waiting location.

    What the stop locations are grouped according to. This is merely for the map and csv files would need to 
    be saved if that is what is required.
    - All stop locations
    - GM stop locations
    - MT stop locations
    - Home stop locations
    - Waiting stop locations
    - Home + MT stop locations
    - Home + GM stop locations
    - Waiting + MT stop locations
    - Waiting + GM stop locations

    These stop locations are then filtered through k-means clustering to reduce the number of points.

    /// OLD CODE ///
    This programme finds the stop locations of the taxis given 20, 10 and 5 minute stop intervals. It then
    plots the data on a map to visualise the stop locations to determine charging locations in
    available_charging.py. This information is also used to determine the trip information of the taxis for
    the scheduling problem

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
from sklearn.cluster import KMeans

source_folder = 'D:/Masters/Simulations/Simulation_4/Usable_Data/'
save_folder = 'D:/Masters/Simulations/Simulation_4/Trip_Data/'

### Suburban Taxi Ranks
### Box coordinates - Stellenbosch Taxi Rank
stop_location_taxi_SL = [
    (-33.932359, 18.857750),  
    (-33.932359, 18.859046),       
    (-33.933172, 18.859046),      
    (-33.933172, 18.857750)       
]


### Functions
def is_point_in_stop(point, location):
    lat, lon = point
    latitudes = [coord[0] for coord in location]
    longitudes = [coord[1] for coord in location]
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

def calculate_center_coordinates(coordinates):
    num_coordinates = len(coordinates)

    if num_coordinates == 0:
        raise ValueError("The list of coordinates is empty.")

    sum_latitude = sum(lat for lat, lon in coordinates)
    sum_longitude = sum(lon for lat, lon in coordinates)

    center_latitude = sum_latitude / num_coordinates
    center_longitude = sum_longitude / num_coordinates

    return center_latitude, center_longitude

def map_letter_to_color(letter):
    color_mapping = {
        'S': 'blue',
        'E': 'green',
        'H': 'red',
        # Add more letters and corresponding colors as needed
    }
    return color_mapping.get(letter.upper(), 'blue')


total_files = len([file for root, dirs, files in os.walk(source_folder) for file in files if file == 'vehicle_day_min.csv']) + 17

final_stop = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])
final_stop_gm = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])
final_stop_mt = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])

final_stop_home = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])
final_stop_not_home = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])

final_stop_home_gm = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])
final_stop_home_mt = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])
final_stop_not_home_gm = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])
final_stop_not_home_mt = pd.DataFrame(columns=['Latitude', 'Longitude', 'Time', 'Home'])

final_stop_20 = pd.DataFrame(columns=['Latitude', 'Longitude'])
all_home_locations = pd.DataFrame(columns=['Latitude', 'Longitude'])



for root, dirs, files in tqdm(os.walk(source_folder), total=total_files, desc='Processing Files'):

    
    for file in files:

        if file == 'vehicle_day_min.csv':


            file_path = os.path.join(root, file)

            # Read the CSV file using pandas and append it to the csv_data list
            vehicle_day = pd.read_csv(file_path)

            vehicle_day['lat_lon'] = vehicle_day['Latitude'].astype(str) + ',' + vehicle_day['Longitude'].astype(str)


            # Identify blocks of True values in 'boolean_column'
            blocks_of_stopped = (vehicle_day['Stop'] != vehicle_day['Stop'].shift(1)).cumsum()

            block_lengths = vehicle_day.groupby(blocks_of_stopped)['Latitude'].count().reset_index()
            block_lengths.columns = ['Block', 'Length']

            true_blocks = vehicle_day[vehicle_day['Stop']].groupby(blocks_of_stopped[vehicle_day['Stop']])['lat_lon'].agg(lambda x: x.mode().iloc[0])

            true_blocks = true_blocks.to_frame(name='lat_lon')
            true_blocks['Block'] = true_blocks.index

            # Merge true_blocks with block_lengths based on the 'Block' column
            true_blocks = pd.merge(true_blocks, block_lengths, on='Block', how='left')

            # Reset the index of true_blocks
            true_blocks.reset_index(drop=True, inplace=True)

            true_blocks.drop('Block', axis=1, inplace=True)

            # Split the 'lat_lon' column into 'Latitude' and 'Longitude'
            true_blocks[['Latitude', 'Longitude']] = true_blocks['lat_lon'].str.split(',', expand=True)
            true_blocks.drop('lat_lon', axis=1, inplace=True)

            # Change the name of the 'Length' column to 'Time'
            true_blocks.rename(columns={'Length': 'Time'}, inplace=True)

            # Convert the values in the 'Time' column to minutes
            true_blocks['Time'] /= 60

            true_blocks['Latitude'] = true_blocks['Latitude'].astype(float)
            true_blocks['Longitude'] = true_blocks['Longitude'].astype(float)

            ### Find the home location
            vehicle_day['Time_of_Day'] = pd.to_datetime(vehicle_day['Time_of_Day'])

            home_charging_location = vehicle_day[(vehicle_day['Time_of_Day'].dt.hour >= 20) |
                                     (vehicle_day['Time_of_Day'].dt.hour <= 3)]
            
            most_common_combination = home_charging_location.groupby(['Latitude', 'Longitude']).size().idxmax()

            true_blocks['Home'] = true_blocks[['Latitude', 'Longitude']].apply(lambda x: is_point_at_home(x, most_common_combination), axis=1)

            """
            true_blocks

            Time        |   Latitude    |   Longitude   |   Home
            ----------------------------------------------------------
            duration    |   (float)     |   (float)     |   (boolean)
            (float)
            """

            # TODO continue with this but not nescessary at the moment
            ### split the waiting time according to when the vehicle is generally driving and when the vehicle is not operational
            # taxi is operational during the hours 06h00 and 09h00, so these stops are picking up passengers
            taxi_morning_split_operation = vehicle_day[(vehicle_day['Time_of_Day'].dt.hour >= 6) & (vehicle_day['Time_of_Day'].dt.hour < 9)]
            
            # taxi is operational during the hours h00 and 09h00, so these stops are picking up passengers
            taxi_afternoon_split_operation = vehicle_day[(vehicle_day['Time_of_Day'].dt.hour >= 15) & (vehicle_day['Time_of_Day'].dt.hour < 20)]



            ### Create the 20-min stop locations
            blocks_20 = (vehicle_day['20_Min_Stop'] != vehicle_day['20_Min_Stop'].shift(1)).cumsum()
            true_blocks_20 = vehicle_day[vehicle_day['20_Min_Stop']].groupby(blocks_20[vehicle_day['20_Min_Stop']])['lat_lon'].agg(lambda x: x.mode().iloc[0]).reset_index(drop=True)
            new_true_blocks_20 = true_blocks_20.str.split(',', expand=True)
            new_true_blocks_20.columns = ['Latitude', 'Longitude']

            new_true_blocks_20['Latitude'] = new_true_blocks_20['Latitude'].astype(float)
            new_true_blocks_20['Longitude'] = new_true_blocks_20['Longitude'].astype(float)

            final_stop_20 = final_stop_20.append(new_true_blocks_20, ignore_index=True)

            ### Append home location cordinates
            all_home_locations = all_home_locations.append({'Latitude': most_common_combination[0], 'Longitude': most_common_combination[1]}, ignore_index=True)


            ### Append for a larger dataset
            final_stop = final_stop.append(true_blocks, ignore_index=True)

            for index, row in true_blocks.iterrows():
                if row['Home']:
                    final_stop_home = final_stop_home.append(row, ignore_index=True)
                else:
                    final_stop_not_home = final_stop_not_home.append(row, ignore_index=True)


            
            ### Determine if this data is for GoMetro or Mix and save accordingly
            match = re.search(r'Vehicle_(\d{1,2})_(\d+)', root)
            vehicle_number = int(match.group(1))

            # GoMetro data
            if vehicle_number <= 17:
                final_stop_gm = final_stop_gm.append(true_blocks, ignore_index=True)

                for index, row in true_blocks.iterrows():
                    if row['Home']:
                        final_stop_home_gm = final_stop_home_gm.append(row, ignore_index=True)
                    else:
                        final_stop_not_home_gm = final_stop_not_home_gm.append(row, ignore_index=True)

            # MixTelematics data    
            else:
                final_stop_mt = final_stop_mt.append(true_blocks, ignore_index=True)

                for index, row in true_blocks.iterrows():
                    if row['Home']:
                        final_stop_home_mt = final_stop_home_mt.append(row, ignore_index=True)
                    else:
                        final_stop_not_home_mt = final_stop_not_home_mt.append(row, ignore_index=True)

print("Processing completed.")
print("Saving Data")

final_stop.to_csv(os.path.join(save_folder, 'all_stop_locations.csv'), index=False)



### Plot points on map to visualise
# All stop points
latitudes = final_stop['Latitude']
longitudes = final_stop['Longitude']
time_legend = final_stop['Time']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
heat_data = list(zip(latitudes, longitudes, time_legend))
HeatMap(heat_data).add_to(map_object)
map_object.save(os.path.join(save_folder, 'all_stop_locations.html'))

# GM stop points
latitudes = final_stop_gm['Latitude']
longitudes = final_stop_gm['Longitude']
time_legend = final_stop_gm['Time']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
heat_data = list(zip(latitudes, longitudes, time_legend))
HeatMap(heat_data).add_to(map_object)
map_object.save(os.path.join(save_folder, 'gm_stop_locations.html'))


# Home stop points
latitudes = final_stop_home['Latitude']
longitudes = final_stop_home['Longitude']
time_legend = final_stop_home['Time']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
heat_data = list(zip(latitudes, longitudes, time_legend))
HeatMap(heat_data).add_to(map_object)
map_object.save(os.path.join(save_folder, 'home_stop_locations.html'))

# Not home stop points
latitudes = final_stop_not_home['Latitude']
longitudes = final_stop_not_home['Longitude']
time_legend = final_stop_not_home['Time']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
heat_data = list(zip(latitudes, longitudes, time_legend))
HeatMap(heat_data).add_to(map_object)
map_object.save(os.path.join(save_folder, 'waiting_stop_locations.html'))

# Home gm stop points
latitudes = final_stop_home_gm['Latitude']
longitudes = final_stop_home_gm['Longitude']
time_legend = final_stop_home_gm['Time']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
heat_data = list(zip(latitudes, longitudes, time_legend))
HeatMap(heat_data).add_to(map_object)
map_object.save(os.path.join(save_folder, 'home_gm_stop_locations.html'))


# Not home gm stop points
latitudes = final_stop_not_home_gm['Latitude']
longitudes = final_stop_not_home_gm['Longitude']
time_legend = final_stop_not_home_gm['Time']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
heat_data = list(zip(latitudes, longitudes, time_legend))
HeatMap(heat_data).add_to(map_object)
map_object.save(os.path.join(save_folder, 'waiting_gm_stop_locations.html'))


### Save stop locations for 20-min and home
final_stop_20.to_csv(os.path.join(save_folder, 'stop_locations_20.csv'), index=False)
all_home_locations.to_csv(os.path.join(save_folder, 'home_stop_locations.csv'), index=False)



####################################################################################################
############################### Create clustered coordinates #######################################
####################################################################################################


### For home locations
print("Clustering for home locations")
file_path = os.path.join(save_folder, 'home_stop_locations.csv')
home_locations = pd.read_csv(file_path)

data = home_locations[['Latitude', 'Longitude']].values

num_clusters = 10

desired_radius = 150

while True:
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    home_locations['cluster'] = kmeans.fit_predict(data)

    # Get the cluster centers
    center_points = kmeans.cluster_centers_

    # Calculate the maximum distance from any point to its assigned cluster center
    max_distance = max(
        haversine((lat, lon), (center_points[cluster][0], center_points[cluster][1]), unit = 'm')
        for lat, lon, cluster in zip(home_locations['Latitude'], home_locations['Longitude'], home_locations['cluster'])
    )

    print(f"Testing {num_clusters} clusters - Max distance of {max_distance}")


    # Check if the maximum distance is less than or equal to the desired radius
    if max_distance <= desired_radius:
        break

    # Increment the number of clusters for the next iteration
    num_clusters += 1

print("Optimal number of clusters:", num_clusters)

home_center_points = pd.DataFrame(data=center_points, columns=['Latitude', 'Longitude'])

latitudes = home_locations['Latitude']
longitudes = home_locations['Longitude']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)

# Plot the original points in blue
for index, row in home_locations.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Point {index}",
        icon=folium.Icon(color='blue')
    ).add_to(map_object)

# Plot the center points in red
for center in center_points:
    folium.Marker(
        location=[center[0], center[1]],
        popup="Cluster Center",
        icon=folium.Icon(color='red')
    ).add_to(map_object)

map_object.save(os.path.join(save_folder, 'home_clusters.html'))

### For 20_min stop locations
print("Clustering for 20 min stop locations")
file_path = os.path.join(save_folder, 'stop_locations_20.csv')
stop_locations = pd.read_csv(file_path)

data = stop_locations[['Latitude', 'Longitude']].values

num_clusters = 10

desired_radius = 150

while True:
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    stop_locations['cluster'] = kmeans.fit_predict(data)

    # Get the cluster centers
    center_points = kmeans.cluster_centers_

    # Calculate the maximum distance from any point to its assigned cluster center
    max_distance = max(
        haversine((lat, lon), (center_points[cluster][0], center_points[cluster][1]), unit = 'm')
        for lat, lon, cluster in zip(stop_locations['Latitude'], stop_locations['Longitude'], stop_locations['cluster'])
    )

    print(f"Testing {num_clusters} clusters - Max distance of {max_distance}")

    # Check if the maximum distance is less than or equal to the desired radius
    if max_distance <= desired_radius:
        break

    # Increment the number of clusters for the next iteration
    num_clusters += 1


stop_center_points = pd.DataFrame(data=center_points, columns=['Latitude', 'Longitude'])

print("Optimal number of clusters:", num_clusters)

latitudes = stop_locations['Latitude']
longitudes = stop_locations['Longitude']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)

# Plot the original points in blue
for index, row in stop_locations.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Point {index}",
        icon=folium.Icon(color='blue')
    ).add_to(map_object)

# Plot the center points in red
for center in center_points:
    folium.Marker(
        location=[center[0], center[1]],
        popup="Cluster Center",
        icon=folium.Icon(color='red')
    ).add_to(map_object)

map_object.save(os.path.join(save_folder, 'stop_20_clusters.html'))


home_locations.to_csv(os.path.join(save_folder, 'clustered_home_locations.csv'), index=False)
stop_locations.to_csv(os.path.join(save_folder, 'clustered_stop_locations.csv'), index=False)

print(home_center_points)
print(stop_center_points)


####################################################################################################
################################### Create joint locations #########################################
####################################################################################################

filtered_locations = pd.DataFrame(columns=['Latitude', 'Longitude', 'Name'])
filter_distance = 150 #m

is_home = False
is_rank = False

stop_num = 1

for index, row in stop_center_points.iterrows():

    for index_home, row_home in home_center_points.iterrows():
        test_distance = haversine((row['Latitude'], row['Longitude']), (row_home['Latitude'], row_home['Longitude']), unit = 'm')

        if test_distance <= filter_distance:
            print(f'Point {index} is a Home Location: {test_distance}')
            is_home = True
            break

    # check if the point is in the location
    if is_point_in_stop((row['Latitude'], row['Longitude']), stop_location_taxi_SL):
        is_rank = True
        print(f'Point {index} is a E1 Location')


    # only add the values if one of these conditions have not been met       
    if is_home == False and is_rank == False:
        print(f'Point {index} is a S{stop_num} Location')
        filtered_locations = filtered_locations.append(
                {'Latitude': row['Latitude'], 'Longitude': row['Longitude'], 'Name': f'S{stop_num}'},
                ignore_index=True
            )
    
        stop_num = stop_num + 1

    is_home = False
    is_rank = False


# Get center co-ordinates for the energy locatio i.e. the taxi rank
center_coordinate_SL_lat, center_coordinate_SL_lon = calculate_center_coordinates(stop_location_taxi_SL)


# add home locations to fitered locations
for index_home, row_home in home_center_points.iterrows():
    filtered_locations = filtered_locations.append(
                {'Latitude': row_home['Latitude'], 'Longitude': row_home['Longitude'], 'Name': f'H{index_home + 1}'},
                ignore_index=True
            )


# add energy locations
filtered_locations = filtered_locations.append(
                {'Latitude': center_coordinate_SL_lat, 'Longitude': center_coordinate_SL_lon, 'Name': 'E1'},
                ignore_index=True
            )


# Save information
filtered_locations.to_csv(os.path.join(save_folder, 'filtered_locations.csv'), index=False)

latitudes = filtered_locations['Latitude']
longitudes = filtered_locations['Longitude']

map_object = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=10, min_zoom=2, max_zoom=20)
# Plot the original points in blue
for index, row in filtered_locations.iterrows():
    first_letter = row['Name'][0] if row['Name'] else ''  # Get the first letter of the Name
    marker_color = map_letter_to_color(first_letter)

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Name'],
        icon=folium.Icon(color=marker_color)
    ).add_to(map_object)

map_object.save(os.path.join(save_folder, 'filtered_locations.html'))

print(filtered_locations)






