from garmin_fit_sdk import Decoder, Stream
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from get_map_image import get_map_image
import requests
import os


def average_all_2x2_squares(array_2d):
    n, m = array_2d.shape
    
    # Calculate the dimensions of the new array
    new_n = n - 1  # Number of starting rows for the 2x2 blocks
    new_m = m - 1  # Number of starting columns for the 2x2 blocks

    # Initialize the new array to hold the averages
    avg_array = np.zeros((new_n, new_m))

    for i in range(new_n):
        for j in range(new_m):
            # Calculate the average of the 2x2 square starting at (i, j)
            avg_array[i, j] = np.mean(array_2d[i:i+2, j:j+2])

    return avg_array

API_KEY = os.getenv("GARMIN_KEY")


# Parsing an existing file:
# -------------------------

stream = Stream.from_file("16570365455_ACTIVITY.fit")
decoder = Decoder(stream)
messages, errors = decoder.read()

#dict_keys(['file_id_mesgs', 'file_creator_mesgs', '288', '327', '326', 'event_mesgs', 'device_info_mesgs', '22', '141', 'device_settings_mesgs', 'user_profile_mesgs', '79', 'sport_mesgs', '13', 'zones_target_mesgs', 'record_mesgs', '233', 'gps_metadata_mesgs', '325', '104', 'lap_mesgs', 'time_in_zone_mesgs', '140', 'session_mesgs', 'activity_mesgs'])

print(type(messages.get('gps_metadata_mesgs')))
print(messages.get('gps_metadata_mesgs'))

recorded_messages =  messages.get('record_mesgs')

gps_divide = 11930465

n = len(recorded_messages)
lat_lon = np.zeros((n, 5))

initial_timestamp = recorded_messages[0]['timestamp']
# go in reverse order because latitude and longitude take a few frames to start recording
for idx, msg in enumerate(reversed(recorded_messages)):
    lat_lon[n - idx - 1, 0:2] = [(msg['timestamp'] - initial_timestamp).seconds, msg['heart_rate']]

    if "position_long" in msg and "position_lat" in msg:
        # [Timestamp, Heart Rate, Latitude, Longitude, Speed]
        lat_lon[n - idx -1, 2:] = [msg['position_lat']/gps_divide, msg['position_long']/gps_divide, msg['enhanced_speed']]
    else:
        # If not in the  data packet, just take the last values for longitude and latitude.
        lat_lon[n - idx -1, 2:] = lat_lon[n - idx, 2:]
    

"""
lat_lon = np.array(()).reshape((0,2))
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:

            lat_lon = np.append(lat_lon, [[point.latitude, point.longitude]], axis = 0)
"""           

x = lat_lon[:, 2]
y = lat_lon[:, 3]

slope, intercept = np.polyfit(y, x, 1)
#For a linear polynomial (so ,1), the formula for the line = slope*x+intercept (ax+b)

# Rotate to account for the fact that the field may be diagonal
theta = np.arctan2(slope, 1) * 180 / np.pi
new_x = x * np.cos(theta) - y * np.sin(theta) 
new_y = x * np.sin(theta) + y * np.cos(theta)

# Number of bins per axis.
a, b = 11*10, 7*10
grid = np.zeros((a, b))

x_lin = np.linspace(min(new_x), max(new_x), a)
x_grid = np.tile(x_lin, (b, 1))

y_lin = np.linspace(min(new_y), max(new_y), b).reshape(b, 1)
y_grid = np.tile(y_lin, (1, a))

indices = np.where(y_lin != 0)[0] 
f = lambda x: x >= x_lin
best_x = [int(np.where(f(x))[0][-1]) for x in new_x]
f = lambda y: y >= y_lin
best_y = [int(np.where(f(y))[0][-1]) for y in new_y]

index_1d = [int(np.ravel_multi_index((best_x[i], best_y[i]), (a, b))) for i in range(len(best_x[:]))]
unique, counts = np.unique(index_1d, return_counts=True)

bins = np.zeros((a*b,))
bins[unique] = counts
bins = bins.reshape((a, b))


# Continuosly shrink our grid by averaging sequentially over nearby squares. This also provides more of a analog heatmap quality.
for i in range(10):
    bins = average_all_2x2_squares(bins)

plt.figure()
# Create the heatmap
sns.heatmap(bins, cmap="viridis")
plt.axis('equal')
plt.show()

np.save("data", lat_lon)

plt.figure()
plt.plot(lat_lon[:,0]/60, lat_lon[:,1])
plt.show()


## my only idea right now is to use the Google Maps feature to put a rectangular polygon (full filled in red) and then image recognize to pick the points 
'''
ax = sns.heatmap(lat_lon, linewidth=0.5)
plt.show()
plt.imshow(lat_lon, extent =[x.min(), x.max(), y.min(), y.max()], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
'''

def something(x, y):
    # Plot the heatmap
    # url variable store url
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    
    # center defines the center of the map,
    # equidistant from all edges of the map. 
    center = str(round((min(x) + max(x))/2, 4)) + ',' + str(round((min(y) + max(y))/2, 4))
    
    # zoom defines the zoom
    # level of the map
    zoom = 18

    # get method of requests module
    # return response object
    r = requests.get(url + "center=" + center + "&zoom=" +
                    str(zoom) + "&size=400x400&key=" +
                                API_KEY)

    # wb mode is stand for write binary mode
    f = open('location.png', 'wb')
    
    # r.content gives content,
    # in this case gives image
    f.write(r.content)

    # close method of file object
    # save and close the file
    f.close()