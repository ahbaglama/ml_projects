

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("train.csv")

# Dropping unnecessary columns 

df = df.drop(['PdDistrict', 'Address', 'Resolution', 'Descript', 'DayOfWeek'], axis = 1) # axis = 1 for column drop, 0 for row drop..


df.isnull().sum()

f = lambda x: (x["Dates"].split())[0] 
df["Dates"] = df.apply(f, axis=1)

f = lambda x: (x["Dates"].split('-'))[0] 
df["Dates"] = df.apply(f, axis=1)

df_2014 = df[(df.Dates == '2014')]
df_2014.head()


scaler = MinMaxScaler()

# Y is latitude and X is longitude

scaler.fit(df_2014[['X']])
df_2014['X_scaled'] = scaler.transform(df_2014[['X']]) 

scaler.fit(df_2014[['Y']])
df_2014['Y_scaled'] = scaler.transform(df_2014[['Y']])


k_range = range(1,15)

list_dist = []

for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df_2014[['X_scaled','Y_scaled']])
    list_dist.append(model.inertia_)


from matplotlib import pyplot as plt

plt.xlabel('K')
plt.ylabel('Distortion value (inertia)')
plt.plot(k_range,list_dist)
plt.show()

model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(df_2014[['X_scaled','Y_scaled']])
y_predicted


df_2014['cluster'] = y_predicted
df_2014


import plotly.express as px   

figure = px.scatter_mapbox(df_2014, lat='Y', lon='X',                       
                       center = dict(lat = 37.8, lon = -122.4), # This is the coordinate of San Francisco..
                       zoom = 9,                                # Zoom of the map
                       opacity = .9,                           # opacity of the map a value between 0 and 1..
                       mapbox_style = 'stamen-terrain',       # basemap 
                       color = 'cluster',                      # Map will draw scatter colors according to cluster number..
                       title = 'San Francisco Crime Districts',
                       width = 1100,
                       height = 700,                     
                       hover_data = ['cluster', 'Category', 'Y', 'X']
                       )

figure.show()



import plotly
plotly.offline.plot(figure, filename = 'maptest.html', auto_open = True)

help(px.scatter_mapbox)

