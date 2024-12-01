from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import folium
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.colors as colors


def train_model_spatial_clustering():
    df = pd.read_csv('./geospatial_data.csv', nrows=100)

    # Sample data
    data = df[['latitude', 'longitude', 'air_quality_index', 'traffic_density']].values

    # Fit DBSCAN
    dbscan = DBSCAN(eps=0.01, min_samples=5)
    labels = dbscan.fit_predict(data)
    df['cluster'] = labels  # Add cluster labels to dataframe

    # Initialize a geographic plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add a simple map background
    ax.stock_img()

    # Set map extent to match the range of your data
    ax.set_extent([
        df['longitude'].min() - 1, df['longitude'].max() + 1,  # Add padding for better visualization
        df['latitude'].min() - 1, df['latitude'].max() + 1
    ], crs=ccrs.PlateCarree())

    # Define a colormap for clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
    colormap = cm.get_cmap('viridis', num_clusters + 1)  # Cluster colors
    color_list = [colors.rgb2hex(colormap(i)) for i in range(colormap.N)]

    print(df.head())

    # Plot the clusters
    for cluster_id in set(labels):
        cluster_data = df[df['cluster'] == cluster_id]
        color = 'black' if cluster_id == -1 else color_list[cluster_id]  # Noise = black
        ax.scatter(
            cluster_data['longitude'],
            cluster_data['latitude'],
            c=color,
            s=20,
            label=f"Cluster {cluster_id}" if cluster_id != -1 else "Noise",
            transform=ccrs.PlateCarree()
        )

    # Add titles and legends
    ax.set_title("Spatial Clustering on Map")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
    plt.show()



