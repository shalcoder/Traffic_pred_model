
# STEP 3: SPATIAL ANALYSIS PREPARATION
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Create GeoDataFrame
df = pd.read_csv('clean_traffic_accidents_dataset.csv')
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Save as shapefile for QGIS
gdf.to_file('traffic_accidents.shp')
print("Shapefile created: traffic_accidents.shp")

# For QGIS Hotspot Analysis:
# 1. Open QGIS
# 2. Layer → Add Layer → Add Vector Layer → Select traffic_accidents.shp
# 3. Processing → Toolbox → Search "Heatmap"
# 4. Double-click "Heatmap (Kernel Density Estimation)"
# 5. Set radius to 1000 meters, output raster size based on your area
# 6. Run to generate hotspot map

# Alternative: Python hotspot analysis
from sklearn.cluster import DBSCAN

# Extract coordinates
coords = df[['Latitude', 'Longitude']].values

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.01, min_samples=10)
df['Cluster'] = dbscan.fit_predict(coords)

# Count accidents per cluster
cluster_summary = df.groupby('Cluster').agg({
    'Accident_ID': 'count',
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Fatalities': 'sum',
    'Injuries': 'sum'
}).round(6)

print("\nAccident Clusters (Hotspots):")
print(cluster_summary)
