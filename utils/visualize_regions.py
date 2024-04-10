from matplotlib import pyplot as plt
import geopandas as gpd

color_mapping = {
    0: 'grey',
    1: 'yellow',
    2: 'orange',
    3: 'cornflowerblue',
    4: 'orangered',
    5: 'thistle',
    6: 'mediumorchid',
    7: 'forestgreen',
    8: 'lightgreen',
}

FUNC_TYPES = {
    0: 'Unassigned',
    1: 'Residential',
    2: 'School',
    3: 'Hospital',
    4: 'Business',
    5: 'Office',
    6: 'Recreation',
    7: 'Park',
    8: 'OpenSpace',
}

shp_file = "E:\\UP\\data\\MADRID\\MADRID.shp"
gdf = gpd.read_file(shp_file)
gdf['type'] = gdf['id']

# beijing:60, chicago:79, madrid:75

for i in range(len(gdf['type'])):
    if gdf['type'][i] > 1 and i <= 75:
        gdf['type'][i] = 0
cmap = plt.get_cmap('viridis', len(gdf['id'].unique()))
fig, ax = plt.subplots(figsize=(12, 8))
gdf['color'] = gdf['type'].map(color_mapping)
gdf.plot(ax=ax, color=gdf['color'], legend=True)

handles = []
labels = []
for key, value in FUNC_TYPES.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[key], markersize=10))
    labels.append(value)

plt.legend(handles, labels, title='Functional Types', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Masked Madrid')

