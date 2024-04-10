import csv
import math
from math import radians

import geopandas as gpd
import pandas as pd
from pyproj import Geod


def center_geolocation(geolocations):
    x = 0  # lon
    y = 0  # lat
    z = 0
    lenth = len(geolocations)
    for lon, lat in geolocations:
        lon = radians(float(lon))  # Convert angle x from degrees to radians
        lat = radians(float(lat))
        x += math.cos(lat) * math.cos(lon)
        y += math.cos(lat) * math.sin(lon)
        z += math.sin(lat)
        x = float(x / lenth)
        y = float(y / lenth)
        z = float(z / lenth)
    return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))


def find_middle_area(numbers):
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    split_index = len(sorted_numbers) // 2
    middle = sorted_numbers[split_index]
    return middle


def feat_trans(raw_data):
    """
    raw data:[land_id, type_id, [POLYGON[...]]]
    processed data:[type_id, lon, lat, area, perimeter, compactness] : N * 6
    """
    type_id = []
    lon = []
    lat = []
    area = []
    perimeter = []
    compactness = []
    cons = []

    for row in raw_data.itertuples():
        # type_id
        type_id.append(row.id)

        # 求每个polygon的中心坐标点
        polygon_points = row.geometry.exterior.coords
        polygon_points = center_geolocation([[x, y] for x, y in polygon_points])
        lon.append(polygon_points[0])
        lat.append(polygon_points[1])

        # area
        geod = Geod(ellps="WGS84")
        area.append(- geod.geometry_area_perimeter(row.geometry)[0])

        # perimeter
        perimeter.append(geod.geometry_area_perimeter(row.geometry)[1])

        # compactness
        compactness.append(4 * math.pi * area[-1] / perimeter[-1] ** 2)

        # print(one_hot_type_id, lon, lat, area, perimeter, compactness)

    middle_area = find_middle_area(area)
    for i in range(len(type_id)):
        if area[i] < middle_area:
            cons.append('small')
        else:
            cons.append('large')
    dataframe = pd.DataFrame(
        {'type_id': type_id, 'lon': lon, 'lat': lat, 'area': area, 'perimeter': perimeter, 'compactness': compactness, 'cons': cons})
    return dataframe


def load_features(csv_path):
    csv_reader = csv.reader(open(csv_path, encoding='utf-8'))
    features_list = []
    for line in csv_reader:
        if line[0] == '':
            pass
        else:
            features = [int(line[0])]
            for x in line[1:6]:
                features.append(float(x))
            features.append(line[6])
            features_list.append(features)
    return features_list


def region_type_mask(dataframe):
    for i in range(len(dataframe['type_id'])):
        if dataframe['type_id'][i] > 1 and i <=75:
            dataframe['type_id'][i] = 0
    return dataframe
