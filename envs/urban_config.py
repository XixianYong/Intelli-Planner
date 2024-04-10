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

# feature position [0]TYPE_ID, [1]LON, [2]LAT, [3]AREA, [4]PERIMETER, [5]COMPACTNESS, [6]CONS
TYPE_ID = 0
LON = TYPE_ID + 1
LAT = TYPE_ID + 2
AREA = TYPE_ID + 3
PERIMETER = TYPE_ID + 4
COMPACTNESS = TYPE_ID + 5
CONS = TYPE_ID + 6

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

LIFE_CIRCLE_SIZE = 500

GREEN_COVERAGE_DEMANDS = 0.15
OPEN_SPACE_COVERAGE_DEMANDS = 0.1
BUSINESS_COVERAGE_DEMANDS = 0.05
OFFICE_COVERAGE_DEMANDS = 0.05
RECREATION_COVERAGE_DEMANDS = 0.05
HOSPITAL_NUM = 2
CLINIC_NUM = 2
KINDERGARTEN_NUM = 1
PRIMARY_SCHOOL_NUM = 3
MIDDLE_SCHOOL_NUM = 2
REGION_SPLIT_IDX = 60
