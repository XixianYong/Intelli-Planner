import csv
import math

import numpy as np
from geopy.distance import geodesic
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from .urban_config import *


class UrbanEnv:
    """Urban environment for LLM-powered RL"""
    FAILURE_REWARD = -1
    INTERMEDIATE_REWARD = 0

    def __init__(self, geo_info, reward_agents):
        self.state = None
        self.geo_info = geo_info
        self.init_state = self.get_init_state(self.geo_info)
        self.transition_dict = dict()
        self.resident_agent = reward_agents['resident']
        self.government_agent = reward_agents['government']
        self.developer_agent = reward_agents['developer']

    def get_init_state(self, geo_info):
        """
        get init state
        :param geo_info: list N * [type, lon, lat, area, perimeter, compactness, size]
        :return: init_state: list [type1, type2, ..., typeN]
        """
        init_state = []
        for i in range(len(geo_info)):
            init_state.append(geo_info[i][TYPE_ID])
        return init_state

    def reset(self):
        """
        reset the environment
        :return: state = init_state
        """
        self.state = self.init_state.copy()
        return self.state

    def get_action_state(self):
        current_region_idx = self.state.index(0)

        # current region features: [type, lon, lat, area, perimeter, compactness, size]
        region_geo_info = self.geo_info[current_region_idx]
        if region_geo_info[CONS] == 'small':
            region_geo_info[CONS] = 0.
        else:
            region_geo_info[CONS] = 1.

        # living circle features: [re_ratio, bu_ratio, of_ratio, re_ration, pa_ratio, op_ratio, ho_ratio, sc_ratio,
        # re_num, bu_num, of_num, re_num, pa_num, op_num, sh_num, lh_num, ss_num, ls_num]
        living_circle_geo_info_dict = self.get_living_circle_type_ratio(current_region_idx)
        living_circle_geo_info = []
        for key in living_circle_geo_info_dict.keys():
            living_circle_geo_info.append(float(living_circle_geo_info_dict[key]))

        # global features: [re_ratio, bu_ratio, of_ratio, re_ration, pa_ratio, op_ratio, ho_ratio, sc_ratio,
        # re_num, bu_num, of_num, re_num, pa_num, op_num, sh_num, lh_num, ss_num, ls_num]
        global_geo_info_dict = self.get_type_ratio()
        global_geo_info = []
        for key in global_geo_info_dict.keys():
            if key != 'total_area':
                global_geo_info.append(float(global_geo_info_dict[key]))

        # target features: [bu_ratio, of_ratio, re_ration, pa_ratio, op_ratio, sh_num, lh_num, ss_num, ls_num]
        target_geo_info = [BUSINESS_COVERAGE_DEMANDS, OFFICE_COVERAGE_DEMANDS, RECREATION_COVERAGE_DEMANDS,
                           GREEN_COVERAGE_DEMANDS, OPEN_SPACE_COVERAGE_DEMANDS,
                           CLINIC_NUM, HOSPITAL_NUM, KINDERGARTEN_NUM, PRIMARY_SCHOOL_NUM + MIDDLE_SCHOOL_NUM]

        # action state: [region_geo_info, living_circle_geo_info, global_geo_info]
        action_state = region_geo_info + living_circle_geo_info + global_geo_info + target_geo_info
        return action_state

    def get_reward(self):
        """
        get reward
        :return: service_reward, ecology_reward, economic_reward, total_reward
        """
        global_region_info = self.get_type_ratio()

        # Living Service reward
        service_reward = 0
        residential_num = 0

        for region_idx in range(len(self.state)):
            if region_idx < REGION_SPLIT_IDX:
                if self.get_type_name(region_idx) == 'Residential':
                    residential_num += 1
                    living_circle_neighborhood_idx = self.get_living_circle_idx(region_idx)

                    school_exist = False
                    hospital_exist = False
                    business_exist = False
                    office_exist = False
                    recreation_exist = False

                    for neighbor in living_circle_neighborhood_idx:
                        neighbor_name = self.get_type_name(neighbor)
                        if neighbor_name == 'School':
                            school_exist = True
                        elif neighbor_name == 'Hospital':
                            hospital_exist = True
                        elif neighbor_name == 'Business':
                            business_exist = True
                        elif neighbor_name == 'Office':
                            office_exist = True
                        elif neighbor_name == 'Recreation':
                            recreation_exist = True
                        else:
                            pass
                    service_reward += (school_exist + hospital_exist + business_exist + office_exist + recreation_exist) / 5
        if residential_num == 0:
            service_reward = 0
        else:
            service_reward = service_reward / residential_num

        if global_region_info['Park'] < GREEN_COVERAGE_DEMANDS or global_region_info['OpenSpace'] < OPEN_SPACE_COVERAGE_DEMANDS:
            ecology_reward = (min(global_region_info['Park'] / GREEN_COVERAGE_DEMANDS, 1) +
                              min(global_region_info['OpenSpace'] / OPEN_SPACE_COVERAGE_DEMANDS, 1)) / 2
        else:
            ecology_reward = 1

        # Economic reward
        valid_business_area = 0
        valid_office_area = 0
        valid_recreation_area = 0
        total_area = global_region_info['total_area']
        for region_idx in range(len(self.state)):
            if region_idx < REGION_SPLIT_IDX:
                if self.get_type_name(region_idx) == 'Business' or self.get_type_name(
                        region_idx) == 'Office' or self.get_type_name(region_idx) == 'Recreation':
                    living_circle_neighborhood_idx = self.get_living_circle_idx(region_idx)
                    for neighbor in living_circle_neighborhood_idx:
                        if self.get_type_name(neighbor) == 'Residential':
                            if self.get_type_name(region_idx) == 'Business':
                                valid_business_area += self.geo_info[region_idx][AREA]
                            elif self.get_type_name(region_idx) == 'Office':
                                valid_office_area += self.geo_info[region_idx][AREA]
                            elif self.get_type_name(region_idx) == 'Recreation':
                                valid_recreation_area += self.geo_info[region_idx][AREA]
                            break
        if (valid_business_area < total_area * BUSINESS_COVERAGE_DEMANDS or
                valid_office_area < total_area * OFFICE_COVERAGE_DEMANDS or
                valid_recreation_area < total_area * RECREATION_COVERAGE_DEMANDS):
            economic_reward = (min(valid_business_area / (total_area * BUSINESS_COVERAGE_DEMANDS), 1) +
                               min(valid_office_area / (total_area * OFFICE_COVERAGE_DEMANDS), 1) +
                               min(valid_recreation_area / (total_area * RECREATION_COVERAGE_DEMANDS), 1)) / 3
        else:
            economic_reward = 1

        # Equity reward
        school_distance_list = self.get_nearest_type_distance_list('School')
        hospital_distance_list = self.get_nearest_type_distance_list('Hospital')
        small_hospital_num = global_region_info['small_hospital_num']
        large_hospital_num = global_region_info['large_hospital_num']
        small_school_num = global_region_info['small_school_num']
        large_school_num = global_region_info['large_school_num']

        num_reward = (pow(math.e, - abs(small_hospital_num - CLINIC_NUM) / 2)
                      + pow(math.e, - abs(large_hospital_num - HOSPITAL_NUM) / 2)
                      + pow(math.e, - abs(small_school_num - KINDERGARTEN_NUM) / 2)
                      + pow(math.e, - abs(large_school_num - MIDDLE_SCHOOL_NUM - PRIMARY_SCHOOL_NUM) / 2)) / 4

        # Gini-coefficient based equity reward
        # school_equity_reward = 1 - gini_coef(school_distance_list)
        # hospital_equity_reward = 1 - gini_coef(hospital_distance_list)

        # Variance-based equity reward
        school_equity_reward = equity_cal(school_distance_list)
        hospital_equity_reward = equity_cal(hospital_distance_list)

        equity_reward = ((school_equity_reward + hospital_equity_reward) / 2 + num_reward) / 2

        # Satisfaction reward
        planning_targets = ("1. Planning Targets: For coverage percentage, "
                            + "Business area:" + str(BUSINESS_COVERAGE_DEMANDS) + ', '
                            + "Office area:" + str(OFFICE_COVERAGE_DEMANDS) + ', '
                            + "Recreation area:" + str(RECREATION_COVERAGE_DEMANDS) + ', '
                            + "Park area:" + str(GREEN_COVERAGE_DEMANDS) + ', '
                            + "OpenSpace area:" + str(OPEN_SPACE_COVERAGE_DEMANDS)
                            + '. For number of facilities, '
                            + "Hospital number:" + str(HOSPITAL_NUM) + ', '
                            + "Clinic number:" + str(CLINIC_NUM) + ', '
                            + "Kindergarten number:" + str(KINDERGARTEN_NUM) + ', '
                            + "Primary School number:" + str(PRIMARY_SCHOOL_NUM) + ', '
                            + "Middle School number:" + str(MIDDLE_SCHOOL_NUM) + '.')
        detailed_information = ("2. Actual detailed Information: For coverage percentage, "
                                + "Business area:" + str(global_region_info['Business']) + ', '
                                + "Office area:" + str(global_region_info['Office']) + ', '
                                + "Recreation area:" + str(global_region_info['Recreation']) + ', '
                                + "Park area:" + str(global_region_info['Park']) + ', '
                                + "OpenSpace area:" + str(global_region_info['OpenSpace'])
                                + '. For number of facilities, '
                                + "Hospital number:" + str(global_region_info['large_hospital_num']) + ', '
                                + "Clinic number:" + str(global_region_info['small_hospital_num']) + ', '
                                + "Kindergarten number:" + str(global_region_info['small_school_num']) + ', '
                                + "Primary School number:" + str(global_region_info['large_school_num'] / 2) + ', '
                                + "Middle School number:" + str(global_region_info['large_school_num'] / 2) + '.')
        requirements = ("Just answer me one word from  'poor', 'average', 'good', 'very good' without anthing else.")
        human_message = HumanMessage(content="Planning Scheme:\n"
                                             + planning_targets + "\n"
                                             + detailed_information + "\n"
                                             + requirements)

        satisfaction_reward = self.get_satisfaction_reward(human_message)

        total_reward = service_reward + ecology_reward + economic_reward + equity_reward + satisfaction_reward

        reward_info = {
            'service_reward': service_reward,
            'ecology_reward': ecology_reward,
            'economic_reward': economic_reward,
            'equity_reward': equity_reward,
            'satisfaction_reward': satisfaction_reward,
            'total_reward': total_reward
        }
        return reward_info

    def get_satisfaction_reward(self, human_message):
        resident_answer = self.resident_agent.send(human_message)
        government_answer = self.government_agent.send(human_message)
        developer_answer = self.developer_agent.send(human_message)

        print('resident_answer: ', resident_answer)
        print('government_answer: ', government_answer)
        print('developer_answer: ', developer_answer)

        if resident_answer == 'poor' or resident_answer == 'Poor':
            resident_satisfaction_reward = 0.25
        elif resident_answer == 'average' or resident_answer == 'Average':
            resident_satisfaction_reward = 0.5
        elif resident_answer == 'good' or resident_answer == 'Good':
            resident_satisfaction_reward = 0.75
        elif resident_answer == 'very good' or resident_answer == 'Very good':
            resident_satisfaction_reward = 1
        else:
            resident_satisfaction_reward = 0

        if government_answer == 'poor' or government_answer == 'Poor':
            government_satisfaction_reward = 0.25
        elif government_answer == 'average' or government_answer == 'Average':
            government_satisfaction_reward = 0.5
        elif government_answer == 'good' or government_answer == 'Good':
            government_satisfaction_reward = 0.75
        elif government_answer == 'very good' or government_answer == 'Very good':
            government_satisfaction_reward = 1
        else:
            government_satisfaction_reward = 0

        if developer_answer == 'poor' or developer_answer == 'Poor':
            developer_satisfaction_reward = 0.25
        elif developer_answer == 'average' or developer_answer == 'Average':
            developer_satisfaction_reward = 0.5
        elif developer_answer == 'good' or developer_answer == 'Good':
            developer_satisfaction_reward = 0.75
        elif developer_answer == 'very good' or developer_answer == 'Very good':
            developer_satisfaction_reward = 1
        else:
            developer_satisfaction_reward = 0

        satisfaction_reward = (resident_satisfaction_reward
                               + government_satisfaction_reward
                               + developer_satisfaction_reward) / 3
        print('satisfaction_reward: ', satisfaction_reward)
        return satisfaction_reward

    def is_done(self):
        """
        check if the environment is done,
        :return: done
        """
        done = True
        for idx in range(len(self.state)):
            if self.get_type_name(idx) == 'Unassigned':
                done = False
                break
        return done

    def step(self, action):
        """
        update state
        :param coord: int
        :param action: int
        :return: next_state, reward, done
        """
        critic_state = self.get_critic_state()
        coord = self.state.index(0)
        self.state[coord] = action + 1
        next_state = self.state
        next_critic_state = self.get_critic_state()

        done = self.is_done()
        if done:
            reward_info = self.get_reward()
        else:
            reward_info = {
                'service_reward': 0,
                'ecology_reward': 0,
                'economic_reward': 0,
                'equity_reward': 0,
                'satisfaction_reward': 0,
                'total_reward': self.INTERMEDIATE_REWARD
            }

        return next_state, reward_info, done, critic_state, next_critic_state

    def get_critic_state(self):
        region_info = self.get_type_ratio()
        critic_state = []
        critic_state.append(float(region_info['Residential']))
        critic_state.append(float(region_info['Business']))
        critic_state.append(float(region_info['Office']))
        critic_state.append(float(region_info['Recreation']))
        critic_state.append(float(region_info['Park']))
        critic_state.append(float(region_info['OpenSpace']))
        critic_state.append(float(region_info['Hospital']))
        critic_state.append(float(region_info['School']))
        critic_state.append(float(region_info['Residential_num']))
        critic_state.append(float(region_info['Business_num']))
        critic_state.append(float(region_info['Office_num']))
        critic_state.append(float(region_info['Recreation_num']))
        critic_state.append(float(region_info['Park_num']))
        critic_state.append(float(region_info['OpenSpace_num']))
        critic_state.append(float(region_info['small_hospital_num']))
        critic_state.append(float(region_info['large_hospital_num']))
        critic_state.append(float(region_info['small_school_num']))
        critic_state.append(float(region_info['large_school_num']))
        return critic_state

    def get_type_name(self, region_idx):
        """
        get type name of the region
        :param region_idx: int
        :return: type_name: str
        """
        type_id = self.state[region_idx]
        type_name = FUNC_TYPES[type_id]
        return type_name

    def get_distance(self, region_idx1, region_idx2):
        """
        get distance between two regions
        :param region_idx1: int
        :param region_idx2: int
        :return: distance: float
        """
        loc1_lon = self.geo_info[region_idx1][LON]
        loc1_lat = self.geo_info[region_idx1][LAT]
        loc2_lon = self.geo_info[region_idx2][LON]
        loc2_lat = self.geo_info[region_idx2][LAT]

        distance = geodesic((loc1_lat, loc1_lon), (loc2_lat, loc2_lon)).km * 1000  # 单位m
        return distance

    def get_living_circle_idx(self, region_idx):
        """
        get the regions in the living circle
        :param region_idx: int
        :return: living_circle_idx: list
        """
        living_circle_radius = LIFE_CIRCLE_SIZE
        living_circle_idx = []
        for i in range(len(self.state)):
            if i == region_idx:
                living_circle_idx.append(i)
            else:
                distance = self.get_distance(region_idx, i)
                if distance <= living_circle_radius:
                    living_circle_idx.append(i)
        return living_circle_idx

    def get_living_circle_type_ratio(self, idx):
        living_circle_idx = self.get_living_circle_idx(idx)
        living_circle_type_ratio = dict()
        residential_area = 0
        business_area = 0
        office_area = 0
        park_area = 0
        hospital_area = 0
        school_area = 0
        recreation_area = 0
        open_space_area = 0
        total_area = 0

        residential_num = 0
        business_num = 0
        office_num = 0
        recreation_num = 0
        park_num = 0
        open_space_num = 0
        small_hospital_num = 0
        large_hospital_num = 0
        small_school_num = 0
        large_school_num = 0

        for region_idx in living_circle_idx:
            type_name = self.get_type_name(region_idx)
            if type_name == 'Residential':
                residential_area += self.geo_info[region_idx][AREA]
                residential_num += 1
            elif type_name == 'Business':
                business_area += self.geo_info[region_idx][AREA]
                business_num += 1
            elif type_name == 'Office':
                office_area += self.geo_info[region_idx][AREA]
                office_num += 1
            elif type_name == 'Park':
                park_area += self.geo_info[region_idx][AREA]
                park_num += 1
            elif type_name == 'Hospital':
                hospital_area += self.geo_info[region_idx][AREA]
                if self.geo_info[region_idx][CONS] == 'small':
                    small_hospital_num += 1
                else:
                    large_hospital_num += 1
            elif type_name == 'School':
                school_area += self.geo_info[region_idx][AREA]
                if self.geo_info[region_idx][CONS] == 'small':
                    small_school_num += 1
                else:
                    large_school_num += 1
            elif type_name == 'Recreation':
                recreation_area += self.geo_info[region_idx][AREA]
                recreation_num += 1
            elif type_name == 'OpenSpace':
                open_space_area += self.geo_info[region_idx][AREA]
                open_space_num += 1
            total_area += self.geo_info[region_idx][AREA]

        living_circle_type_ratio['Residential'] = round(residential_area / total_area, 4)
        living_circle_type_ratio['Business'] = round(business_area / total_area, 4)
        living_circle_type_ratio['Office'] = round(office_area / total_area, 4)
        living_circle_type_ratio['Recreation'] = round(recreation_area / total_area, 4)
        living_circle_type_ratio['Park'] = round(park_area / total_area, 4)
        living_circle_type_ratio['OpenSpace'] = round(open_space_area / total_area, 4)
        living_circle_type_ratio['Hospital'] = round(hospital_area / total_area, 4)
        living_circle_type_ratio['School'] = round(school_area / total_area, 4)

        living_circle_type_ratio['Residential_num'] = residential_num
        living_circle_type_ratio['Business_num'] = business_num
        living_circle_type_ratio['Office_num'] = office_num
        living_circle_type_ratio['Recreation_num'] = recreation_num
        living_circle_type_ratio['Park_num'] = park_num
        living_circle_type_ratio['OpenSpace_num'] = open_space_num
        living_circle_type_ratio['small_hospital_num'] = small_hospital_num
        living_circle_type_ratio['large_hospital_num'] = large_hospital_num
        living_circle_type_ratio['small_school_num'] = small_school_num
        living_circle_type_ratio['large_school_num'] = large_school_num

        return living_circle_type_ratio

    def get_type_ratio(self):
        """
        get the ratio of each type
        :return: type_ratio: dict()
        """
        type_ratio = dict()
        residential_area = 0
        business_area = 0
        office_area = 0
        park_area = 0
        hospital_area = 0
        school_area = 0
        recreation_area = 0
        open_space_area = 0
        total_area = 0

        residential_num = 0
        business_num = 0
        office_num = 0
        recreation_num = 0
        park_num = 0
        open_space_num = 0
        small_hospital_num = 0
        large_hospital_num = 0
        small_school_num = 0
        large_school_num = 0

        for region_idx in range(len(self.state)):
            if region_idx < REGION_SPLIT_IDX:
                type_name = self.get_type_name(region_idx)
                if type_name == 'Residential':
                    residential_area += self.geo_info[region_idx][AREA]
                    residential_num += 1
                elif type_name == 'Business':
                    business_area += self.geo_info[region_idx][AREA]
                    business_num += 1
                elif type_name == 'Office':
                    office_area += self.geo_info[region_idx][AREA]
                    office_num += 1
                elif type_name == 'Park':
                    park_area += self.geo_info[region_idx][AREA]
                    park_num += 1
                elif type_name == 'Hospital':
                    hospital_area += self.geo_info[region_idx][AREA]
                    if self.geo_info[region_idx][CONS] == 'small':
                        small_hospital_num += 1
                    else:
                        large_hospital_num += 1
                elif type_name == 'School':
                    school_area += self.geo_info[region_idx][AREA]
                    if self.geo_info[region_idx][CONS] == 'small':
                        small_school_num += 1
                    else:
                        large_school_num += 1
                elif type_name == 'Recreation':
                    recreation_area += self.geo_info[region_idx][AREA]
                    recreation_num += 1
                elif type_name == 'OpenSpace':
                    open_space_area += self.geo_info[region_idx][AREA]
                    open_space_num += 1
                total_area += self.geo_info[region_idx][AREA]
        type_ratio['Residential'] = round(residential_area / total_area, 4)
        type_ratio['Business'] = round(business_area / total_area, 4)
        type_ratio['Office'] = round(office_area / total_area, 4)
        type_ratio['Recreation'] = round(recreation_area / total_area, 4)
        type_ratio['Park'] = round(park_area / total_area, 4)
        type_ratio['OpenSpace'] = round(open_space_area / total_area, 4)
        type_ratio['Hospital'] = round(hospital_area / total_area, 4)
        type_ratio['School'] = round(school_area / total_area, 4)

        type_ratio['Residential_num'] = residential_num
        type_ratio['Business_num'] = business_num
        type_ratio['Office_num'] = office_num
        type_ratio['Recreation_num'] = recreation_num
        type_ratio['Park_num'] = park_num
        type_ratio['OpenSpace_num'] = open_space_num
        type_ratio['small_hospital_num'] = small_hospital_num
        type_ratio['large_hospital_num'] = large_hospital_num
        type_ratio['small_school_num'] = small_school_num
        type_ratio['large_school_num'] = large_school_num

        type_ratio['total_area'] = total_area

        return type_ratio

    def get_nearest_type_distance_list(self, region_type):
        """
        get the nearest distance of a certain region type to residential area
        :param region_type: str
        :return: nearest_distance: list [float, float, ...]
        """
        nearest_distance = []
        for region_idx in range(len(self.state)):
            if region_idx < REGION_SPLIT_IDX:
                if self.get_type_name(region_idx) == 'Residential':
                    distance = self.get_nearest_distance(region_idx, region_type)
                    if distance == 100000:
                        pass
                    else:
                        nearest_distance.append(distance)
        return nearest_distance

    def get_nearest_distance(self, region_idx, region_type):
        nearest_distance = 100000
        for i in range(len(self.state)):
            if self.get_type_name(i) == region_type:
                distance = self.get_distance(region_idx, i)
                if distance < nearest_distance:
                    nearest_distance = distance
        return nearest_distance

    def get_living_circle_info(self, region_idx):
        live_circle_idx = self.get_living_circle_idx(region_idx)
        live_circle_info = []
        for idx in live_circle_idx:
            live_circle_info.append(self.get_type_name(idx))
        return set(live_circle_info)

    def get_global_info(self):
        global_info = self.get_type_ratio()
        return global_info

    def random_action(self):
        """
        get random action
        :return: action
        """
        for idx in range(len(self.state)):
            if self.get_type_name(idx) == 'Unassigned':
                act = np.random.randint(1, len(FUNC_TYPES))
                self.state[idx] = act

        return self.get_reward()


def gini_coef(wealths):
    if len(wealths) == 0:
        return 1.0
    else:
        # 归一化处理
        # min_value = min(wealths)
        # max_value = max(wealths)
        # wealths = (np.array(wealths) - min_value) / (max_value - min_value)
        # 计算gini系数
        cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
        sum_wealths = cum_wealths[-1]
        xarray = np.array(range(0, len(cum_wealths))) / float(len(cum_wealths) - 1)
        yarray = cum_wealths / sum_wealths
        B = np.trapz(yarray, x=xarray)
        A = 0.5 - B
        return A / (A + B)


def equity_cal(distances):
    if len(distances) == 0:
        return 0
    else:
        max_distance = max(distances)
        min_distance = min(distances)
        s = ((max_distance - min_distance) / 800) ** 3
        equity = math.e ** (-s)
        return equity


class ActionAgent:
    def __init__(
            self,
            model: ChatOpenAI,
            system_message: SystemMessage,
    ) -> None:
        self.model = model
        self.system_message = system_message

    def send(self, human_message) -> str:
        prompt = [self.system_message, human_message]
        # print(prompt)
        message = self.model(prompt).content  #
        return message


def create_reward_agent(model, prompt):
    resident_system_message = SystemMessage(content="MISSION:\n" + prompt["resident"])
    government_system_message = SystemMessage(content="MISSION:\n" + prompt["government"])
    developer_system_message = SystemMessage(content="MISSION:\n" + prompt["developer"])

    resident = ActionAgent(model, resident_system_message)
    government = ActionAgent(model, government_system_message)
    developer = ActionAgent(model, developer_system_message)
    
    reward_agents = dict()
    reward_agents['resident'] = resident
    reward_agents['government'] = government
    reward_agents['developer'] = developer

    return reward_agents


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

