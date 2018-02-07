import numpy as np
import os
import fnmatch
import json
import pickle
import pandas as pd
from pprint import pprint
from datetime import *

from tools.A_star_alibaba import extract_start_hours


def assignment_for_A_star_route_min(cf):
    """
    Give a diction of a day for
    'goal_city':
        'start_hour':
            {'num_steps', 'max_cost_sum', 'wind_cost_sum', 'rainfall_cost_sum'}

    We will assign the goal city starting point to a slot of 18*6
    which means to fill 1 to one of the 108 slots
    for hoursly slot, we also assign a priority from 1-6,
    :return:
    """

    cost_dict = collec_json_file_min(cf)
    assignment_dict = {}
    for day in cf.day_list:
        day_cost_dict = cost_dict[day]
        assignment_dict[day] = {}
        assignment_dict[day]['ave_max_cost'] = {}
        assignment_dict[day]['ratio_max_cost_to_manhattan'] = {}
        assignment_dict[day]['max_cost'] = {}
        assignment_dict[day]['num_steps'] = {}
        # # First we constract a cost matrix with dim: num_cities (10) * num_timeslots (18 *6)
        # day_assignment_matrix = np.ones(shape=(10, 18*6)) * np.inf
        for goal_city in cf.goal_city_list:
            day_goalcity_cost_dict = day_cost_dict[goal_city]
            sorted_ave_max_cost = sort_dicts(day_goalcity_cost_dict, 'ave_max_cost')
            sorted_ratio_to_manhattan = sort_dicts(day_goalcity_cost_dict, 'ratio_max_cost_to_manhattan')
            sorted_max_cost = sort_dicts(day_goalcity_cost_dict, 'max_cost')
            sorted_num_steps = sort_dicts(day_goalcity_cost_dict, 'num_steps')

            assignment_dict[day]['ave_max_cost'][goal_city] = sorted_ave_max_cost[:10]
            assignment_dict[day]['ratio_max_cost_to_manhattan'][goal_city] = sorted_ratio_to_manhattan[:10]
            assignment_dict[day]['max_cost'][goal_city] = sorted_max_cost[:10]
            assignment_dict[day]['num_steps'][goal_city] = sorted_num_steps[:10]

        # assignment_dict[day] = {}
        # # First we constract a cost matrix with dim: num_cities (10) * num_timeslots (18 *6)
        # cost_matrix = np.ones(shape=(10, 18)) * np.inf
        # step_fraction_matrix = np.ones(shape=(10, 18)) * np.inf
        # for goal_city in cf.goal_city_list:
        #     for start_hour in cost_dict[day][goal_city].keys():
        #         num_steps = cost_dict[day][goal_city][start_hour]['num_steps']
        #         max_cost = cost_dict[day][goal_city][start_hour]['max_cost']
        #
        #         cost_matrix[goal_city-1, start_hour-cf.hour_unique[0]] = max_cost
        #         # this step fraction will indicate the fraction of times remaining by whole hour
        #         # if two goal city need to start fly from the same hour, the higher this fraction this,
        #         # the more in advance we need to fly this ballon first
        #         step_fraction = np.mod(num_steps, cf.hourly_travel_distance)
        #         step_fraction_matrix[goal_city-1, start_hour-cf.hour_unique[0]] = step_fraction
        #
        # # We start assigining the slot from the furtherest city to the closest city
        # dist_manhattan = np.zeros(10)
        # city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
        # for goal_city in range(1, 11):
        #     start_hours, mins, dist_manhattan[goal_city-1] = extract_start_hours(cf, city_data_df, goal_city)
        #
        # # this will sort the distance from small to large
        # assignment_dict_day = {}
        # assignment_matrix = np.zeros(shape=(10, 18*6))
        # goal_city_priority = np.argsort(dist_manhattan)
        # assignment_dict_day, assignment_matrix, cities_cannot_reach = get_assignment(cf, goal_city_priority[::-1], cost_matrix, dist_manhattan, assignment_dict_day, assignment_matrix)
        # # Now we dot the unreachable cities:
        # assignment_dict_day, assignment_matrix, cities_cannot_reach = get_assignment(cf, cities_cannot_reach, cost_matrix, dist_manhattan, assignment_dict_day, assignment_matrix, cannot_reach_flag=True)
        #
        # print('day: %d' % day)
        # pprint(assignment_dict_day)
        # assignment_dict[day] = assignment_dict_day

    # pprint(assignment_dict)
    # Now we read the assignment matrix and combine the csv files:
    # write_combined_csv_file(cf, assignment_dict)

    route_selection_result = {}
    route_selection_result[(6, 1)] = (12, 50)
    route_selection_result[(6, 2)] = (16, 40)
    route_selection_result[(6, 3)] = (15, 40)
    route_selection_result[(6, 4)] = (14, 30)
    route_selection_result[(6, 5)] = (13, 30)
    route_selection_result[(6, 6)] = (4, 10)
    route_selection_result[(6, 7)] = (4, 0)
    route_selection_result[(6, 8)] = (5, 20)
    route_selection_result[(6, 9)] = (16, 0)
    route_selection_result[(6, 10)] = (15, 20)

    route_selection_result[(7, 1)] = (9, 50)
    route_selection_result[(7, 2)] = (17, 30)
    route_selection_result[(7, 3)] = (6, 40)
    route_selection_result[(7, 4)] = (3, 0)
    route_selection_result[(7, 5)] = (11, 10)
    route_selection_result[(7, 6)] = (9, 30)
    route_selection_result[(7, 7)] = (10, 10)
    route_selection_result[(7, 8)] = (9, 20)
    route_selection_result[(7, 9)] = (10, 50)
    route_selection_result[(7, 10)] = (16, 0)

    route_selection_result[(8, 1)] = (13, 0)
    route_selection_result[(8, 2)] = (16, 40)
    route_selection_result[(8, 3)] = (16, 0)
    route_selection_result[(8, 4)] = (14, 40)
    route_selection_result[(8, 5)] = (13, 30)
    route_selection_result[(8, 6)] = (9, 10)
    route_selection_result[(8, 7)] = (10, 20)
    route_selection_result[(8, 8)] = (3, 10)
    route_selection_result[(8, 9)] = (18, 40)
    route_selection_result[(8, 10)] = (16, 30)

    route_selection_result[(9, 1)] = (3, 0)
    route_selection_result[(9, 2)] = (5, 30)
    route_selection_result[(9, 3)] = (3, 10)
    route_selection_result[(9, 4)] = (9, 50)
    route_selection_result[(9, 5)] = (7, 50)
    route_selection_result[(9, 6)] = (6, 30)
    route_selection_result[(9, 7)] = (7, 40)
    route_selection_result[(9, 8)] = (8, 0)
    route_selection_result[(9, 9)] = (4, 0)
    route_selection_result[(9, 10)] = (3, 20)

    route_selection_result[(10, 1)] = (3, 20)
    route_selection_result[(10, 2)] = (11, 10)
    route_selection_result[(10, 3)] = (3, 10)
    route_selection_result[(10, 4)] = (3, 50)
    route_selection_result[(10, 5)] = (11, 40)
    route_selection_result[(10, 6)] = (3, 40)
    route_selection_result[(10, 7)] = (8, 30)
    route_selection_result[(10, 8)] = (3, 30)
    route_selection_result[(10, 9)] = (11, 50)
    route_selection_result[(10, 10)] = (11, 0)

    with open(cf.start_hour_min_filename, 'wb') as handle:
        pickle.dump(route_selection_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


    write_combined_csv_file_min(cf, route_selection_result)

    # return assignment_dict


def sort_dicts(day_goalcity_dict, sorted_key):
    items = day_goalcity_dict.items()
    lists = [[v[1][sorted_key], v[0], v[1]] for v in items]
    lists.sort()
    sorted_items = [(item[1], item[0]) for item in lists]

    return sorted_items


def get_assignment(cf, city_list, cost_matrix, dist_manhattan, assignment_dict, assignment_matrix, cannot_reach_flag=False):
    cities_cannot_reach = []
    for goal_city in city_list:
        goal_city_min_start_hour = np.argmin(cost_matrix[goal_city])
        if cost_matrix[goal_city][goal_city_min_start_hour] > dist_manhattan[goal_city] * cf.threshold_manhattan_distance\
                and not cannot_reach_flag:
            cities_cannot_reach.append(goal_city)
            print('Cannot reach the goal city: %d (manhattan dist: %d) with cost %2.f, we will consider its route later'
                  % (goal_city+1, int(dist_manhattan[goal_city]), cost_matrix[goal_city][goal_city_min_start_hour]))
        else:
            for s in range(6):
                if np.sum(assignment_matrix[:, goal_city_min_start_hour*6 + s]) >= 1:
                    print("Hour %d, slot %d has been occupied" % (goal_city_min_start_hour + cf.hour_unique[0], s))
                else:
                    # we assign this slot for this city
                    assignment_matrix[goal_city, goal_city_min_start_hour*6 + s] = 1
                    assignment_dict[goal_city] = {}
                    assignment_dict[goal_city]['hour'] = goal_city_min_start_hour + cf.hour_unique[0]
                    assignment_dict[goal_city]['slot'] = s
                    print('Assigning city %d, to hour: %d, slot: %d, with manhattan distance: %d and cost: %.2f.'
                          % (goal_city + 1, goal_city_min_start_hour + cf.hour_unique[0], s, dist_manhattan[goal_city],
                             cost_matrix[goal_city][goal_city_min_start_hour]))
                    break
            if s == 5:
                assert('All the slot has been occupied, choose next possible!')


    return assignment_dict, assignment_matrix, cities_cannot_reach


def collec_json_file_min(cf):
    """
    This script is to collect all cost from json files into a dict
    :param cf:
    :return:
    """
    cost_dict = {}
    for day in cf.day_list:
        cost_dict[day] = {}
        for goal_city in cf.goal_city_list:
            cost_dict[day][goal_city] = {}
            file_patterns = cf.file_patterns_min % (day, goal_city)
            files = fnmatch.filter(os.listdir(cf.cost_num_steps_dir_min), file_patterns)
            for f in files:
                _splits = f.split('_')
                start_hour = int(_splits[8])
                start_min = int(_splits[10].split('.')[0])
                data = json.load(open(os.path.join(cf.cost_num_steps_dir_min, f)))
                dict = {}
                dict['max_cost'] = data['max_cost']
                dict['ave_max_cost'] = data['ave_max_cost']
                dict['num_steps'] = data['num_steps']
                dict['ratio_max_cost_to_manhattan'] = data['ratio_max_cost_to_manhattan']
                cost_dict[day][goal_city][(start_hour, start_min)] = dict

    return cost_dict


def write_combined_csv_file(cf, assignment_dict):
    """
    This script is used to collect all the generated csv files (days, cities) to generate the required submission file
    :param cf:
    :return:
    """
    frames = []
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            start_hour = assignment_dict[day][goal_city-1]['hour']
            slot = assignment_dict[day][goal_city-1]['slot']
            csv_file_name_hour = cf.csv_patterns % (day, goal_city, start_hour) + '.csv'
            city_data_hour_df = pd.read_csv(os.path.join(cf.cost_num_steps_dir, csv_file_name_hour), index_col=None, names=['target', 'date', 'time', 'xid', 'yid'])
            sub_df = a_star_submission_3d(day, goal_city, city_data_hour_df, start_hour, slot)
            frames.append(sub_df)
    sub_csv = pd.concat(frames, axis=0)
    sub_csv.to_csv(os.path.join(cf.cost_num_steps_dir, cf.combined_csv_name), header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])


def write_combined_csv_file_min(cf, assignment_dict):
    """
    This script is used to collect all the generated csv files (days, cities) to generate the required submission file
    :param cf:
    :return:
    """
    frames = []
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            start_hour = assignment_dict[(day, goal_city)][0]
            start_min = assignment_dict[(day, goal_city)][1]
            csv_file_name_hour_min = cf.csv_patterns_min % (day, goal_city, start_hour, start_min) + '.csv'
            city_data_hour_df = pd.read_csv(os.path.join(cf.cost_num_steps_dir_min, csv_file_name_hour_min), index_col=None, names=['target', 'date', 'time', 'xid', 'yid'])
            sub_df = a_star_submission_3d(day, goal_city, city_data_hour_df, start_hour, start_min)
            frames.append(sub_df)
    sub_csv = pd.concat(frames, axis=0)
    sub_csv.to_csv(os.path.join(cf.cost_num_steps_dir_min, cf.combined_csv_name_min), header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])


def a_star_submission_3d(day, goal_city, city_data_hour_df, start_hour, start_min):
    #### create one submit path
    # A random time to get the time string right
    ti = datetime(2017, 11, 21, start_hour, start_min)
    row_list = []

    for ip in range(len(city_data_hour_df)):
        dict = {'target': goal_city,
                'date': day,
                'time': ti.strftime('%H:%M'),
                'xid': city_data_hour_df.iloc[ip]['xid'],
                'yid': city_data_hour_df.iloc[ip]['yid']}
        ti = ti + timedelta(minutes=2)
        row_list.append(dict)

    sub_df = pd.DataFrame(row_list)
    return sub_df