import numpy as np
import os
import fnmatch
import json
import pandas as pd
import pickle
from pprint import pprint
from datetime import *

from tools.A_star_alibaba import extract_start_hours
from tools.Assignment_for_A_star_route_min import collec_json_file_min


def assignment_for_A_star_route(cf):
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

    cost_dict = collec_json_file(cf)
    assignment_dict = {}
    for day in cf.day_list:
        print('############day: %d#############' % day)
        assignment_dict[day] = {}
        # First we constract a cost matrix with dim: num_cities (10) * num_timeslots (18 *6)
        cost_matrix = np.ones(shape=(10, 18)) * np.inf
        step_fraction_matrix = np.ones(shape=(10, 18)) * np.inf
        for goal_city in cf.goal_city_list:
            for start_hour in cost_dict[day][goal_city].keys():
                num_steps = cost_dict[day][goal_city][start_hour]['num_steps']
                max_cost = cost_dict[day][goal_city][start_hour]['max_cost']

                cost_matrix[goal_city-1, start_hour-cf.hour_unique[0]] = max_cost
                # this step fraction will indicate the fraction of times remaining by whole hour
                # if two goal city need to start fly from the same hour, the higher this fraction this,
                # the more in advance we need to fly this ballon first
                step_fraction = np.mod(num_steps, cf.hourly_travel_distance)
                step_fraction_matrix[goal_city-1, start_hour-cf.hour_unique[0]] = step_fraction

        # We start assigining the slot from the furtherest city to the closest city
        dist_manhattan = np.zeros(10)
        city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
        for goal_city in range(1, 11):
            _, _, dist_manhattan[goal_city-1] = extract_start_hours(cf, city_data_df, goal_city)

        # this will sort the distance from small to large
        assignment_dict_day = {}
        assignment_matrix = np.zeros(shape=(10, 18*6))
        goal_city_priority = np.argsort(dist_manhattan)
        assignment_dict_day, assignment_matrix, cities_cannot_reach = get_assignment(cf, goal_city_priority[::-1], cost_matrix, dist_manhattan, assignment_dict_day, assignment_matrix)
        # Now we dot the unreachable cities:
        assignment_dict_day, assignment_matrix, cities_cannot_reach = get_assignment(cf, cities_cannot_reach, cost_matrix, dist_manhattan, assignment_dict_day, assignment_matrix, cannot_reach_flag=True)

        pprint(assignment_dict_day)
        assignment_dict[day] = assignment_dict_day

    pprint(assignment_dict)
    # Now we read the assignment matrix and combine the csv files:
    write_combined_csv_file(cf, assignment_dict)


def assignment_for_A_star_route_10min(cf):
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
    assignment_dict_all = {}
    for day in cf.day_list:
        print('############day: %d#############' % day)
        assignment_dict[day] = {}
        # First we constract a cost matrix with dim: num_cities (10) * num_timeslots (18 *6)
        cost_matrix = np.ones(shape=(10, 18*6)) * np.inf
        for goal_city in cf.goal_city_list:
            for (start_hour, start_min) in cost_dict[day][goal_city].keys():
                max_cost = cost_dict[day][goal_city][(start_hour, start_min)]['max_cost']
                cost_matrix[goal_city-1, (start_hour-cf.hour_unique[0]) * 6 + int(start_min/10)] = max_cost

        # We start assigining the slot from the furtherest city to the closest city
        dist_manhattan = np.zeros(10)
        city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
        for goal_city in range(1, 11):
            _, _, dist_manhattan[goal_city-1] = extract_start_hours(cf, city_data_df, goal_city)

        # this will sort the distance from small to large
        assignment_dict_day = {}
        assignment_matrix = np.zeros(shape=(10, 18*6))
        goal_city_priority = np.argsort(dist_manhattan)
        assignment_dict_day, assignment_matrix, cities_cannot_reach = get_assignment_min(cf, cost_dict[day], goal_city_priority[::-1], cost_matrix, dist_manhattan, assignment_dict_day, assignment_matrix)
        # Now we dot the unreachable cities:
        assignment_dict_day, assignment_matrix, cities_cannot_reach = get_assignment_min(cf, cost_dict[day], cities_cannot_reach, cost_matrix, dist_manhattan, assignment_dict_day, assignment_matrix, cannot_reach_flag=True)

        pprint(assignment_dict_day)
        assignment_dict[day] = assignment_dict_day
        # This is used for WZN to generate A star 3D
        for goal_city in cf.goal_city_list:
            assignment_dict_all[(day, goal_city)] = (assignment_dict_day[goal_city - 1]['hour'], assignment_dict_day[goal_city - 1]['min'])

    with open(os.path.join(cf.cost_num_steps_dir, 'assignment_dict_all.pickle'), 'wb') as fp:
        pickle.dump(assignment_dict_all, fp)


    pprint(assignment_dict)
    # Now we read the assignment matrix and combine the csv files:
    write_combined_csv_file_min(cf, assignment_dict)


def get_assignment(cf, city_list, cost_matrix, dist_manhattan, assignment_dict, assignment_matrix, cannot_reach_flag=False):
    cities_cannot_reach = []
    for goal_city in city_list:
        flag_assigned = False
        goal_city_min_start_hour_index = np.argsort(cost_matrix[goal_city])
        index = 0
        goal_city_min_start_hour = goal_city_min_start_hour_index[index]
        if cost_matrix[goal_city][goal_city_min_start_hour] > dist_manhattan[goal_city] * cf.threshold_manhattan_distance\
                and not cannot_reach_flag:
            cities_cannot_reach.append(goal_city)
            print('Cannot reach the goal city: %d (manhattan dist: %d) with cost %2.f, we will consider its route later'
                  % (goal_city+1, int(dist_manhattan[goal_city]), cost_matrix[goal_city][goal_city_min_start_hour]))
        else:
            while not flag_assigned:
                for s in range(6):
                    if np.sum(assignment_matrix[:, goal_city_min_start_hour*6 + s]) >= 1:
                        print("Hour %d, slot %d has been occupied" % (goal_city_min_start_hour + cf.hour_unique[0], s))
                    else:
                        # we assign this slot for this city
                        flag_assigned = True
                        assignment_matrix[goal_city, goal_city_min_start_hour*6 + s] = 1
                        assignment_dict[goal_city] = {}
                        assignment_dict[goal_city]['hour'] = goal_city_min_start_hour + cf.hour_unique[0]
                        assignment_dict[goal_city]['slot'] = s
                        print('Assigning city %d, to hour: %d, slot: %d, with manhattan distance: %d and cost: %.2f.'
                              % (goal_city + 1, goal_city_min_start_hour + cf.hour_unique[0], s, dist_manhattan[goal_city],
                                 cost_matrix[goal_city][goal_city_min_start_hour]))
                        break
                if s == 5:
                    print('Hour %d, slot is full, choose next minimum.' % goal_city_min_start_hour)
                    index += 1
                    goal_city_min_start_hour = goal_city_min_start_hour_index[index]

    return assignment_dict, assignment_matrix, cities_cannot_reach


def get_assignment_min(cf, cost_dict, city_list, cost_matrix, dist_manhattan, assignment_dict, assignment_matrix, cannot_reach_flag=False):
    cities_cannot_reach = []
    for goal_city in city_list:
        goal_city_min_start_hour_index = np.argsort(cost_matrix[goal_city])
        index = 0
        goal_city_min_start_hour = goal_city_min_start_hour_index[index]
        if cost_matrix[goal_city][goal_city_min_start_hour] > dist_manhattan[goal_city] * cf.threshold_manhattan_distance\
                and not cannot_reach_flag:
            cities_cannot_reach.append(goal_city)
            print('Cannot reach the goal city: %d (manhattan dist: %d) with cost %2.f, we will consider its route later'
                  % (goal_city+1, int(dist_manhattan[goal_city]), cost_matrix[goal_city][goal_city_min_start_hour]))
        else:
            flag_assigned = False
            while not flag_assigned:
                hour = int(goal_city_min_start_hour//6) + cf.hour_unique[0]
                min = int(np.mod(goal_city_min_start_hour, 6)) * 10
                if np.sum(assignment_matrix[:, goal_city_min_start_hour]) == 1:
                    index += 1
                    goal_city_min_start_hour = goal_city_min_start_hour_index[index]
                    print("Hour %d, Min %d has been occupied, choose next slot" % (hour, min))
                else:
                    # we assign this slot for this city
                    flag_assigned = True
                    assignment_matrix[goal_city, goal_city_min_start_hour] = 1
                    assignment_dict[goal_city] = {}
                    assignment_dict[goal_city]['hour'] = hour
                    assignment_dict[goal_city]['min'] = min
                    assignment_dict[goal_city]['num_steps'] = cost_dict[goal_city+1][(hour, min)]['num_steps']
                    print('Assigning city %d, to hour: %d, min: %d, with manhattan distance: %d, num_steps: %d, and cost: %.2f.'
                          % (goal_city + 1, hour, min, dist_manhattan[goal_city], assignment_dict[goal_city]['num_steps'], cost_matrix[goal_city][goal_city_min_start_hour]))

    return assignment_dict, assignment_matrix, cities_cannot_reach


def collec_json_file(cf):
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
            file_patterns = cf.file_patterns % (day, goal_city)
            files = fnmatch.filter(os.listdir(cf.cost_num_steps_dir), file_patterns)
            for f in files:
                start_hour = int(f[len(file_patterns)-6:-5])
                cost_dict[day][goal_city][start_hour] = {}
                data = json.load(open(os.path.join(cf.cost_num_steps_dir, f)))
                cost_dict[day][goal_city][start_hour]['max_cost'] = data['max_cost']
                cost_dict[day][goal_city][start_hour]['num_steps'] = data['num_steps']
                cost_dict[day][goal_city][start_hour]['rainfall_cost'] = data['rainfall_cost']
                cost_dict[day][goal_city][start_hour]['wind_cost'] = data['wind_cost']

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
            start_hour = assignment_dict[day][goal_city-1]['hour']
            start_min = assignment_dict[day][goal_city-1]['min']
            csv_file_name_hour = cf.csv_patterns % (day, goal_city, start_hour, start_min)
            city_data_hour_df = pd.read_csv(os.path.join(cf.cost_num_steps_dir, csv_file_name_hour), index_col=None, names=['target', 'date', 'time', 'xid', 'yid'])
            sub_df = a_star_submission_3d(day, goal_city, city_data_hour_df, start_hour, int(start_min/10))
            frames.append(sub_df)
    sub_csv = pd.concat(frames, axis=0)
    combined_csv_name = cf.csv_patterns.split(']')[0] + '].csv'
    sub_csv.to_csv(os.path.join(cf.cost_num_steps_dir, combined_csv_name), header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])


def a_star_submission_3d(day, goal_city, city_data_hour_df, start_hour, slot):
    #### create one submit path
    # A random time to get the time string right
    ti = datetime(2017, 11, 21, start_hour, slot*10)
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

