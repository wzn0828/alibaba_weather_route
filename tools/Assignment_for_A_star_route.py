import numpy as np

from tools.A_star_alibaba import extract_start_hours

def assignment_for_A_star_route(cf, cost_dict):
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

    # First we constract a cost matrix with dim: num_cities (10) * num_timeslots (18 *6)
    cost_matrix = np.zeros(shape=(10, 18))
    step_fraction_matrix = np.zeros(shape=(10, 18))

    for goal_city in range(10):
        start_hour = cost_dict['goal_city']['start_hour']
        num_steps = cost_dict['goal_city']['start_hour']['num_steps']
        max_cost_sum = cost_dict['goal_city']['start_hour']['max_cost_sum']
        cost_matrix[goal_city, start_hour] = max_cost_sum

        # this step fraction will indicate the fraction of times remaining by whole hour
        # if two goal city need to start fly from the same hour, the higher this fraction this,
        # the more in advance we need to fly this ballon first
        step_fraction = np.mod(num_steps, cf.hourly_travel_distance)
        step_fraction_matrix[goal_city, start_hour] = step_fraction

    # We start assigining the slot from the furtherest city to the closest city
    dist_manhattan = np.zeros(10)
    for goal_city in range(1, 11):
        start_hours, dist_manhattan[goal_city-1] = extract_start_hours(cf, goal_city)

    # this will sort the distance from small to large
    assignment_matrix = np.zeros(shape=(10, 18*6))
    goal_city_priority = np.argsort(dist_manhattan)
    for goal_city in goal_city_priority[::-1]:
        goal_city_minin_start_hour = np.argmin(cost_matrix[goal_city])
        for s in range(6):
            if np.sum(assignment_matrix[:, goal_city_minin_start_hour+s]) >= 1:
                print("Hour %d, slot %d has been occupied" % (goal_city_minin_start_hour, s))
            else:
                # we assign this slot for this city
                assignment_matrix[goal_city, goal_city_minin_start_hour + s] = 1
                break

    # Now we read the assignment matrix and combine the csv files:


