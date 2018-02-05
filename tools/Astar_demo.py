from tools.Astar import *
from tools.Astar_3D import GridWithWeights_3D, a_star_search_3D
import numpy as np

def main_article():
    # data from main article
    diagram4 = GridWithWeights(10, 10)
    diagram4.walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
    diagram4.weights = {loc: 5 for loc in [(3, 4), (3, 5), (4, 1), (4, 2),
                                           (4, 3), (4, 4), (4, 5), (4, 6),
                                           (4, 7), (4, 8), (5, 1), (5, 2),
                                           (5, 3), (5, 4), (5, 5), (5, 6),
                                           (5, 7), (5, 8), (6, 2), (6, 3),
                                           (6, 4), (6, 5), (6, 6), (6, 7),
                                           (7, 3), (7, 4), (7, 5)]}
    start, goal = (1, 4), (7, 8)
    came_from, cost_so_far = a_star_search(diagram4, start, goal)
    draw_grid(diagram4, width=3, point_to=came_from, start=start, goal=goal)
    print()
    draw_grid(diagram4, width=3, number=cost_so_far, start=start, goal=goal)
    print()


def a_start_goal_not_reachable():
    # what if the goal is not reachable?
    diagram = GridWithWeights(3, 3)
    diagram.walls = [(1, 1), (1, 0)]
    start = (0, 0)
    goals = [(0, 2), (2, 2)]

    for goal in goals:
        came_from, cost_so_far = a_star_search(diagram, start, goal)

        draw_grid(diagram, width=3, point_to=came_from, start=start, goal=goal)
        print()
        draw_grid(diagram, width=3, number=cost_so_far, start=start, goal=goal)
        print()


def a_start_3d_demo():
    diagram = GridWithWeights_3D(4, 4, 8, wall_wind=15)
    diagram.weights = np.ones(shape=(4, 4, 8))
    # we manually add some barriers here
    wall_wind = 15
    diagram.weights[2, 2, 2] = wall_wind
    diagram.weights[2, 3, 2] = wall_wind
    diagram.weights[3, 2, 2] = wall_wind
    diagram.weights[2, 2, 3] = wall_wind
    diagram.weights[2, 3, 3] = wall_wind
    diagram.weights[3, 2, 3] = wall_wind
    diagram.weights[2, 2, 4] = wall_wind
    diagram.weights[2, 3, 4] = wall_wind
    diagram.weights[3, 2, 4] = wall_wind
    diagram.weights[2, 2, 5] = wall_wind
    diagram.weights[2, 3, 5] = wall_wind
    diagram.weights[3, 2, 5] = wall_wind

    start_loc_3D = (1, 0, 0)
    goal_loc_3D = [(3, 3, t) for t in range(8)]
    came_from, cost_so_far, current, wind_costs, rainfall_costs = a_star_search_3D(diagram, start_loc_3D, goal_loc_3D)


    route_list = []
    current_loc = list(set(goal_loc_3D) & set(came_from.keys()))

    if not len(current_loc):
        print('We cannot reach the goal, continue!')
    else:
        find_loc = current_loc[0]
        while came_from[find_loc] is not None:
            prev_loc = came_from[find_loc]
            route_list.append(prev_loc)
            find_loc = prev_loc
    print(route_list)

if __name__ == "__main__":
    a_start_3d_demo()


