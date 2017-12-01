from tools.Astar import *


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


if __name__ == "__main__":
    main_article()


