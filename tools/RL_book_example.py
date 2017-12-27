import numpy as np
import matplotlib.pyplot as plt
from tools.Maze3D import Maze_3D
from tools.Dyna_3D import Dyna_3D
from timeit import default_timer as timer
from tools.Astar_3D import a_star_search_3D, walk_final_grid_go_to, convert_3D_maze_to_grid, draw_grid_3d
from config.configuration import Configuration


def main():
    # Load configuration files
    configuration = Configuration('/home/stevenwudi/PycharmProjects/alibaba_weather_route/config/diwu.py')
    cf = configuration.load()
    # get the original 6*9 maze
    time_length = 25
    maze = Maze_3D(height=7,
                   width=9,
                   time_length=time_length,
                   start_state=(2, 0, 0),
                    goal_states=[[0, 8]],
                    return_to_start=False,
                    reward_goal=cf.reward_goal,
                    reward_move=cf.reward_move,
                    reward_obstacle=cf.reward_obstacle,
                    cf=cf)
    # set up goal states
    gs = []
    for t in range(time_length):
        gs.append(tuple([0, 8, t]))
    maze.GOAL_STATES = gs
    # set up obstacles
    maze.wind_real_day_hour_total = np.zeros(shape=(maze.WORLD_HEIGHT, maze.WORLD_WIDTH, maze.TIME_LENGTH))
    obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    for t in range(time_length-5):
        for item in obstacles:
            maze.wind_real_day_hour_total[item[0], item[1], t] = cf.wall_wind
    obstacles = [[2, 4], [3, 4], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    for t in range(5, time_length):
        for item in obstacles:
            maze.wind_real_day_hour_total[item[0], item[1], t] = cf.wall_wind
    for t in range(5, time_length-3):
        maze.wind_real_day_hour_total[1, 8, t] = cf.wall_wind
    mazes = [maze]
    # Dyna model hyper
    rand = np.random.RandomState(0)
    planningSteps = 5
    alpha = 0.5
    gamma = 0.99
    theta = 1e-4
    epsilon = 0.1
    # track the # of backups
    runs = 1
    numOfMazes = 1
    numOfModels = 5
    backups = np.zeros((runs, numOfModels, numOfMazes))

    for mazeIndex, maze in enumerate(mazes):
        ############## A star search algorithm  ######################################
        start_time = timer()
        diagram = convert_3D_maze_to_grid(maze, cf)
        came_from, cost_so_far = a_star_search_3D(diagram, tuple(maze.START_STATE), maze.GOAL_STATES)
        go_to_all, steps = walk_final_grid_go_to(tuple(maze.START_STATE), maze.GOAL_STATES, came_from)
        if True:
            draw_grid_3d(diagram, came_from=came_from, start=tuple(maze.START_STATE),
                         goal=tuple(maze.GOAL_STATES), title='A star')
        print('Finish, using %.2f sec!' % (timer() - start_time))
        # the following expand dims just to cater to challenge might have 10 models
        maze.wind_real_day_hour_total = np.tile(maze.wind_real_day_hour_total, [10,1,1,1])
        ##############End  A star search algorithm  ######################################
        model_Dyna_Q = Dyna_3D(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=planningSteps,
                             qLearning=True,
                             expected=False,
                             alpha=alpha,
                             priority=False)

        model_Dyna_Prioritized_Sweeping_ES = Dyna_3D(rand=rand,
                                         maze=maze,
                                         epsilon=epsilon,
                                         gamma=gamma,
                                         planningSteps=planningSteps,
                                         qLearning=False,
                                         expected=True,
                                         alpha=alpha,
                                         priority=True,
                                         theta=theta)

        model_Dyna_Prioritized_Sweeping_Q = Dyna_3D(rand=rand,
                                         maze=maze,
                                         epsilon=epsilon,
                                         gamma=gamma,
                                         planningSteps=planningSteps,
                                         qLearning=True,
                                         expected=False,
                                         alpha=alpha,
                                         priority=True,
                                         theta=theta)

        model_Dyna_Prioritized_Sweeping_Q_A_star = Dyna_3D(rand=rand,
                                                           maze=maze,
                                                           epsilon=epsilon,
                                                           gamma=gamma,
                                                           planningSteps=planningSteps,
                                                           qLearning=True,
                                                           expected=False,
                                                           alpha=alpha,
                                                           priority=False,
                                                           theta=theta,
                                                           policy_init=go_to_all)

        model_Dyna_Prioritized_Sweeping_Q_heuristic = Dyna_3D(rand=rand,
                                                           maze=maze,
                                                           epsilon=epsilon,
                                                           gamma=gamma,
                                                           planningSteps=planningSteps,
                                                           qLearning=True,
                                                           expected=False,
                                                           alpha=alpha,
                                                           priority=True,
                                                           theta=theta,
                                                           heuristic=True)

        models = [model_Dyna_Prioritized_Sweeping_Q_heuristic, model_Dyna_Q, model_Dyna_Prioritized_Sweeping_Q_A_star,
                  model_Dyna_Prioritized_Sweeping_Q, model_Dyna_Prioritized_Sweeping_ES]

        models = [model_Dyna_Q, model_Dyna_Prioritized_Sweeping_Q_A_star,
                  model_Dyna_Prioritized_Sweeping_Q, model_Dyna_Prioritized_Sweeping_ES]

        for m, model in enumerate(models):
            for run in range(0, runs):
                print('run:', run, model.name, 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)
                start_time = timer()
                # track steps / backups for each episode
                steps = []
                # play for an episode
                while True:
                    steps.append(model.play())
                    # print best action w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)
                    # check whether the (relaxed) optimal path is found
                    came_from = model.checkPath(len(go_to_all.keys()))
                    if came_from:
                        draw_grid_3d(diagram, came_from=came_from, start=tuple(maze.START_STATE),
                                     goal=tuple(maze.GOAL_STATES), title=model.name)
                        print(steps)
                        break

                backups[run][m][mazeIndex] = np.sum(steps)
                print('Finish, using %.2f sec!' % (timer() - start_time))

    # Dyna-Q performs several backups per step
    backups = np.sum(backups, axis=0)
    # average over independent runs
    backups /= float(runs)
    print(backups)
    plt.show()


if __name__ == "__main__":
    main()