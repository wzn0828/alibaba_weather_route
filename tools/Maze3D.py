import numpy as np


class Maze_3D:
    """
    A wrapper class for a 3D maze, containing all the information about the maze.
    Maze has the third dimension t
    to other maze
    """
    def __init__(self,
                 height,
                 width,
                 time_length,
                 start_state,
                 goal_states,
                 return_to_start=False,
                 strong_wind_return=False,
                 reward_goal=0.0,
                 reward_move=-1.0,
                 reward_obstacle=-100.,
                 maxSteps=1e5,
                 wind_real_day_hour_total=None,
                 cf={}):

        self.WORLD_WIDTH = width
        self.WORLD_HEIGHT = height
        self.TIME_LENGTH = time_length
        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTION_STAY = 4
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_STAY]
        # start state
        self.START_STATE = start_state
        # goal states
        self.GOAL_STATES = goal_states
        # all reward
        self.reward_goal = reward_goal
        self.reward_move = reward_move
        self.reward_obstacle = reward_obstacle

        self.wind_real_day_hour_total = wind_real_day_hour_total
        self.return_to_start = return_to_start
        self.strong_wind_return = strong_wind_return
        self.maxSteps = maxSteps

        self.cf = cf

    def takeAction(self, state, action):
        """
        :param state:
        :param action:
        :param stochastic_wind:
        :return:
        """
        x, y, t = state
        # the time always goes forward
        t += 1
        if t >= self.TIME_LENGTH:
            # It will be a very undesirable state, we go back to the start state
            x, y, t = self.START_STATE
            reward = self.reward_obstacle
            return [x, y, t], reward

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        elif action == self.ACTION_STAY:
             x, y = x, y

        current_loc_time_wind = self.wind_real_day_hour_total[self.cf.temp_model, x, y, t]
        if current_loc_time_wind >= self.cf.wall_wind:
            if self.return_to_start:
                x, y, t = self.START_STATE
            elif self.strong_wind_return:
                x, y, _ = state

            reward = self.reward_obstacle

        elif tuple([x, y, t]) in self.GOAL_STATES:
            reward = self.reward_goal
        else:
            if self.cf.risky:
                reward = self.reward_move
            elif self.cf.wind_exp:
                current_loc_time_wind -= self.cf.wind_exp_mean
                current_loc_time_wind /= self.cf.wind_exp_std
                reward = self.reward_move + (-1) * np.exp(current_loc_time_wind).astype('int')
            else:
                current_loc_time_wind /= self.cf.risky_coeff
                reward = self.reward_move + (-1) * current_loc_time_wind

        return [x, y, t], reward


