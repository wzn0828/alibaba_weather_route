import numpy as np
import heapq
import itertools


class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()
        self.REMOVED = '<removed-task>'

    def empty(self):
        return not self.entry_finder

    def removeItem(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def addItem(self, item, priority=0):
        if item in self.entry_finder:
            self.removeItem(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def popTask(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')


class Dyna_3D:
    """
    # model for planning in Dyna
    """
    def __init__(self,
                 rand,
                 maze,
                 gamma=0.95,
                 epsilon=0.1,
                 planningSteps=5,
                 expected=False,
                 qLearning=False,
                 double=False,
                 alpha=0.5,
                 plus=False,
                 kappa=1e-4,
                 priority=False,
                 ucb=False,
                 theta=1e-4,
                 policy_init=[],
                 optimal_length_relax=1.2,
                 heuristic=False,
                 increase_epsilon=0,
                 polynomial_alpha=False,
                 polynomial_alpha_coefficient=0.8):
        """
        :param rand: @rand: an instance of np.random.RandomState for sampling
        :param maze:
        :param planningSteps:
        :param expected:
        :param qLearning:
        :param alpha:
        :param kappa:
        """
        self.model = dict()
        self.rand = rand
        self.maze = maze
        self.stateActionValues = np.zeros((maze.WORLD_HEIGHT, maze.WORLD_WIDTH, maze.TIME_LENGTH, len(maze.actions)))
        # expected sarsa
        self.expected = expected
        # Q-learning
        self.qlearning = qLearning
        # Double Q-learning
        self.double = double
        if self.qlearning:
            self.name = 'Dyna-Q'

        elif self.expected:
            self.name = 'DynaExpectedSarsa'
        else:
            self.name = 'Dyna-Sarsa'

        if self.double:
            self.name += '_Double'
            self.double_first = True  # True means updating the first state action value, False will update the second
            self.stateActionValues2 = np.zeros((maze.WORLD_HEIGHT, maze.WORLD_WIDTH, maze.TIME_LENGTH, len(maze.actions)))

        # discount
        self.gamma = gamma
        # probability for exploration
        self.epsilon = epsilon
        # step size
        self.alpha = alpha
        # weight for elapsed time
        self.timeWeight = 0
        # n-step planning
        self.planningSteps = planningSteps

        # for every maxSteps fail to reach the goal, we increase the epilson
        self.increase_epsilon = increase_epsilon

        # threshold for priority queue
        self.theta = theta

        # Dyna Plus algorithm
        self.plus = plus  # flag for Dyna+ algorithm
        if self.plus:
            self.name += '_plus'
            self.kappa = kappa  # Time weight
            self.time = 0  # track the total time

        self.priority = priority
        if self.priority:
            self.theta = theta
            self.name += '_PrioritizedSweeping'
            self.priorityQueue = PriorityQueue()
            # track predessors for every state
            self.predecessors = dict()
            if heuristic:
                self.name += '_heuristic'
            self.heuristic = heuristic
            self.heuristic_const = self.heuristic_fn(self.maze.START_STATE, self.maze.GOAL_STATES)

        # policy initilisation
        self.policy_init = policy_init
        if type(self.policy_init):
            self.name += '_A_star'

        # polynomial learning rate for alpha
        self.polynomial_alpha = polynomial_alpha
        if polynomial_alpha:
            self.polynomial_alpha_coefficient = polynomial_alpha_coefficient
            self.alpha_count = np.zeros(self.stateActionValues.shape).astype(int)

        # for checking the optimal path length and a stopping criterion
        self.optimal_length_relax = optimal_length_relax
        # Upper-Confidence-Bound Action Selection
        self.ucb = ucb
        if self.ucb:
            self.time = 0
            self.name += '_UCB'

        self.name = self.name + '_planning:_%d' % planningSteps

    def insert(self, priority, state, action):
        """
        add a state-action pair into the priority queue with priority
        :param priority:
        :param state:
        :param action:
        :return:
        """
        # note the priority queue is a minimum heap, so we use -priority
        self.priorityQueue.addItem((tuple(state), action), -priority)

    def feed(self, currentState, action, newState, reward):
        """"
        feed the model with previous experience
        """
        if self.plus:
            self.time += 1
            if tuple(currentState) not in self.model.keys():
                self.model[tuple(currentState)] = dict()
                # Actions that had never been tried before from a state were allowed to be
                # considered in the planning step
                for action_ in self.maze.actions:
                    if action_ != action:
                        # Such actions would lead back to the same state with a reward of zero
                        # Notice that the minimum time stamp is 1 instead of 0
                        self.model[tuple(currentState)][action_] = [list(currentState), 0, 1]
            self.model[tuple(currentState)][action] = [list(newState), reward, self.time]
        elif self.ucb:
            self.time += 1
            if tuple(currentState) not in self.model.keys():
                self.model[tuple(currentState)] = dict()
                # Actions that had never been tried before from a state were allowed to be
                # considered in the planning step
                for action_ in self.maze.actions:
                    self.model[tuple(currentState)][action_] = [list(currentState), 0, 1]

            N_t_a = self.model[tuple(currentState)][action][2]
            self.model[tuple(currentState)][action] = [list(newState), reward, N_t_a+1]

        else:
            if tuple(currentState) not in self.model.keys():
                self.model[tuple(currentState)] = dict()
            self.model[tuple(currentState)][action] = [list(newState), reward]

        # priority queue
        if self.priority:
            if tuple(newState) not in self.predecessors.keys():
                self.predecessors[tuple(newState)] = set()
            self.predecessors[tuple(newState)].add((tuple(currentState), action))

    def sample(self):
        """
        sample from previous experience
        :return:
        """
        if self.priority:
            # get the  first item in the priority queue
            (state, action), priority = self.priorityQueue.popTask()

            if self.plus:
                newState, reward, time = self.model[state][action]
                # adjust reward with elapsed time since last visit
                priority -= self.kappa * np.sqrt(self.time - time)
            elif self.ucb:
                newState, reward, N_t_a = self.model[state][action]
                # adjust reward with elapsed time since last visit
                priority -= self.ucb * np.sqrt(np.log(self.time) / N_t_a)
            else:
                newState, reward = self.model[state][action]
            return -priority, list(state), action, list(newState), reward
        else:
            # randomly sample from previous experience
            stateIndex = self.rand.choice(range(0, len(self.model.keys())))
            state = list(self.model)[stateIndex]
            actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
            action = list(self.model[state])[actionIndex]
            if self.plus:
                newState, reward, time = self.model[state][action]
                # adjust reward with elapsed time since last visit
                reward += self.kappa * np.sqrt(self.time - time)
            elif self.ucb:
                newState, reward, N_t_a = self.model[state][action]
                # adjust reward with elapsed time since last visit
                reward += self.ucb * np.sqrt(np.log(self.time) / N_t_a)
            else:
                newState, reward = self.model[state][action]

            return list(state), action, list(newState), reward

    def get_predecessor(self, state):
        """
        get predecessor for prioritize sweeping: for all, S', A' predicted to lead to S
        :param state:
        :return:
        """
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for statePre, actionPre in list(self.predecessors[tuple(state)]):
            predecessors.append([list(statePre), actionPre, self.model[statePre][actionPre][1]])
        return predecessors

    def return_action(self, start_state, end_state):
        x1, y1, t1 = start_state
        x2, y2, t2 = end_state
        if (y1-y2) == 1 and x1 == x2:
            return self.maze.ACTION_LEFT
        elif (y2-y1) == 1 and x1 == x2:
            return self.maze.ACTION_RIGHT
        elif y1 == y2 and (x1-x2) == 1:
            return self.maze.ACTION_UP
        elif y1 == y2 and (x2-x1) == 1:
            return self.maze.ACTION_DOWN
        elif x1 == x2 and y1 == y2:
            return self.maze.ACTION_STAY
        else:
            raise AssertionError('Action illegal')

    def chooseAction(self, state):
        """
        Choose an action based on epsilon-greedy algorithm
        However, if there is an initial policy, we follow that policy with epsilon-greedy
        for exploration
        :param state:
        :return:
        """
        if len(self.policy_init):
            a_star_traj = self.policy_init[self.a_star_model]
            if tuple(state) in a_star_traj.keys():
                go_to = a_star_traj[tuple(state)]
                # if type(go_to) == list:
                #     go_to = random.choice(go_to)
                return self.return_action(tuple(state), go_to), [], []

        else:
            # we follow an epsilon greedy alorithm and requires valid neighbours
            viable_neighbours = self.maze.neighbors(state)
            viable_actions = self.maze.viable_actions(state, viable_neighbours)

            # This is an exception that we have reached our terminal states
            if state[2] == self.maze.TIME_LENGTH-1:
                viable_actions = np.array([self.maze.ACTION_STAY])

            if np.random.binomial(1, self.epsilon) == 1:
                action = np.random.choice(viable_actions)
            else:
                if not self.double:
                    values = self.stateActionValues[state[0], state[1], state[2], viable_actions]
                else:
                    values = [item1 + item2 for item1, item2 in
                              zip(self.stateActionValues[state[0], state[1], state[2], viable_actions], self.stateActionValues2[state[0], state[1], state[2], viable_actions])]

                action = viable_actions[self.rand.choice([action for action, value in enumerate(values) if value == np.max(values)])]

            return action, viable_actions, viable_neighbours

    def play(self, environ_step=False):
        """
        # play for an episode for Dyna-Q algorithm
        # @stateActionValues: state action pair values, will be updated
        # @model: model instance for planning
        # @planningSteps: steps for planning
        # @maze: a maze instance containing all information about the environment
        # @dynaParams: several params for the algorithm
        :return:  if with environ_step as True, we will only return the actually interaction with the environment
        """
        currentState = self.maze.START_STATE
        steps = 0
        terminal_flag = False
        if self.polynomial_alpha:
            self.alpha_action = []
        while tuple(currentState) not in self.maze.GOAL_STATES and not terminal_flag:
            ######################## Interaction with the environment ###############################
            steps += 1
            # get action
            if steps == 1 or self.qlearning or len(self.policy_init):
                currentAction, viable_actions, viable_neighbours = self.chooseAction(currentState)
            # take action
            newState, reward, terminal_flag = self.maze.takeAction(currentState, currentAction)

            viable_neighbours = self.maze.neighbors(newState)
            viable_actions = self.maze.viable_actions(newState, viable_neighbours)
            # if we arrive the goal via the boundary
            if newState[2] == self.maze.TIME_LENGTH-1 and terminal_flag:
                viable_actions = np.array([self.maze.ACTION_STAY])

            if self.qlearning:
                # Q-Learning update
                if not self.double:
                    action_value_delta = reward + self.gamma * np.max(self.stateActionValues[newState[0], newState[1], newState[2], viable_actions]) - \
                                         self.stateActionValues[currentState[0], currentState[1], currentState[2], currentAction]
                else:
                    # the random seed only generates once here (during prioritized sweeping)
                    if self.rand.binomial(1, 0.5) == 1:
                        self.double_first = True
                        currentStateActionValues = self.stateActionValues
                        anotherStateActionValues = self.stateActionValues2
                    else:
                        self.double_first = False
                        currentStateActionValues = self.stateActionValues2
                        anotherStateActionValues = self.stateActionValues
                    bestAction = viable_actions[np.argmax(currentStateActionValues[newState[0], newState[1], newState[2], viable_actions])]
                    targetValue = anotherStateActionValues[newState[0], newState[1], newState[2], bestAction]
                    action_value_delta = reward + self.gamma * targetValue - currentStateActionValues[currentState[0], currentState[1], currentState[2], currentAction]
            else:
                # sarsa update
                newAction, viable_actions, viable_neighbours = self.chooseAction(newState)
                if not self.expected:
                    valueTarget = self.stateActionValues[newState[0], newState[1], newState[2], newAction]
                    action_value_delta = reward + self.gamma * valueTarget - self.stateActionValues[
                        currentState[0], currentState[1], currentState[2], currentAction]

                elif self.expected:
                    # expected sarsa update
                    valueTarget = 0.0
                    if not self.double:
                        actionValues = self.stateActionValues[newState[0], newState[1], newState[2], viable_actions]
                        bestActions = viable_actions[np.argwhere(actionValues == np.max(actionValues))]
                        for action in viable_actions:
                            if action in bestActions:
                                valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(viable_actions)) * \
                                               self.stateActionValues[newState[0], newState[1], newState[2], action]
                            else:
                                valueTarget += self.epsilon / len(viable_actions) * self.stateActionValues[newState[0], newState[1], newState[2], action]
                        action_value_delta = reward + self.gamma * valueTarget - self.stateActionValues[
                            currentState[0], currentState[1], currentState[2], currentAction]

                    else:
                        # the random seed only generates once here (during prioritized sweeping)
                        if self.rand.binomial(1, 0.5) == 1:
                            self.double_first = True
                            currentStateActionValues = self.stateActionValues
                            anotherStateActionValues = self.stateActionValues2
                        else:
                            self.double_first = False
                            currentStateActionValues = self.stateActionValues2
                            anotherStateActionValues = self.stateActionValues

                        actionValues = currentStateActionValues[newState[0], newState[1], newState[2], viable_actions]
                        bestActions = viable_actions[np.argwhere(actionValues == np.max(actionValues))]
                        for action in viable_actions:
                            if action in bestActions:
                                valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(viable_actions)) * \
                                               anotherStateActionValues[newState[0], newState[1], newState[2], action]
                            else:
                                valueTarget += self.epsilon / len(viable_actions) * anotherStateActionValues[newState[0], newState[1], newState[2], action]

                        action_value_delta = reward + self.gamma * valueTarget - currentStateActionValues[currentState[0], currentState[1], currentState[2], currentAction]

            if not self.priority:
                if not self.polynomial_alpha:
                    self.stateActionValues[currentState[0], currentState[1], currentState[2], currentAction] += self.alpha * action_value_delta
                else:
                    self.alpha_count[currentState[0], currentState[1], currentState[2], currentAction] += 1
                    alpha_action_temp = 1. / ((self.alpha_count[currentState[0], currentState[1], currentState[2], currentAction]) ** self.polynomial_alpha_coefficient)
                    self.alpha_action.append(alpha_action_temp)
                    self.stateActionValues[currentState[0], currentState[1], currentState[2], currentAction] += alpha_action_action_temp * action_value_delta

            else:
                if self.heuristic:
                    priority = np.abs(action_value_delta) - self.heuristic_fn(currentState, self.maze.GOAL_STATES) /self.heuristic_const
                else:
                    priority = np.abs(action_value_delta)
                if np.abs(action_value_delta) > self.theta:
                    self.insert(priority, currentState, currentAction)

            ######################## feed the model with experience ###############################
            self.feed(currentState, currentAction, newState, reward)

            if not self.qlearning:
                currentAction = newAction
            currentState = newState

            ######################## Planning from the model ###############################
            for t in range(0, self.planningSteps):
                if self.priority:
                    if self.priorityQueue.empty():
                        #  keep planning until the priority queue becomes empty will converge much faster
                        break
                    else:
                        # get a sample with highest priority from the model
                        priority, stateSample, actionSample, newStateSample, rewardSample = self.sample()
                else:
                    # sample experience from the model
                    stateSample, actionSample, newStateSample, rewardSample = self.sample()

                viable_neighbours = self.maze.neighbors(newStateSample)
                viable_actions = self.maze.viable_actions(newStateSample, viable_neighbours)
                # if we arrive the goal via the boundary
                if newStateSample[2] == self.maze.TIME_LENGTH - 1 and tuple(newStateSample) in self.maze.GOAL_STATES:
                    viable_actions = np.array([self.maze.ACTION_STAY])

                if self.qlearning:
                    if not self.double:
                        action_value_delta = rewardSample + self.gamma * np.max(self.stateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], viable_actions]) - \
                                             self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample]
                    else:
                        if self.double_first:
                            currentStateActionValues = self.stateActionValues
                            anotherStateActionValues = self.stateActionValues2
                        else:
                            currentStateActionValues = self.stateActionValues2
                            anotherStateActionValues = self.stateActionValues
                        bestAction = viable_actions[np.argmax(currentStateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], viable_actions])]
                        targetValue = anotherStateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], bestAction]
                        action_value_delta = rewardSample + self.gamma * targetValue - currentStateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample]

                else:
                    # sarsa or expected sarsa update
                    if not self.expected:
                        newAction_sample, viable_actions, viable_neighbours = self.chooseAction(newStateSample)
                        valueTarget = self.stateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], newAction_sample]
                        action_value_delta = rewardSample + self.gamma * valueTarget - self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample]

                    elif self.expected:
                        valueTarget = 0.0
                        if not self.double:
                            # calculate the expected value of new state expected sarsa update
                            actionValues = self.stateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], viable_actions]
                            bestActions = viable_actions[np.argwhere(actionValues == np.max(actionValues))]
                            for action in viable_actions:
                                if action in bestActions:
                                    valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(viable_actions)) * \
                                                   self.stateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], action]
                                else:
                                    valueTarget += self.epsilon / len(viable_actions) * self.stateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], action]
                                    # Sarsa update
                            action_value_delta = rewardSample + self.gamma * valueTarget - self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample]
                        else:
                            if self.double_first:
                                currentStateActionValues = self.stateActionValues
                                anotherStateActionValues = self.stateActionValues2
                            else:
                                currentStateActionValues = self.stateActionValues2
                                anotherStateActionValues = self.stateActionValues

                            actionValues = currentStateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], viable_actions]
                            bestActions = viable_actions[np.argwhere(actionValues == np.max(actionValues))]
                            for action in viable_actions:
                                if action in bestActions:
                                    valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(viable_actions)) * \
                                                   anotherStateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], action]
                                else:
                                    valueTarget += self.epsilon / len(viable_actions) * anotherStateActionValues[newStateSample[0], newStateSample[1], newStateSample[2], action]

                            action_value_delta = reward + self.gamma * valueTarget - currentStateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample]

                if not self.polynomial_alpha:
                    if not self.double:
                        self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample] += self.alpha * action_value_delta
                    else:
                        if self.double_first:
                            self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample] += self.alpha * action_value_delta
                        else:
                            self.stateActionValues2[stateSample[0], stateSample[1], stateSample[2], actionSample] += self.alpha * action_value_delta

                else:
                    if not self.double:
                        self.alpha_count[stateSample[0], stateSample[1], stateSample[2], actionSample] += 1
                        alpha_action_temp = 1. / ((self.alpha_count[stateSample[0], stateSample[1], stateSample[
                            2], actionSample]) ** self.polynomial_alpha_coefficient)
                        self.alpha_action.append(alpha_action_temp)
                        self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample] += alpha_action_temp * action_value_delta
                    else:
                        if self.double_first:
                            self.alpha_count_1[stateSample[0], stateSample[1], stateSample[2], actionSample] += 1
                            alpha_action_temp = 1. / ((self.alpha_count_1[stateSample[0], stateSample[1], stateSample[2], actionSample]) ** self.polynomial_alpha_coefficient)
                            self.alpha_action.append(alpha_action_temp)
                            self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], actionSample] += alpha_action_temp * action_value_delta
                        else:
                            self.alpha_count_2[stateSample[0], stateSample[1], stateSample[2], actionSample] += 1
                            alpha_action_temp = 1. / ((self.alpha_count_2[stateSample[0], stateSample[1], stateSample[2], actionSample]) ** self.polynomial_alpha_coefficient)
                            self.alpha_action.append(alpha_action_temp)
                            self.stateActionValues2[stateSample[0], stateSample[1], stateSample[2], actionSample] += alpha_action_temp * action_value_delta

                if self.priority:
                    # deal with all the predecessors of the sample states
                    # print(stateSample, end=': ')
                    # print('get_predecessor--> len(%d)' % len(self.get_predecessor(stateSample)))
                    for statePre, actionPre, rewardPre in self.get_predecessor(stateSample):
                        viable_neighbours = self.maze.neighbors(stateSample)
                        viable_actions = self.maze.viable_actions(stateSample, viable_neighbours)
                        # if we arrive the goal via the boundary
                        if stateSample[2] == self.maze.TIME_LENGTH - 1 and tuple(stateSample) in self.maze.GOAL_STATES:
                            viable_actions = np.array([self.maze.ACTION_STAY])
                        if self.qlearning:
                            if not self.double:
                                action_value_delta = rewardPre + self.gamma * np.max(self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], viable_actions]) - \
                                                     self.stateActionValues[statePre[0], statePre[1], statePre[2], actionPre]
                            else:
                                if self.double_first:
                                    currentStateActionValues = self.stateActionValues
                                    anotherStateActionValues = self.stateActionValues2
                                else:
                                    currentStateActionValues = self.stateActionValues2
                                    anotherStateActionValues = self.stateActionValues
                                bestAction = viable_actions[np.argmax(currentStateActionValues[stateSample[0], stateSample[1], stateSample[2], viable_actions])]
                                targetValue = anotherStateActionValues[stateSample[0], stateSample[1], stateSample[2], bestAction]
                                action_value_delta = rewardPre + self.gamma * targetValue - currentStateActionValues[statePre[0], statePre[1], statePre[2], actionPre]

                        else:
                            # sarsa or expected sarsa update
                            if not self.expected:
                                newAction_sample, viable_actions, viable_neighbours = self.chooseAction(stateSample)
                                valueTarget = self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], newAction_sample]
                                action_value_delta = rewardPre + self.gamma * valueTarget - self.stateActionValues[statePre[0], statePre[1], statePre[2], actionPre]

                            elif self.expected:
                                # calculate the expected value of new state
                                valueTarget = 0.0
                                if not self.double:
                                    actionValues = self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], viable_actions]
                                    bestActions = viable_actions[np.argwhere(actionValues == np.max(actionValues))]
                                    for action in viable_actions:
                                        if action in bestActions:
                                            valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(viable_actions)) * \
                                                           self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], action]
                                        else:
                                            valueTarget += self.epsilon / len(viable_actions) * self.stateActionValues[stateSample[0], stateSample[1], stateSample[2], action]

                                    action_value_delta = rewardPre + self.gamma * valueTarget - self.stateActionValues[statePre[0], statePre[1], statePre[2], actionPre]
                                else:
                                    if self.double_first:
                                        currentStateActionValues = self.stateActionValues
                                        anotherStateActionValues = self.stateActionValues2
                                    else:
                                        currentStateActionValues = self.stateActionValues2
                                        anotherStateActionValues = self.stateActionValues
                                    actionValues = currentStateActionValues[stateSample[0], stateSample[1], stateSample[2], viable_actions]
                                    bestActions = viable_actions[np.argwhere(actionValues == np.max(actionValues))]
                                    for action in viable_actions:
                                        if action in bestActions:
                                            valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(viable_actions)) * \
                                                           anotherStateActionValues[stateSample[0], stateSample[1], stateSample[2], action]
                                        else:
                                            valueTarget += self.epsilon / len(viable_actions) * \
                                                           anotherStateActionValues[stateSample[0], stateSample[1], stateSample[2], action]

                                    action_value_delta = reward + self.gamma * valueTarget - currentStateActionValues[statePre[0], statePre[1], statePre[2], actionPre]

                        if self.heuristic:
                            priority = np.abs(action_value_delta) - self.heuristic_fn(statePre,
                                                                                      self.maze.GOAL_STATES) / self.heuristic_const
                        else:
                            priority = np.abs(action_value_delta)

                        if np.abs(action_value_delta) > self.theta:
                            self.insert(priority, statePre, actionPre)

                if not environ_step:
                    steps += 1
            # check whether it has exceeded the step limit
            if steps > self.maze.maxSteps:
                print(currentState)
                break
        return steps

    def heuristic_fn(self, a, b):
        """
        https://en.wikipedia.org/wiki/A*_search_algorithm
         For the algorithm to find the actual shortest path, the heuristic function must be admissible,
         meaning that it never overestimates the actual cost to get to the nearest goal node.
         That's easy!
        :param a:
        :param b:
        :return:
        """

        (x1, y1, t1) = a
        (x2, y2, t2) = b[0]
        return abs(x1 - x2) + abs(y1 - y2)

    def checkPath(self, optimal_length, set_wind_to_zeros=False):
        """
        This function check whether the greedy action selection will lead to the goal states
        Check whether state-action values are already optimal
        :return:
        """
        # get the length of optimal path
        # optimal_length is the length of optimal path of the original maze
        # 1.2 means it's a relaxed optifmal path
        # for the path, we also set the wind to zeros
        if set_wind_to_zeros:
            wind_real_day_hour_total = self.maze.wind_real_day_hour_total
            self.maze.wind_real_day_hour_total = 0 * wind_real_day_hour_total
        maxSteps = optimal_length * self.optimal_length_relax

        currentState = self.maze.START_STATE
        steps = 0
        came_from = {}
        came_from[currentState] = None
        total_Q = 0
        total_reward = 0
        if self.double:
            stateActionValues = (self.stateActionValues + self.stateActionValues2)/2.
        else:
            stateActionValues = self.stateActionValues

        while tuple(currentState) not in self.maze.GOAL_STATES:

            bestAction = np.argmax(stateActionValues[currentState[0], currentState[1], currentState[2], :])
            total_Q += np.max(stateActionValues[currentState[0], currentState[1], currentState[2], :])
            nextState, reward, terminal_flag = self.maze.takeAction(currentState, bestAction)
            total_reward += reward
            came_from[tuple(nextState)] = tuple(currentState)
            currentState = nextState
            steps += 1

            if steps > maxSteps or currentState[2] >= self.maze.TIME_LENGTH:
                if set_wind_to_zeros:
                    self.maze.wind_real_day_hour_total = wind_real_day_hour_total
                return came_from, tuple(currentState), False, total_Q, total_reward, stateActionValues.sum()
        if set_wind_to_zeros:
            self.maze.wind_real_day_hour_total = wind_real_day_hour_total
        return came_from, tuple(currentState), True, total_Q, total_reward, stateActionValues.sum()
