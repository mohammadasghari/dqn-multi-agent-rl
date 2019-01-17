"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import random
import operator
import numpy as np
import pygame
import sys
import os

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
ORANGE = (255, 128, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 60
HEIGHT = 60

# This sets the margin between each cell
MARGIN = 1


class agentslandmarks:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    A = [UP, DOWN, LEFT, RIGHT, STAY]
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self, args, current_path):
        self.game_mode = args['game_mode']
        self.reward_mode = args['reward_mode']
        self.num_agents = args['agents_number']
        self.num_landmarks = self.num_agents
        self.grid_size = args['grid_size']
        self.state_size = (self.num_agents + self.num_landmarks) * 2
        self.agents_positions = []
        self.landmarks_positions = []

        self.render_flag = args['render']
        self.recorder_flag = args['recorder']
        # enables visualizer
        if self.render_flag:
            [self.screen, self.my_font] = self.gui_setup()
            self.step_num = 1

            resource_path = os.path.join(current_path, 'environments')  # The resource folder path
            resource_path = os.path.join(resource_path, 'agents_landmarks')  # The resource folder path
            image_path = os.path.join(resource_path, 'images')  # The image folder path

            img = pygame.image.load(os.path.join(image_path, 'agent.jpg')).convert()
            self.img_agent = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'landmark.jpg')).convert()
            self.img_landmark = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'agent_landmark.jpg')).convert()
            self.img_agent_landmark = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'agent_agent_landmark.jpg')).convert()
            self.img_agent_agent_landmark = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'agent_agent.jpg')).convert()
            self.img_agent_agent = pygame.transform.scale(img, (WIDTH, WIDTH))

            if self.recorder_flag:
                self.snaps_path = os.path.join(current_path, 'results_agents_landmarks')  # The resource folder path
                self.snaps_path = os.path.join(self.snaps_path, 'snaps')  # The resource folder path

        self.cells = []
        self.positions_idx = []

        # self.agents_collide_flag = args['collide_flag']
        # self.penalty_per_collision = args['penalty_collision']
        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]

        positions_idx = []

        if self.game_mode == 0:
            # first enter the positions for the landmarks and then for the agents. If the grid is n*n, then the
            # positions are
            #  0                1             2     ...     n-1
            #  n              n+1           n+2     ...    2n-1
            # 2n             2n+1          2n+2     ...    3n-1
            #  .                .             .       .       .
            #  .                .             .       .       .
            #  .                .             .       .       .
            # (n-1)*n   (n-1)*n+1     (n-1)*n+2     ...   n*n+1
            # , e.g.,
            # positions_idx = [0, 6, 23, 24] where 0 and 6 are the positions of landmarks and 23 and 24 are positions
            # of agents
            positions_idx = []

        if self.game_mode == 1:
            positions_idx = np.random.choice(len(cells), size=self.num_landmarks + self.num_agents,
                                             replace=False)

        return [cells, positions_idx]

    def reset(self):  # initialize the world

        self.terminal = False
        [self.cells, self.positions_idx] = self.set_positions_idx()

        # separate the generated position indices for walls, pursuers, and evaders
        landmarks_positions_idx = self.positions_idx[0:self.num_landmarks]
        agents_positions_idx = self.positions_idx[self.num_landmarks:self.num_landmarks + self.num_agents]

        # map generated position indices to positions
        self.landmarks_positions = [self.cells[pos] for pos in landmarks_positions_idx]
        self.agents_positions = [self.cells[pos] for pos in agents_positions_idx]

        initial_state = list(sum(self.landmarks_positions + self.agents_positions, ()))

        return initial_state

    def step(self, agents_actions):
        # update the position of agents
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)

        if self.reward_mode == 0:

            binary_cover_list = []

            for landmark in self.landmarks_positions:
                distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                             for agent_pos in self.agents_positions]

                min_dist = min(distances)

                if min_dist == 0:
                    binary_cover_list.append(min_dist)
                else:
                    binary_cover_list.append(1)

            # check the terminal case
            if sum(binary_cover_list) == 0:
                reward = 0
                self.terminal = True
            else:
                reward = -1
                self.terminal = False

        if self.reward_mode == 1:

            binary_cover_list = []

            for landmark in self.landmarks_positions:
                distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                             for agent_pos in self.agents_positions]

                min_dist = min(distances)

                if min_dist == 0:
                    binary_cover_list.append(0)
                else:
                    binary_cover_list.append(1)

            reward = -1 * sum(binary_cover_list)
            # check the terminal case
            if reward == 0:
                self.terminal = True
            else:
                self.terminal = False

        if self.reward_mode == 2:

            # calculate the sum of minimum distances of agents to landmarks
            reward = 0
            for landmark in self.landmarks_positions:
                distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                             for agent_pos in self.agents_positions]

                reward -= min(distances)

            # check the terminal case
            if reward == 0:
                self.terminal = True

        new_state = list(sum(self.landmarks_positions + self.agents_positions, ()))

        return [new_state, reward, self.terminal]

    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        for idx in xrange(len(pos_list)):
            if act_list[idx] != 4:
                pos_act_applied = map(operator.add, pos_list[idx], self.A_DIFF[act_list[idx]])
                # checks to make sure the new pos in inside the grid
                for i in xrange(0, 2):
                    if pos_act_applied[i] < 0:
                        pos_act_applied[i] = 0
                    if pos_act_applied[i] >= self.grid_size:
                        pos_act_applied[i] = self.grid_size - 1
                positions_action_applied.append(tuple(pos_act_applied))
            else:
                positions_action_applied.append(pos_list[idx])

        final_positions = []

        for pos_idx in xrange(len(pos_list)):
            if positions_action_applied[pos_idx] == pos_list[pos_idx]:
                final_positions.append(pos_list[pos_idx])
            elif positions_action_applied[pos_idx] not in pos_list and positions_action_applied[
                pos_idx] not in positions_action_applied[
                                0:pos_idx] + positions_action_applied[
                                             pos_idx + 1:]:
                final_positions.append(positions_action_applied[pos_idx])
            else:
                final_positions.append(pos_list[pos_idx])

        return final_positions

    def action_space(self):
        return len(self.A)

    def render(self):

        pygame.time.delay(500)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill(BLACK)
        text = self.my_font.render("Step: {0}".format(self.step_num), 1, WHITE)
        self.screen.blit(text, (5, 15))

        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (row, column)

                frequency = self.find_frequency(pos, self.agents_positions)

                if pos in self.landmarks_positions and frequency >= 1:
                    if frequency == 1:
                        self.screen.blit(self.img_agent_landmark,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    else:
                        self.screen.blit(self.img_agent_agent_landmark,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

                elif pos in self.landmarks_positions:
                    self.screen.blit(self.img_landmark,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

                elif frequency >= 1:
                    if frequency == 1:
                        self.screen.blit(self.img_agent,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    elif frequency > 1:
                        self.screen.blit(self.img_agent_agent,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    else:
                        print('Error!')
                else:
                    pygame.draw.rect(self.screen, WHITE,
                                     [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
                                      HEIGHT])

        if self.recorder_flag:
            file_name = "%04d.png" % self.step_num
            pygame.image.save(self.screen, os.path.join(self.snaps_path, file_name))

        if not self.terminal:
            self.step_num += 1

    def gui_setup(self):

        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        board_size_x = (WIDTH + MARGIN) * self.grid_size
        board_size_y = (HEIGHT + MARGIN) * self.grid_size

        window_size_x = int(board_size_x)
        window_size_y = int(board_size_y * 1.2)

        window_size = [window_size_x, window_size_y]
        screen = pygame.display.set_mode(window_size)

        # Set title of screen
        pygame.display.set_caption("Agents-and-Landmarks Game")

        myfont = pygame.font.SysFont("monospace", 30)

        return [screen, myfont]

    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq
