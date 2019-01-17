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


class PredatorsPrey(object):

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    A = [UP, DOWN, LEFT, RIGHT, STAY]
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0,0)]

    def __init__(self, args, current_path):

        self.num_predators = args['agents_number']
        self.num_preys = 1
        self.preys_mode = args['preys_mode']
        self.num_walls = 0
        self.grid_size = args['grid_size']

        self.game_mode = args['game_mode']
        self.reward_mode = args['reward_mode']

        self.state_size = (self.num_preys + self.num_predators + self.num_walls)*2
        self.predators_positions = []
        self.preys_positions = []
        self.walls_positions = []
        self.render_flag = args['render']
        self.recorder_flag = args['recorder']
        # enables visualizer
        if self.render_flag:
            [self.screen, self.my_font] = self.gui_setup()
            self.step_num = 1

            resource_path = os.path.join(current_path, 'environments')  # The resource folder path
            resource_path = os.path.join(resource_path, 'predators_prey')  # The resource folder path
            image_path = os.path.join(resource_path, 'images')  # The image folder path

            img = pygame.image.load(os.path.join(image_path, 'predator_prey.jpg')).convert()
            self.img_predator_prey = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'predator.jpg')).convert()
            self.img_predator = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'prey.jpg')).convert()
            self.img_prey = pygame.transform.scale(img, (WIDTH, WIDTH))

            if self.recorder_flag:
                self.snaps_path = os.path.join(current_path, 'results_predators_prey')  # The resource folder path
                self.snaps_path = os.path.join(self.snaps_path, 'snaps')  # The resource folder path

        self.cells = []
        self.agents_positions_idx = []

        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]

        positions_idx = []

        if self.game_mode == 0:
            # first enter the positions for the agents (predators) and the single prey. If the grid is n*n,
            # then the positions are
            #  0                1             2     ...     n-1
            #  n              n+1           n+2     ...    2n-1
            # 2n             2n+1          2n+2     ...    3n-1
            #  .                .             .       .       .
            #  .                .             .       .       .
            #  .                .             .       .       .
            # (n-1)*n   (n-1)*n+1     (n-1)*n+2     ...   n*n+1
            # , e.g.,
            # positions_idx = [0, 6, 23, 24] where 0, 6, and 23 are the positions of the agents 24 is the position
            # of the prey
            positions_idx = []

        if self.game_mode == 1:
            positions_idx = np.random.choice(len(cells), size=self.num_predators + self.num_preys, replace=False)

        return [cells, positions_idx]

    def reset(self):  # initialize the world
        self.terminal = False
        self.num_catches = 0

        [self.cells, self.agents_positions_idx] = self.set_positions_idx()

        # separate the generated position indices for walls, predators, and preys
        walls_positions_idx = self.agents_positions_idx[0:self.num_walls]
        predators_positions_idx = self.agents_positions_idx[self.num_walls:self.num_walls + self.num_predators]
        preys_positions_idx = self.agents_positions_idx[self.num_walls + self.num_predators:]

        # map generated position indices to positions
        self.walls_positions = [self.cells[pos] for pos in walls_positions_idx]
        self.predators_positions = [self.cells[pos] for pos in predators_positions_idx]
        self.preys_positions = [self.cells[pos] for pos in preys_positions_idx]

        initial_state = list(sum(self.walls_positions + self.predators_positions + self.preys_positions, ()))

        return initial_state

    def fix_prey(self):
        return 4

    def actor_prey_random(self):
        return random.randrange(self.action_space())

    def actor_prey_random_escape(self, prey_index):
        prey_pos = self.preys_positions[prey_index]
        [_, action_to_neighbors] = self.empty_neighbor_finder(prey_pos)

        return random.choice(action_to_neighbors)

    def neighbor_finder(self, pos):
        neighbors_pos = []
        action_to_neighbor = []
        pos_repeat = [pos for _ in xrange(4)]
        for idx in xrange(4):
            neighbor_pos = map(operator.add, pos_repeat[idx], self.A_DIFF[idx])
            if neighbor_pos[0] in range(0,self.grid_size) and neighbor_pos[1] in range(0,self.grid_size)\
                    and neighbor_pos not in self.walls_positions:
                neighbors_pos.append(neighbor_pos)
                action_to_neighbor.append(idx)

        neighbors_pos.append(pos)
        action_to_neighbor.append(4)

        return [neighbors_pos, action_to_neighbor]

    def empty_neighbor_finder(self, pos):
        neighbors_pos = []
        action_to_neighbor = []
        pos_repeat = [pos for _ in xrange(4)]
        for idx in xrange(4):
            neighbor_pos = map(operator.add, pos_repeat[idx], self.A_DIFF[idx])
            if neighbor_pos[0] in range(0,self.grid_size) and neighbor_pos[1] in range(0, self.grid_size)\
                    and neighbor_pos not in self.walls_positions:
                neighbors_pos.append(neighbor_pos)
                action_to_neighbor.append(idx)

        neighbors_pos.append(pos)
        action_to_neighbor.append(4)

        empty_neighbors_pos = []
        action_to_empty_neighbor = []

        for idx in xrange(len(neighbors_pos)):
            if tuple(neighbors_pos[idx]) not in self.predators_positions:
                empty_neighbors_pos.append(neighbors_pos[idx])
                action_to_empty_neighbor.append(action_to_neighbor[idx])

        return [empty_neighbors_pos, action_to_empty_neighbor]

    def step(self, predators_actions):
        # update the position of preys
        preys_actions = []
        for prey_idx in xrange(len(self.preys_positions)):
            if self.preys_mode == 0:
                preys_actions.append(self.fix_prey())
            elif self.preys_mode == 1:
                preys_actions.append(self.actor_prey_random_escape(prey_idx))
            elif self.preys_mode == 2:
                preys_actions.append(self.actor_prey_random())
            else:
                print('Invalid mode for the prey')

        self.preys_positions = self.update_positions(self.preys_positions, preys_actions)
        # update the position of predators
        self.predators_positions = self.update_positions(self.predators_positions, predators_actions)
        # check whether any predator catches any prey
        [reward, self.terminal] = self.check_catching()
        new_state = list(sum(self.walls_positions + self.predators_positions + self.preys_positions,()))

        return [new_state, reward, self.terminal]

    def check_catching(self):
        new_preys_position = []
        terminal_flag = False
        # checks to see whether the position of any prey is the same of as the position of any predator

        if self.reward_mode == 0:

            for prey_pos in self.preys_positions:
                new_preys_position.append(prey_pos)

            distances = 0
            for predator in self.predators_positions:
                distances += np.linalg.norm(np.array(predator) - np.array(self.preys_positions[0]), 1)

            [prey_empty_neigbours, _] = self.empty_neighbor_finder(self.preys_positions[0])

            # check the terminal case
            if int(distances) == self.num_predators - 1 or len(prey_empty_neigbours) == 0:
                terminal_flag = True
                reward = 0

            else:
                reward = -1

        elif self.reward_mode == 1:

            for prey_pos in self.preys_positions:
                new_preys_position.append(prey_pos)

            distances = 0
            for predator in self.predators_positions:
                distances += np.linalg.norm(np.array(predator) - np.array(self.preys_positions[0]), 1)

            [prey_empty_neigbours, _] = self.empty_neighbor_finder(self.preys_positions[0])

            # check the terminal case
            if int(distances) == self.num_predators - 1 or len(prey_empty_neigbours) == 0:
                terminal_flag = True
                reward = 0

            else:
                reward = -1 * distances

        else:
            print('Invalid game mode')

        self.preys_positions = new_preys_position

        return [reward, terminal_flag]

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
            elif positions_action_applied[pos_idx] not in pos_list and positions_action_applied[pos_idx] not in positions_action_applied[
                                                                                          0:pos_idx] + positions_action_applied[
                                                                                                       pos_idx + 1:]:
                final_positions.append(positions_action_applied[pos_idx])
            else:
                final_positions.append(pos_list[pos_idx])

        return final_positions

    def action_space(self):
        return len(self.A)

    def render(self):

        pygame.time.wait(500)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill(BLACK)
        text = self.my_font.render("Step: {0}".format(self.step_num), 1, WHITE)
        self.screen.blit(text, (5, 15))

        # for row in range(self.grid_size):
        #     for column in range(self.grid_size):
        #         pos = (row, column)
        #         if pos in self.predators_positions and pos in self.preys_positions:
        #             color = ORANGE
        #         elif pos in self.predators_positions:
        #             color = BLUE
        #         elif pos in self.preys_positions:
        #             color = RED
        #         else:
        #             color = WHITE
        #         pygame.draw.rect(self.screen, color,
        #                          [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
        #                           HEIGHT])

        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (row, column)
                if pos in self.predators_positions and pos in self.preys_positions:
                    self.screen.blit(self.img_predator_prey,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                elif pos in self.predators_positions:
                    self.screen.blit(self.img_predator,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                elif pos in self.preys_positions:
                    self.screen.blit(self.img_prey,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                else:
                    color = WHITE
                    pygame.draw.rect(self.screen, color,
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

        window_size_x = int(board_size_x*1.01)
        window_size_y = int(board_size_y * 1.2)

        window_size = [window_size_x, window_size_y]
        screen = pygame.display.set_mode(window_size, 0, 32)

        # Set title of screen
        pygame.display.set_caption("Predators-and-Prey Game")

        myfont = pygame.font.SysFont("monospace", 30)

        return [screen, myfont]
