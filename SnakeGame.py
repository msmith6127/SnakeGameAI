import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4




class SnakeGame:

    def __init__(self, w=640, h=480):  # dimensions

        #  define constant values for game such
        self.Point = namedtuple('Point', 'x, y')

        self.WHITE = (255, 255, 255)
        self.RED = (200, 0, 0)
        self.BLUE1 = (0, 0, 255)
        self.BLUE2 = (0, 100, 255)
        self.BLACK = (0, 0, 0)

        # define game parameters for speed and size
        self.BLOCK_SIZE = 20
        self.SPEED = 2  # adjust the self.SPEED of the snake to your liking

        # display parameters
        self.w = w
        self.h = h


        # define game parameters for start of game, current game state at start of play through
        self.direction = Direction.RIGHT
        self.head = self.Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      self.Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      self.Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0



        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    
    
    


    def _update_ui(self):
            self.display.fill(self.BLACK)
            for pt in self.snake:
                pygame.draw.rect(self.display, self.BLUE1, pygame.Rect(pt.x, pt.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
                pygame.draw.rect(self.display, self.BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

            pygame.draw.rect(self.display, self.RED, pygame.Rect(self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE))

            text = font.render("Score: " + str(self.score), True, self.WHITE)
            self.display.blit(text, [0, 0])
            pygame.display.flip()


    def reset(self):  # game state
        self.direction = Direction.RIGHT
        self.head = self.Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      self.Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      self.Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
      x = random.randint(0, (self.w - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
      y = random.randint(0, (self.h - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
      self.food = self.Point(x, y)
      if self.food in self.snake:
            self._place_food()


    def is_collision(self, pt=None):
        if pt is None: #pt is the head of the snake
            pt = self.head
        if pt.x > self.w - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - self.BLOCK_SIZE or pt.y < 0:
            return True #if snake hits the side
        if pt in self.snake[1:]:
            return True  #if snake hits itself
        return False

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(self.SPEED)
        return reward, game_over, self.score

    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0,0,1] aka left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = self.Point(x, y)


#function numerically represents the current game state
    def get_state(self):
        head = self.head
        point_l = self.Point(head.x - self.BLOCK_SIZE, head.y)
        point_r = self.Point(head.x + self.BLOCK_SIZE, head.y)
        point_u = self.Point(head.x, head.y - self.BLOCK_SIZE)
        point_d = self.Point(head.x, head.y + self.BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Check danger straight, right, left
        danger_straight = (
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d))
        )

        danger_right = (
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d))
        )

        danger_left = (
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d))
        )

        # Food location relative to head
        food_left = self.food.x < self.head.x
        food_right = self.food.x > self.head.x
        food_up = self.food.y < self.head.y
        food_down = self.food.y > self.head.y

        state = [
            int(danger_straight),  # 1
            int(danger_right),     # 2
            int(danger_left),      # 3

            int(dir_l),            # 4
            int(dir_r),            # 5
            int(dir_u),            # 6
            int(dir_d),            # 7

            int(food_left),        # 8
            int(food_right),       # 9
            int(food_up),          # 10
            int(food_down)         # 11
        ]

        return np.array(state, dtype=int)

if __name__ == '__main__':


    # running game for test
    # initialize starting game
    test_game = SnakeGame()



    # loop to run game manually
    while True:

        # choose next direction [forward, right, left]
        next_action = [1, 0, 0]


        # check player input for next action
        for curr_event in pygame.event.get():
            # Quit the game
            if curr_event.type == pygame.QUIT:
                running = False

            # Key pressed
            if curr_event.type == pygame.KEYDOWN:

                # right turn
                if curr_event.key == pygame.K_d:
                    next_action = [0, 1, 0]

                # left turn
                elif curr_event.key == pygame.K_a:
                    next_action = [0, 0, 1]

                    # straight turn if no key pressed
                elif curr_event.key == pygame.K_w:
                    next_action = [1, 0, 0]




        # play game with current action, save current state as a result
        curr_reward, curr_game_over, curr_score = test_game.play_step(action=next_action)
        print(test_game.get_state())


        # if game over end program
        if curr_game_over:
            print("Git Gud")
            break


