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
        self.SPEED = 3000  # adjust the self.SPEED of the snake to your liking

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
        self.prev_head = self.head



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
        self.prev_head = self.head


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

        self.prev_head = self.Point(self.head.x, self.head.y) #store prev head position 

        for event in pygame.event.get(): #handle events
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        #move
        self._move(action)
        self.snake.insert(0, self.head)
        
        #initialize reward
        reward = 0
        game_over = False
        
        # check for collision or timeout
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # check for food
        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()
        
            # ADDED CHANGE1: proximity reward
            # calculate distance
            prev_dist = abs(self.prev_head.x - self.food.x) + abs(self.prev_head.y - self.food.y)
            curr_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

            if curr_dist < prev_dist:
                reward = reward + 0.2  # small positive reward for moving closer to food
            else:
                reward = reward - 0.05  # small negative reward for moving away to food
        
        # ADDED CHANGE2: Check proximity to walls - add small penalty for being close to walls
        wall_danger = False
        if self.head.x <= self.BLOCK_SIZE or self.head.x >= self.w - 2*self.BLOCK_SIZE:
            wall_danger = True
        if self.head.y <= self.BLOCK_SIZE or self.head.y >= self.h - 2*self.BLOCK_SIZE:
            wall_danger = True
    
        if wall_danger:
            reward = reward - 0.3  # Small penalty for being close to walls

        # ADDED CHANGE2: Check proximity to self - add small penalty for potential self-collisions
        for block in self.snake[3:]:  # skip the first few blocks
            if abs(self.head.x - block.x) <= 2 * self.BLOCK_SIZE and abs(self.head.y - block.y) <= 2 * self.BLOCK_SIZE:
                reward = reward - 0.3  # small penalty for being close to body
                break
            
        # ADDED CHANGE3: small time penalty to encourage efficiency
        reward = reward - 0.005  # very small penalty for each step
        
        #ADDED CHANGE4: flood fill penalty
        total_spaces = (self.w // self.BLOCK_SIZE) * (self.h // self.BLOCK_SIZE) - len(self.snake)
        free_spaces = self._count_free_spaces(self.head)
        free_space_percentage = free_spaces / total_spaces

        #signif penalty if snake trapped self
        if free_space_percentage < 0.5: #less than hald the board is accessible
            reward = reward - (0.5 - free_space_percentage) * 2.0 #scales with severity
            
        if free_space_percentage <0.3:
            reward = reward - 1.0 #penalize heavilt if snake has limited its movement
        

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
        
    def _count_free_spaces(self, start_point):
    
        if self.is_collision(start_point):
            return 0
        
        visited = set()
        queue = [start_point]
    
        while queue:
            current = queue.pop(0)
        
            # Skip if already visited
            if (current.x, current.y) in visited:
                continue
            
            visited.add((current.x, current.y))
        
            # Check all four adjacent cells
            for dx, dy in [(self.BLOCK_SIZE, 0), (-self.BLOCK_SIZE, 0), 
                        (0, self.BLOCK_SIZE), (0, -self.BLOCK_SIZE)]:
                next_point = self.Point(current.x + dx, current.y + dy)
            
                # Check if valid space
                if (0 <= next_point.x < self.w and 
                    0 <= next_point.y < self.h and 
                    not self.is_collision(next_point)):
                    queue.append(next_point)
    
        return len(visited)


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
        
        # ADDED CHANGE4: free space calculation for possible moves (flood fill)
        total_spaces = (self.w // self.BLOCK_SIZE) * (self.h // self.BLOCK_SIZE) - len(self.snake)
        # calculate current free space percentage
        current_free_spaces = self._count_free_spaces(self.head) / total_spaces
    
        # simulate each possible move and check resulting free spaces
        temp_snake = self.snake.copy()
        if len(temp_snake) > 0:
            temp_snake.pop(0) # temporarily remove head from snake to simulate moves
    
        # simulate straight move
        straight_pt = None
        if dir_r:
            straight_pt = self.Point(head.x + self.BLOCK_SIZE, head.y)
        elif dir_l:
            straight_pt = self.Point(head.x - self.BLOCK_SIZE, head.y)
        elif dir_u:
            straight_pt = self.Point(head.x, head.y - self.BLOCK_SIZE)
        elif dir_d:
            straight_pt = self.Point(head.x, head.y + self.BLOCK_SIZE)
    
        # if straight is dangerous, set free space to 0
        if self.is_collision(straight_pt):
            free_straight = 0
        else:
            # temporarily add new head
            orig_snake = self.snake
            self.snake = [straight_pt] + temp_snake
            free_straight = self._count_free_spaces(straight_pt) / total_spaces
            self.snake = orig_snake  # restore original snake
    
        # repeat for right turn
        right_pt = None
        if dir_u:
            right_pt = self.Point(head.x + self.BLOCK_SIZE, head.y)
        elif dir_d:
            right_pt = self.Point(head.x - self.BLOCK_SIZE, head.y)
        elif dir_l:
            right_pt = self.Point(head.x, head.y - self.BLOCK_SIZE)
        elif dir_r:
            right_pt = self.Point(head.x, head.y + self.BLOCK_SIZE)
    
        if self.is_collision(right_pt):
            free_right = 0
        else:
            # temporarily add new head
            orig_snake = self.snake
            self.snake = [right_pt] + temp_snake
            free_right = self._count_free_spaces(right_pt) / total_spaces
            self.snake = orig_snake  # restore original snake
    
        # repeat for left turn
        left_pt = None
        if dir_u:
            left_pt = self.Point(head.x - self.BLOCK_SIZE, head.y)
        elif dir_d:
            left_pt = self.Point(head.x + self.BLOCK_SIZE, head.y)
        elif dir_l:
            left_pt = self.Point(head.x, head.y + self.BLOCK_SIZE)
        elif dir_r:
            left_pt = self.Point(head.x, head.y - self.BLOCK_SIZE)
    
        if self.is_collision(left_pt):
            free_left = 0
        else:
            # temporarily add new head
            orig_snake = self.snake
            self.snake = [left_pt] + temp_snake
            free_left = self._count_free_spaces(left_pt) / total_spaces
            self.snake = orig_snake  # restore original snake
    
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
            int(food_down),        # 11
            
            current_free_spaces,   # 12 (NEW)
            free_straight,         # 13 (NEW)
            free_right,            # 14 (NEW)
            free_left              # 15 (NEW)
        ]

        return np.array(state, dtype=float)  # changed to float to handle continuous values

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


