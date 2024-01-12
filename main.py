import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pickle
import collections

# Tetris constants
WIDTH, HEIGHT = 10, 20
SCREEN_SIZE = (300, 600)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
BLOCK_SIZE = SCREEN_WIDTH // WIDTH
# Constants for the sidebars
#SIDE_BAR_WIDTH = 5 * BLOCK_SIZE  # Assuming each sidebar is 5 blocks wide
#SCREEN_WIDTH = (SCREEN_SIZE[0] + SIDE_BAR_WIDTH)
# Update the screen size to include sidebars
#SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
FPS = 30
GAME_SPEED = 60
MODE_HOLLOW = 0
MODE_BLOCK = 1
RETARDED_CONSTANT = 25
LEARNING_RATE = 0.4
DISCOUNT_FACTOR = 0.4
EXPLORATION_RATE = 0.06

# Tetris pieces
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]
#SHAPES = [
#    [[1, 1, 1, 1]],
#    [[1, 1], [1, 1]]
#]

m = [-4, -2, 0, 2, 4]

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
                #black     #red         #green       #blue
BLOCK_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                #yellow        #orange        #purple        #aqua
                (255, 255, 0), (255, 165, 0), (128, 0, 128), (0, 255, 255)]

I_SHAPE = [[1, 1, 1, 1]]
O_SHAPE = [[1, 1], [1, 1]]
T_SHAPE = [[0, 1, 0], [1, 1, 1]]
J_SHAPE = [[1, 0, 0], [1, 1, 1]]
L_SHAPE = [[0, 0, 1], [1, 1, 1]]
Z_SHAPE = [[1, 1, 0], [0, 1, 1]]
S_SHAPE = [[0, 1, 1], [1, 1, 0]]

PIECE_NAME_TO_INT = {
    'I': 0,
    'O': 1,
    'T': 2,
    'J': 3,
    'L': 4,
    'S': 5,
    'Z': 6
}
# Add color constants for each shape
AQUA = (0, 255, 255)    # I shape
YELLOW = (255, 255, 0)   # O shape
PURPLE = (128, 0, 128)   # T shape
BLUE = (0, 0, 255)       # J shape
RED = (255, 0, 0)        # L shape
GREEN = (0, 255, 0)      # S shape
ORANGE = (255, 165, 0)   # Z shape
#AQUA = WHITE    # I shape
#YELLOW = WHITE   # O shape
#PURPLE = WHITE   # T shape
#BLUE = WHITE       # J shape
#RED = WHITE        # L shape
#GREEN = WHITE      # S shape
#ORANGE = WHITE   # Z shape

# Colors for each shape
SHAPE_COLORS = [AQUA, YELLOW, PURPLE, BLUE, RED, GREEN, ORANGE]

class Piece:
    def __init__(self, name, shape, color):
        self.name = name
        self.shape = shape
        self.color = color

class Tetris:
    def __init__(self):
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.color_board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.srs_index = 0
        self.srs_array_index = 0
        self.I = Piece('I', I_SHAPE, AQUA)
        self.O = Piece('O', O_SHAPE, YELLOW)
        self.T = Piece('T', T_SHAPE, PURPLE)
        self.J = Piece('J', J_SHAPE, BLUE)
        self.L = Piece('L', L_SHAPE, ORANGE)
        self.S = Piece('S', S_SHAPE, GREEN)
        self.Z = Piece('Z', Z_SHAPE, RED)
        self.shapes_array = [[self.I, self.O, self.T, self.J, self.L, self.S, self.Z], [self.I, self.O, self.T, self.J, self.L, self.S, self.Z]]
        #self.current_piece = self.new_piece()
        self.upcoming_pieces = collections.deque(maxlen=7)
        #self.initialize_upcoming_pieces()
        self.das_direction = None
        self.das_timer = 0
        self.falling_timer = 0
        self.held_piece = None
        self.can_hold = True  # Flag to check if holding a piece is allowed

    def initialize_upcoming_pieces(self):
        random.shuffle(self.shapes_array[self.srs_array_index])  # Shuffle in place without reassignment
        i = 0
        for _ in range(7):
            #print(f"idx{self.srs_array_index}: {self.shapes_array[self.srs_array_index][i].name}")
            #i += 1
            self.add_new_piece_to_upcoming()


    def add_new_piece_to_upcoming(self):
        piece = self.shapes_array[self.srs_array_index][self.srs_index]
        self.upcoming_pieces.append(piece)
        if self.srs_index == 6:
            self.srs_index = 0
            self.srs_array_index ^= 1
            random.shuffle(self.shapes_array[self.srs_array_index])  # Shuffle in place without reassignment
            #i = 0
            #for _ in range(7):
            #    print(f"idx{self.srs_array_index}: {self.shapes_array[self.srs_array_index][i].name}")
            #    i += 1
        else:
            self.srs_index += 1

    def new_piece(self):
        self.reset_hold()
        next_piece = self.upcoming_pieces.popleft()
        self.add_new_piece_to_upcoming()
        #print(next_piece.name)
        piece = {'shape': next_piece.shape, 'x': WIDTH // 2 - len(next_piece.shape[0]) // 2, 'y': 0, 'color': next_piece.color, 'name': next_piece.name}
        return piece

    def collide(self, piece, offset=(0, 0)):
        for y, row in enumerate(piece['shape']):
            for x, value in enumerate(row):
                if value and (
                        x + piece['x'] + offset[0] < 0 or
                        x + piece['x'] + offset[0] >= WIDTH or
                        y + piece['y'] + offset[1] >= HEIGHT or
                        self.board[y + piece['y'] + offset[1]][x + piece['x'] + offset[0]]):
                    return True
        return False

    def rotate_piece_clockwise(self):
        new_shape = list(zip(*reversed(self.current_piece['shape'])))
        temp_piece = {'shape': new_shape,
                      'x': self.current_piece['x'],
                      'y': self.current_piece['y']}
        if not self.collide(temp_piece):
            self.current_piece['shape'] = new_shape

    def rotate_piece_counterclockwise(self):
        new_shape = list(zip(*self.current_piece['shape']))[::-1]
        temp_piece = {'shape': new_shape,
                      'x': self.current_piece['x'],
                      'y': self.current_piece['y']}
        if not self.collide(temp_piece):
            self.current_piece['shape'] = new_shape

    def rotate_piece_180(self):
        # Flip the piece both horizontally and vertically
        new_shape = [row[::-1] for row in self.current_piece['shape'][::-1]]
        temp_piece = {'shape': new_shape,
                      'x': self.current_piece['x'],
                      'y': self.current_piece['y']}
        if not self.collide(temp_piece):
            self.current_piece['shape'] = new_shape

    def move_piece(self, dx, dy):
        if not self.collide(self.current_piece, offset=(dx, dy)):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
        else:
            if dy:
                self.merge_piece()
    def hard_drop(self):
        offset = 0
        while not self.collide(self.current_piece, offset=(0, offset + 1)):
            offset += 1
        # Move the piece to the lowest possible position instantly
        self.move_piece(0, offset)

    def hold_piece(self):
        if self.can_hold:
            if self.held_piece:
                # Swap current piece with held piece
                self.held_piece, self.current_piece = self.current_piece, self.held_piece
                self.current_piece['x'] = WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
                self.current_piece['y'] = 0
            else:
                # Hold the current piece and get a new one
                self.held_piece = self.current_piece
                self.current_piece = self.new_piece()

            self.can_hold = False  # Disable holding until the next piece

    def reset_hold(self):
        # Reset the hold ability when a new piece is locked in place
        self.can_hold = True

    def merge_piece(self):
        for y, row in enumerate(self.current_piece['shape']):
            for x, value in enumerate(row):
                if value:
                    self.board[y + self.current_piece['y']][x + self.current_piece['x']] = 1
                    self.color_board[y + self.current_piece['y']][x + self.current_piece['x']] = self.current_piece['color']
        self.clear_lines()
        self.current_piece = self.new_piece()

    def update_das(self):
        # Implement DAS logic
        if self.das_direction is not None:
            if self.das_timer == 10:
                self.move_piece(self.das_direction[0], self.das_direction[1])
            else:
                self.das_timer += 1

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.board) if all(row)]
        for line in lines_to_clear:
            del self.board[line]
            self.board.insert(0, [0] * WIDTH)
            self.clear_line += 1
            if self.reward < 0:
                self.reward *= -1

            self.reward += (3000 * (1.01 ** (self.clear_line)))

    def draw_upcoming_pieces(self, screen):
        x_start = SCREEN_WIDTH + 10  # Adjust as needed
        y_start = 10  # Adjust as needed
        gap = 100  # Gap between pieces, adjust as needed

        for i, piece in enumerate(self.upcoming_pieces):
            shape = piece.shape
            color = piece.color
            for y, row in enumerate(shape):
                for x, value in enumerate(row):
                    if value:
                        pygame.draw.rect(screen, color, (x_start + x * BLOCK_SIZE, y_start + i * gap + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                        pygame.draw.rect(screen, BLACK, (x_start + x * BLOCK_SIZE, y_start + i * gap + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    def draw_held_piece(self, screen):
        if self.held_piece:
            x_start = 10  # X position to start drawing the held piece
            y_start = 10  # Y position to start drawing the held piece
            for y, row in enumerate(self.held_piece['shape']):
                for x, value in enumerate(row):
                    if value:
                        pygame.draw.rect(screen, self.held_piece['color'],
                                         (x_start + x * BLOCK_SIZE, y_start + y * BLOCK_SIZE,
                                          BLOCK_SIZE, BLOCK_SIZE))
                        pygame.draw.rect(screen, BLACK,
                                         (x_start + x * BLOCK_SIZE, y_start + y * BLOCK_SIZE,
                                          BLOCK_SIZE, BLOCK_SIZE), 1)

    def draw_sidebar(self, screen):
        # Draw the sidebars as a background for "HOLD" and "NEXT" zones
        pygame.draw.rect(screen, BLACK, (0, 0, SIDE_BAR_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH - SIDE_BAR_WIDTH, 0, SIDE_BAR_WIDTH, SCREEN_HEIGHT))
        # Add labels for "HOLD" and "NEXT"
        font = pygame.font.Font(None, 36)
        hold_label = font.render('HOLD', True, WHITE)
        next_label = font.render('NEXT', True, WHITE)
        screen.blit(hold_label, (10, 10))  # Adjust coordinates as needed
        screen.blit(next_label, (SCREEN_WIDTH - SIDE_BAR_WIDTH + 10, 10))  # Adjust coordinates as needed

    def draw(self, screen):
        screen.fill(BLACK)
        #self.draw_sidebar(screen)
        self.draw_grid(screen)
        self.draw_upcoming_pieces(screen)
        self.draw_held_piece(screen)

        # Draw filled blocks
        for y, row in enumerate(self.board):
            for x, value in enumerate(row):
                if value:
                    pygame.draw.rect(screen, self.color_board[y][x], (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)  # Draw black border

        # Draw falling piece
        current_piece_shape = self.current_piece['shape']
        for y, row in enumerate(current_piece_shape):
            for x, value in enumerate(row):
                if value:
                    pygame.draw.rect(screen, self.current_piece['color'], (
                        (x + self.current_piece['x']) * BLOCK_SIZE, (y + self.current_piece['y']) * BLOCK_SIZE,
                        BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(screen, BLACK, (
                        (x + self.current_piece['x']) * BLOCK_SIZE, (y + self.current_piece['y']) * BLOCK_SIZE,
                        BLOCK_SIZE, BLOCK_SIZE), 1)  # Draw black border

    # Helper function to draw the grid
    def draw_grid(self, screen):
        for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
            pygame.draw.line(screen, WHITE, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
            pygame.draw.line(screen, WHITE, (0, y), (SCREEN_WIDTH, y))

    def update(self):
        if self.collide(self.current_piece, offset=(0, 1)):
            self.merge_piece()
        else:
            if (self.falling_timer == 30):
                self.move_piece(0, 1)
                self.falling_timer = 0
            else:
                self.falling_timer += 1

    def reset(self):
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.reward = 1
        self.highest_row = 0
        self.height = 0
        self.clear_line = 0
        self.srs_index = 0
        #self.srs_array_index = 0
        self.das_direction = None
        self.falling_timer = 0
        self.initialize_upcoming_pieces()
        self.current_piece = self.new_piece()
        self.held_piece = None

    def is_game_over(self):
        # The game is over if the new piece collides with existing blocks at the top
        return self.collide(self.current_piece, offset=(0, 0))

    def pause_game(self):
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        paused = False
            pygame.time.delay(100)  # Adjust the delay as needed

def main(train_episodes=1000000):
    pygame.init()
    #screen = pygame.display.set_mode(SCREEN_SIZE)
    screen = pygame.display.set_mode((SCREEN_WIDTH + 200, SCREEN_HEIGHT))  # Adjust the width to make room for upcoming pieces
    pygame.display.set_caption('Tetris AI with ANN')
    #reward = 9898
    reward = 1
    clear_line = 0
    episode_offset = 0
    early_stop_counter = 0

    clock = pygame.time.Clock()
    tetris = Tetris()

    tetris.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                # Restart
                if event.key == pygame.K_p:
                    tetris.reset()
                    # continue to the next iteration of the loop to restart the game
                    continue
            if event.type == pygame.KEYDOWN:
                # Move left
                if event.key == pygame.K_j:
                    tetris.move_piece(-1, 0)
                    tetris.das_direction = (-1, 0)
                # Move right
                if event.key == pygame.K_l:
                    tetris.move_piece(1, 0)
                    tetris.das_direction = (1, 0)
                # Rotate clockwise
                if event.key == pygame.K_f:
                    tetris.rotate_piece_clockwise()
                # Rotate counter-clockwise
                if event.key == pygame.K_s:
                    tetris.rotate_piece_counterclockwise()
                # Rotate 180 degree
                if event.key == pygame.K_d:
                    tetris.rotate_piece_180()
                # Soft drop
                if event.key == pygame.K_k:
                    tetris.move_piece(0, 1)
                    tetris.das_direction = (0, 1)
                # Hard drop
                if event.key == pygame.K_SPACE:
                    tetris.hard_drop()
                # Hold
                if event.key == pygame.K_i:
                    tetris.hold_piece()
            if event.type == pygame.KEYUP:
                if (event.key == pygame.K_j) or \
                   (event.key == pygame.K_l) or \
                   (event.key == pygame.K_k):
                    tetris.das_direction = None
                    tetris.falling_timer = 0
                    tetris.das_timer = 0

        tetris.update_das()
        tetris.update()
        tetris.draw(screen)
        pygame.display.flip()
        clock.tick(GAME_SPEED)

        if tetris.is_game_over():
            tetris.pause_game()
            tetris.reset()

    pygame.quit()

if __name__ == '__main__':
    main()
