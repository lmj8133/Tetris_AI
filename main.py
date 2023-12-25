import pygame
import random

# Tetris constants
WIDTH, HEIGHT = 10, 20
SCREEN_SIZE = (300, 600)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
BLOCK_SIZE = SCREEN_WIDTH // WIDTH
FPS = 30

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

m = [-4, -2, 0, 2, 4]

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
                #black     #red         #green       #blue
BLOCK_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                #yellow        #orange        #purple        #aqua
                (255, 255, 0), (255, 165, 0), (128, 0, 128), (0, 255, 255)]

class Tetris:
    def __init__(self):
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.current_piece = self.new_piece()
        self.counter = 0

    def new_piece(self):
        shape = random.choice(SHAPES)
        #shape = SHAPES[1]
        self.state = 0
        piece = {'shape': shape, 'x': WIDTH // 2 - len(shape[0]) // 2, 'y': 0}
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

    def rotate_piece(self):
        new_shape = list(zip(*reversed(self.current_piece['shape'])))
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

    def merge_piece(self):
        for y, row in enumerate(self.current_piece['shape']):
            for x, value in enumerate(row):
                if value:
                    self.board[y + self.current_piece['y']][x + self.current_piece['x']] = 1

        self.clear_lines()
        self.current_piece = self.new_piece()

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.board) if all(row)]
        for line in lines_to_clear:
            del self.board[line]
            self.board.insert(0, [0] * WIDTH)

    def update(self):
        # Rotate the piece if possible
        #self.rotate_piece()
        if self.state == 0:
            self.move_piece(m[self.counter], 0)
            self.counter += 1
            if self.counter > 4:
                self.counter = 0
            self.state = 1

    def draw(self, screen):
        screen.fill(BLACK)
        self.draw_grid(screen)

        # Draw filled blocks
        for y, row in enumerate(self.board):
            for x, value in enumerate(row):
                if value:
                    pygame.draw.rect(screen, WHITE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)  # Draw black border

        # Draw falling piece
        current_piece_shape = self.current_piece['shape']
        for y, row in enumerate(current_piece_shape):
            for x, value in enumerate(row):
                if value:
                    pygame.draw.rect(screen, WHITE, (
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

class TetrisAI(Tetris):
    def update(self):
        # Randomly rotate or move the piece
        action = random.choice(["rotate", "move_left", "move_right"])
        if action == "rotate":
            self.rotate_piece()
        elif action == "move_left":
            self.move_piece(-1, 0)
        elif action == "move_right":
            self.move_piece(1, 0)

        # Move the piece downward
        if self.collide(self.current_piece, offset=(0, 1)):
            self.merge_piece()
        else:
            self.move_piece(0, 1)

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Tetris AI')

    clock = pygame.time.Clock()
    tetris = TetrisAI()  # Use the TetrisAI class instead

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        tetris.update()
        tetris.draw(screen)
        pygame.display.flip()
        clock.tick(5)  # Adjust the speed of the game

if __name__ == '__main__':
    main()
