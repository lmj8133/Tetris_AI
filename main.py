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
WIDTH, HEIGHT = 10, 1
SCREEN_SIZE = (300, 30)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
BLOCK_SIZE = SCREEN_WIDTH // WIDTH
FPS = 30
GAME_SPEED = 1000
MODE_HOLLOW = 0
MODE_BLOCK = 1
RETARDED_CONSTANT = 25
LEARNING_RATE = 0.4
DISCOUNT_FACTOR = 0.4
EXPLORATION_RATE = 0.06
    #def __init__(self, learning_rate=0.7, discount_factor=0.4, exploration_rate=0.06):

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
                #red         #green       #blue
BLOCK_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                #yellow        #orange        #purple        #aqua
                (255, 255, 0), (255, 165, 0), (128, 0, 128), (0, 255, 255)]

I_SHAPE = [[1, 1, 1, 1]]
O_SHAPE = [[1, 1], [1, 1]]
T_SHAPE = [[0, 1, 0], [1, 1, 1]]
J_SHAPE = [[1, 0, 0], [1, 1, 1]]
L_SHAPE = [[0, 0, 1], [1, 1, 1]]
Z_SHAPE = [[1, 1, 0], [0, 1, 1]]
S_SHAPE = [[0, 1, 1], [1, 1, 0]]

I_SHAPE = [[1]] 
O_SHAPE = [[1]] 
T_SHAPE = [[1]] 
J_SHAPE = [[1]] 
L_SHAPE = [[1]] 
Z_SHAPE = [[1]] 
S_SHAPE = [[1]] 

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
        self.reward = 1
        self.clear_line = 0
        self.retarded = 0
        self.upcoming_pieces = collections.deque(maxlen=7)
        #self.initialize_upcoming_pieces()


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
                    self.color_board[y + self.current_piece['y']][x + self.current_piece['x']] = self.current_piece['color']

        self.clear_lines()
        self.discount_factor = self.discount_factor * 0.95
        self.current_piece = self.new_piece()

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

    def draw(self, screen):
        screen.fill(BLACK)
        self.draw_grid(screen)
        #self.draw_upcoming_pieces(screen)

        # Draw filled blocks
        for y, row in enumerate(self.board):
            for x, value in enumerate(row):
                #if value:
                pygame.draw.rect(screen, random.choice(BLOCK_COLORS), (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    #pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)  # Draw black border

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

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class TetrisAIWithANN(Tetris):
    def __init__(self, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR, exploration_rate=EXPLORATION_RATE):
        super().__init__()

        # Define neural network parameters
        input_size = WIDTH * HEIGHT + 2
        output_size = 5  # Number of possible actions (rotate, move_left, move_right, hard_drop, do_nothing)
        self.q_network = QNetwork(input_size, output_size)
        #self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization
        #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.9, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        #torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9, last_epoch=-1, verbose=True)
        self.criterion = nn.MSELoss()

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.replay_buffer_size = 5000  # or a size that fits your memory constraints
        self.replay_buffer = collections.deque(maxlen=self.replay_buffer_size)
        #self.target_update_frequency = 200  # Adjust the frequency based on your training needs
        self.target_update_frequency = 100  # Adjust the frequency
        self.target_q_network = QNetwork(input_size, output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.step_count = 0
        self.highest_row = 0
        self.lowest_row = 0
        self.height = 0


    def calculate_reward(self):
        lines_cleared = self.clear_line
        hollows = self.calculate_hollow(HEIGHT, 1, WIDTH, MODE_HOLLOW)

        #reward = 100 * (2 ** lines_cleared) - 50 * hollows
        reward = 500 * (2 ** lines_cleared)
        return reward

    def calculate_parity(self, height, start_width, end_width):
        parity = 1
        for x in range(start_width - 1, end_width):
            for y in range(HEIGHT - height, HEIGHT):
                if self.board[y][x] == 1:
                    if (y + x) % 2 == 1:
                        parity *= -1
        return parity


    def calculate_hollow(self, height, start_width, end_width, mode):
        #print(f"S: {WIDTH - start_width}, E: {WIDTH - end_width}")
        hollow = 0
        for x in range(WIDTH - end_width, WIDTH - start_width + 1):
            for y in range(HEIGHT - height, HEIGHT):
                #print(f"({x}, {y}) = {self.board[y][x]}")
                if self.board[y][x] == mode:
                    hollow += y
        #print('---')
        return hollow

    def get_height(self):
        height = HEIGHT
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.board[y][x] == 1 and height > y:
                    height = y
        return HEIGHT - height

    def get_width(self):
        end_width = WIDTH
        start_width = 0
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if self.board[y][x] == 1:
                    if end_width > x:
                        end_width = x
                    elif start_width < x:
                        start_width = x
        return [WIDTH - start_width, WIDTH - end_width]

    def update_replay_buffer(self, state_key, action, reward, new_state_key, done):
        self.replay_buffer.append((state_key, action, reward, new_state_key, done))

    def sample_from_replay_buffer(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return zip(*self.replay_buffer)

        samples = random.sample(self.replay_buffer, batch_size)
        return map(list, zip(*samples))

    def update_target_network(self):
        if self.step_count % self.target_update_frequency == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def decay_exploration_rate(self, episode):
        if self.retarded > RETARDED_CONSTANT:
            self.exploration_rate = 0.1
            self.retarded = 0
            self.learning_rate *= 0.9
        else:
            self.exploration_rate = max(0.01, self.exploration_rate * 0.995)

    def state_key(self):
        # Convert the board state to a flattened numpy array
        board_state = np.array(self.board).flatten()

        # Include the number of cleared lines as part of the state
        lines_cleared_state = np.array([self.clear_line])

        # Convert the name of the current piece to an integer
        current_piece_state = np.array([PIECE_NAME_TO_INT[self.current_piece['name']]])

        # Combine the board state and the lines cleared into a single state representation
        combined_state = np.concatenate([board_state, lines_cleared_state, current_piece_state])
        return combined_state

    #def state_key(self):
    #    # Convert the board state to a flattened numpy array
    #    board_state = np.array(self.board).flatten()

    #    # Include the number of cleared lines as part of the state
    #    lines_cleared_state = np.array([self.clear_line])
    #    current_piece_state = np.array([self.current_piece['name']])

    #    # Combine the board state and the lines cleared into a single state representation
    #    combined_state = np.concatenate([board_state, lines_cleared_state, current_piece_state])
    #    return combined_state

#    def state_key(self):
#        # Convert the board state to a flattened numpy array
#        return np.array(self.board).flatten()

    def choose_action(self):
        state_key = torch.tensor(self.state_key(), dtype=torch.float32).unsqueeze(0)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(["rotate", "move_left", "move_right", "hard_drop", "do_nothing"])
            #return random.choice(["move_left", "move_right"])
        else:
            with torch.no_grad():
                q_values = self.q_network(state_key)
            return max(zip(["rotate", "move_left", "move_right", "hard_drop", "do_nothing"], q_values[0]), key=lambda x: x[1])[0]
            #return max(zip(["move_left", "move_right"], q_values[0]), key=lambda x: x[1])[0]

    def update_q_network(self, action, reward, new_state_key):
        state_key = torch.tensor(self.state_key(), dtype=torch.float32).unsqueeze(0)
        new_state_key = torch.tensor(new_state_key, dtype=torch.float32).unsqueeze(0)

        q_values = self.q_network(state_key)
        new_q_values = self.target_q_network(new_state_key)

        max_future_q, _ = torch.max(new_q_values, dim=1)
        current_q = q_values[0, ["rotate", "move_left", "move_right", "hard_drop", "do_nothing"].index(action)]
        #current_q = q_values[0, ["move_left", "move_right"].index(action)]


        target_q = q_values.clone()
        target_q[0, ["rotate", "move_left", "move_right", "hard_drop", "do_nothing"].index(action)] = (
                1 - self.learning_rate
        #target_q[0, ["move_left", "move_right"].index(action)] = (
        #        1 - self.learning_rate
        ) * current_q.item() + self.learning_rate * (reward + self.discount_factor * max_future_q.item())

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.update_target_network()

# Inside the update function of the TetrisAIWithANN class
    def update(self):
        action = self.choose_action()

        if action == "rotate":
            self.rotate_piece()
        elif action == "move_left":
            self.move_piece(-1, 0)
        elif action == "move_right":
            self.move_piece(1, 0)
        elif action == "hard_drop":
            offset = 0
            while not self.collide(self.current_piece, offset=(0, offset + 1)):
                offset += 1
            # Move the piece to the lowest possible position instantly
            self.move_piece(0, offset)

        new_state_key = self.state_key()

        if self.collide(self.current_piece, offset=(0, 1)):
            self.merge_piece()
            self.height = self.get_height()
            width = self.get_width()
            if (self.height < 20):
                if (self.height > self.highest_row):
                    self.highest_row = self.height
                    self.reward -= 100
                    self.hollow = self.calculate_hollow(self.highest_row, width[0], width[1], MODE_HOLLOW)
                    self.reward -= self.hollow
                else:
                    self.reward += 200
                    self.hollow = self.calculate_hollow(self.highest_row, width[0], width[1], MODE_BLOCK)
                    self.reward += self.hollow
                if self.calculate_parity(self.highest_row, width[0], width[1]) == -1:
                    self.reward -= 50
        else:
            self.move_piece(0, 1)

        self.update_replay_buffer(
            torch.tensor(self.state_key(), dtype=torch.float32).unsqueeze(0),
            action,
            self.reward,
            new_state_key,
            self.is_game_over(),
        )

        if len(self.replay_buffer) >= 50:
            # Train the Q-network with a batch of experiences from the replay buffer
            batch_size = 64
            states, actions, rewards, new_states, dones = self.sample_from_replay_buffer(batch_size)

            # Add a check to ensure the lengths are consistent
            if len(actions) == len(rewards) == len(new_states):
                for i in range(len(actions)):
                    self.update_q_network(actions[i], rewards[i], new_states[i])

    def reset(self, episode):
        #print("++++++++++++++")
        self.decay_exploration_rate(episode)
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        #self.counter = 0
        self.reward = 1
        self.highest_row = 0
        self.height = 0
        self.clear_line = 0
        self.srs_index = 0
        #self.srs_array_index = 0
        self.initialize_upcoming_pieces()
        self.current_piece = self.new_piece()
        self.discount_factor = DISCOUNT_FACTOR

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

    def get_reward(self):
        # You can define your scoring mechanism based on the number of cleared lines, etc.
        # For simplicity, let's use the number of lines cleared as the reward.
        #return sum(1 for row in self.board if all(row))
        return self.reward

def main(train_episodes=1000000):
    pygame.init()
    #screen = pygame.display.set_mode(SCREEN_SIZE)
    screen = pygame.display.set_mode((SCREEN_WIDTH + 200, SCREEN_HEIGHT))  # Adjust the width to make room for upcoming pieces
    pygame.display.set_caption('Press P to Next Round')
    #reward = 9898
    reward = 1
    clear_line = 0
    episode_offset = 0
    early_stop_counter = 0
    episode = 0

    clock = pygame.time.Clock()
    tetris = TetrisAIWithANN()
    tetris.reset(episode)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    tetris.reset(episode)
                    # continue to the next iteration of the loop to restart the game
                    continue

        tetris.update()
        tetris.draw(screen)
        pygame.display.flip()
        clock.tick(GAME_SPEED)

        if tetris.is_game_over():
            tetris.pause_game()
            tetris.reset(episode)

if __name__ == '__main__':
    main()
