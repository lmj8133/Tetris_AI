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
FPS = 30
GAME_SPEED = 100000

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

class Tetris:
    def __init__(self):
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.current_piece = self.new_piece()
        #self.counter = 0
        self.reward = 1
        self.clear_line = 0

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
            self.clear_line += 1
            if self.reward < 0:
                self.reward = 0

            self.reward += (200 * (1.01 ** (self.clear_line)))

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

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class TetrisAIWithANN(Tetris):
    def __init__(self, learning_rate=0.01, discount_factor=0.9, exploration_rate=0.1):
        super().__init__()

        # Define neural network parameters
        input_size = WIDTH * HEIGHT
        output_size = 5  # Number of possible actions (rotate, move_left, move_right, hard_drop, do_nothing)
        self.q_network = QNetwork(input_size, output_size)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.replay_buffer_size = 5000  # or a size that fits your memory constraints
        self.replay_buffer = collections.deque(maxlen=self.replay_buffer_size)
        self.target_update_frequency = 200  # Adjust the frequency based on your training needs
        self.target_q_network = QNetwork(input_size, output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.step_count = 0
        self.highest_row = 0
        self.lowest_row = 0
        self.height = 0

    def calculate_parity(self, height, start_width, end_width):
        parity = 1
        for x in range(start_width - 1, end_width):
            for y in range(HEIGHT - height, HEIGHT):
                if self.board[y][x] == 1:
                    if (y + x) % 2 == 1:
                        parity *= -1
        return parity


    def calculate_hollow(self, height, start_width, end_width):
        hollow = 0
        for x in range(start_width - 1, end_width):
            for y in range(HEIGHT - height, HEIGHT):
                if self.board[y][x] == 0:
                    hollow += 1
        return hollow

    def get_height(self):
        highest_height = HEIGHT
        lowest_height = 0
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.board[y][x] == 1 and highest_height > y:
                    highest_height = y
                if self.board[y][x] == 0 and lowest_height < y:
                    lowest_height = y
        return [HEIGHT - lowest_height, HEIGHT - highest_height]

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
        self.exploration_rate = max(0.1, 1.0 - 0.0001 * episode)  # Adjust the decay rate as needed

    def state_key(self):
        # Convert the board state to a flattened numpy array
        return np.array(self.board).flatten()

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
            if (self.height[1] > self.highest_row):
                self.highest_row = self.height[1]
                self.reward -= 20
            else:
                self.reward += 100
            if (self.height[0] > self.lowest_row):
                self.lowest_row = self.height[0]
                self.reward += 200
            else:
                self.reward -= 40
            hollow = self.calculate_hollow(self.highest_row, width[0], width[1])
            if self.calculate_parity(self.highest_row, width[0], width[1]) == -1:
                self.reward -= 50
            self.reward -= hollow
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
        self.decay_exploration_rate(episode)
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.current_piece = self.new_piece()
        #self.counter = 0
        self.reward = 1
        self.highest_row = 0
        self.height = 0
        self.clear_line = 0

    def is_game_over(self):
        # The game is over if the new piece collides with existing blocks at the top
        return self.collide(self.current_piece, offset=(0, 0))

    def get_reward(self):
        # You can define your scoring mechanism based on the number of cleared lines, etc.
        # For simplicity, let's use the number of lines cleared as the reward.
        #return sum(1 for row in self.board if all(row))
        return self.reward

def main(train_episodes=1000000):
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Tetris AI with ANN')
    reward = 1
    clear_line = 0

    clock = pygame.time.Clock()
    tetris = TetrisAIWithANN()

    # Load pre-trained weights
    #tetris.q_network.load_state_dict(torch.load("q_network.pth"))
    #tetris.q_network.eval()  # Set the model to evaluation mode

    for episode in range(train_episodes):
        tetris.reset(episode)

        while not tetris.is_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            tetris.update()
            tetris.draw(screen)
            pygame.display.flip()
            clock.tick(GAME_SPEED)

        # Training loop
        #reward = tetris.get_reward()  # Use the reward as the total reward for simplicity
        # print(f"Episode {episode + 1}/{train_episodes} - reward: {total_reward}")

        # You may want to adjust the reward mechanism based on your specific objectives

        # Training the Q-network
        for _ in range(50):  # Adjust the number of training steps per episode
            tetris.update()  # Update the Q-network through interactions with the environment

        # Save the trained Q-network if needed
        # torch.save(tetris.q_network.state_dict(), f"q_network_episode_{episode + 1}.pth")
        #if reward < tetris.get_reward() and tetris.get_reward() > 16:
        if reward == 1:
            reward = tetris.get_reward()  # Use the reward as the total reward for simplicity
            print("First Data")
            torch.save(tetris.q_network.state_dict(), f"q_network_{tetris.get_reward()}_{tetris.clear_line}.pth")
            torch.save(tetris.q_network.state_dict(), f"q_network.pth")
            tetris.q_network.load_state_dict(torch.load("q_network.pth"))
            tetris.q_network.eval()  # Set the model to evaluation mode
            #print(f"Episode {episode + 1}/{train_episodes} - reward: {tetris.get_reward()}")

        print(f"Episode {episode + 1}/{train_episodes} - reward: {tetris.get_reward()}, clear: {tetris.clear_line}")
        if reward < tetris.get_reward():
        #if tetris.get_reward() != 0:
            reward = tetris.get_reward()  # Use the reward as the total reward for simplicity
            #torch.save(tetris.q_network.state_dict(), "q_network.pth")
            torch.save(tetris.q_network.state_dict(), f"q_network_{tetris.get_reward()}_{tetris.clear_line}.pth")
            torch.save(tetris.q_network.state_dict(), f"q_network.pth")
            tetris.q_network.load_state_dict(torch.load("q_network.pth"))
            tetris.q_network.eval()  # Set the model to evaluation mode
            #print(f"Episode {episode + 1}/{train_episodes} - reward: {tetris.get_reward()}")
            print("Highest Score!!!")

        if clear_line < tetris.clear_line:
            print("New Record!!!")
            clear_line = tetris.clear_line
            torch.save(tetris.q_network.state_dict(), f"q_network_{tetris.get_reward()}_{tetris.clear_line}.pth")
            torch.save(tetris.q_network.state_dict(), f"q_network.pth")
            tetris.q_network.load_state_dict(torch.load("q_network.pth"))
            tetris.q_network.eval()  # Set the model to evaluation mode

    pygame.quit()

if __name__ == '__main__':
    main()
