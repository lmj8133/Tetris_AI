import pygame
import socket
import json

# Define constants similar to your client code
WIDTH, HEIGHT = 10, 20
SCREEN_SIZE = (300, 600)
BLOCK_SIZE = SCREEN_SIZE[0] // WIDTH
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define the colors for each shape
SHAPE_COLORS = [(0, 255, 255), (255, 255, 0), (128, 0, 128), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 165, 0)]

def deserialize_board(serialized_board):
    return json.loads(serialized_board)

def draw_block(screen, color, x, y):
    pygame.draw.rect(screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_board(screen, board):
    screen.fill(BLACK)
    for y, row in enumerate(board):
        for x, block in enumerate(row):
            if block:
                color = SHAPE_COLORS[block - 1]  # Adjust color index
                draw_block(screen, color, x, y)

def receive_complete_json(client_socket):
    buffer = ""
    while True:
        try:
            data = client_socket.recv(4096).decode("utf-8")
            if not data:
                break
            buffer += data
            while True:
                try:
                    # Try to decode the buffer
                    decoded_json, idx = json.JSONDecoder().raw_decode(buffer)
                    yield decoded_json
                    buffer = buffer[idx:]
                except json.JSONDecodeError:
                    # Data is not yet complete, wait for more
                    break
        except socket.error as e:
            print(f"Socket error: {e}")
            break

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 5555))
server.listen()

while True:
    client, address = server.accept()
    print(f"Connection from {address} has been established.")

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Tetris Server')

    for decoded_board in receive_complete_json(client):
        # Draw the board
        draw_board(screen, decoded_board)
        pygame.display.flip()
