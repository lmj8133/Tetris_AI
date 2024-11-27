import os
import pygame
import socket
import json
import select

# Define constants similar to your client code
WIDTH, HEIGHT = 10, 20
SCREEN_SIZE = (600, 600)  # Double the width to accommodate two boards
BLOCK_SIZE = SCREEN_SIZE[0] // (2 * WIDTH)  # Adjust block size accordingly
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define the colors for each shape
SHAPE_COLORS = [(0, 255, 255), (255, 255, 0), (128, 0, 128), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 165, 0)]

def deserialize_board(serialized_board):
    return json.loads(serialized_board)

def draw_block(screen, color, x, y, offset=0):
    pygame.draw.rect(screen, color, ((x + offset) * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, BLACK, ((x + offset) * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_board(screen, board, offset=0):
    for y, row in enumerate(board):
        for x, block in enumerate(row):
            if block:
                color = SHAPE_COLORS[block - 1]  # Adjust color index
                draw_block(screen, color, x, y, offset)

clients_ready = set()  # Track ready clients

def handle_client_data(client_socket, offset):
    try:
        data = client_socket.recv(4096)
        if data:
            decoded_data = data.decode("utf-8")
            if decoded_data == "ready":
                # Once one client sends "ready", broadcast "start" to both clients
                print(f"Client {clients[client_socket]} is ready")
                for sock in clients.keys():
                    sock.send(bytes("start", "utf-8"))
                    print(f"Sent 'start' to {clients[sock]}")
            elif decoded_data == "gameover":
                # Once one client sends "gameover", broadcast "gameover" to both clients
                for sock in clients.keys():
                    sock.send(bytes("gameover", "utf-8"))
                    print(f"Sent 'gameover' to {clients[sock]}")
            else:
                # Assume the received data is either board data or garbage data
                broadcast_data(client_socket, data)  # Broadcast the received data to other clients
        else:
            return False
    except Exception as e:
        print(f"Error handling client data: {e}")
        return False
    return True

def broadcast_data(sender_socket, data):
    for client_socket in clients.keys():
        if client_socket != sender_socket:  # Don't send back to the sender
            try:
                if isinstance(data, str):
                    data = data.encode("utf-8")
                data += b'\n'  # Append newline to indicate end of each JSON object
                client_socket.send(data)
            except Exception as e:
                print(f"Error broadcasting to {clients[client_socket]}: {e}")
                client_socket.close()
                del clients[client_socket]

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 5555))
server.listen(2)  # Listen for up to 2 connections
server.setblocking(0)  # Make the server non-blocking

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Tetris Server')

clients = {}
client_offsets = {}

try:
    while True:
        # Use select to wait for readability on multiple sockets, including the server socket itself
        readable, _, _ = select.select([server] + list(clients.keys()), [], [], 0.1)
        
        for sock in readable:
            if sock is server:
                # Accept new connections
                client, address = server.accept()
                print(f"Connection from {address} has been established.")
                client.setblocking(0)
                clients[client] = address
                client_offsets[client] = len(clients) - 1  # Offset based on client count
                if len(clients) == 2:
                    for sock in clients.keys():
                        sock.send(bytes("connected", "utf-8"))  # Notify clients that they are connected
            else:
                # Handle client data
                offset = client_offsets[sock] * WIDTH
                if not handle_client_data(sock, offset):
                    print(f"Closing connection to {clients[sock]}")
                    del client_offsets[sock]
                    sock.close()
                    del clients[sock]

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise Exception("Pygame Quit")

except Exception as e:
    print(f"Server terminated: {e}")

finally:
    server.close()
    pygame.quit()

