#!/usr/bin/env python3

# In this design, the server spawns four environment threads 
# (each loading the env via Isaac Lab’s loader) and then listens for client connections. 
# When a client connects it must first send its assigned environment ID (an integer from 0 to 3).
# Then the corresponding environment thread enters a loop where it sends the latest observation to 
# that client, waits for an action response, applies that action (via env.step), and sends back the 
# step’s outcome (observation, reward, done flag, etc.).

# big thanks to Issac Lab : https://isaac-sim.github.io/IsaacLab/main/index.html
# and skrl gym : https://skrl.readthedocs.io/en/latest/api/envs/isaaclab.html
# for easy integration
import socket
import threading
import json
import numpy as np
import time

# Import Isaac Lab environment loader (Torch version here)
from skrl.envs.loaders.torch import load_isaaclab_env

# Global settings
NUM_ENVS = 4
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9999

# Global dictionaries to hold the environment instances and their locks.
envs = {}
env_connections = {}  # Mapping: env_id -> client socket (once connected)
env_locks = {}

# This function runs in a separate thread for each env instance.
def env_worker(env_id):
    # Load the AnyMal C Rough environment for a single env instance.
    # Note: Ensure the task name matches the registered task for AnyMal C Rough.
    env = load_isaaclab_env(
        task_name="Isaac-AnymalC-RoughEnv-v0",  # Adjust if needed.
        num_envs=1,
        headless=True,
        cli_args=[],
        show_cfg=False
    )
    envs[env_id] = env
    env_locks[env_id] = threading.Lock()
    print(f"[Env {env_id}] Environment loaded.")
    
    # Wait until a client connects for this env.
    while env_id not in env_connections or env_connections[env_id] is None:
        time.sleep(0.1)
    
    client_sock = env_connections[env_id]
    try:
        obs = env.reset()
        while True:
            # Send the current observation to the client.
            # (If obs is a numpy array, convert it to a list.)
            data = {"observation": obs.tolist() if isinstance(obs, np.ndarray) else obs}
            client_sock.sendall((json.dumps(data) + "\n").encode())

            # Wait to receive an action from the client.
            action_line = ""
            while "\n" not in action_line:
                chunk = client_sock.recv(4096).decode()
                if not chunk:
                    raise ConnectionError("Client disconnected.")
                action_line += chunk
            action = json.loads(action_line.strip())
            
            # Step the environment with the received action.
            obs, reward, done, info = env.step(action)
            # Package the step result.
            result = {
                "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
                "reward": reward,
                "done": done,
                "info": info
            }
            client_sock.sendall((json.dumps(result) + "\n").encode())
            
            if done:
                obs = env.reset()
    except Exception as e:
        print(f"[Env {env_id}] Exception: {e}")
    finally:
        print(f"[Env {env_id}] Closing client connection.")
        client_sock.close()
        env_connections[env_id] = None

# TCP server handler for incoming client connections.
def handle_client(conn, addr):
    print(f"[Server] Connection from {addr}")
    try:
        # Expect the client to immediately send its environment id (as a string, terminated by newline).
        env_id_line = ""
        while "\n" not in env_id_line:
            chunk = conn.recv(1024).decode()
            if not chunk:
                raise ConnectionError("Client disconnected before sending env_id.")
            env_id_line += chunk
        env_id = int(env_id_line.strip())
        print(f"[Server] Client {addr} assigned to env {env_id}")
        
        # Register the connection for this env id.
        env_connections[env_id] = conn
        
        # Now, the env_worker for this env_id (running in its own thread) will handle the communication.
        # This handler thread will simply keep the connection open.
        while True:
            time.sleep(1)
            # Optionally, implement a heartbeat here.
    except Exception as e:
        print(f"[Server] Error with client {addr}: {e}")
    finally:
        print(f"[Server] Closing connection from {addr}")
        conn.close()
        # Ensure we mark this env as unassigned.
        env_connections[env_id] = None

def start_server():
    # Start environment threads.
    for env_id in range(NUM_ENVS):
        env_connections[env_id] = None  # Initialize with no connection.
        t = threading.Thread(target=env_worker, args=(env_id,), daemon=True)
        t.start()
    
    # Create TCP server socket.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(NUM_ENVS)
    print(f"[Server] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    # Accept incoming client connections.
    while True:
        conn, addr = server.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        client_thread.start()

if __name__ == "__main__":
    start_server()
