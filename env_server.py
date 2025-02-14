#!/usr/bin/env python3
import socket
import threading
import json
import numpy as np
import time

# Import the Isaac Lab environment loader from skrl.
from skrl.envs.loaders.torch import load_isaaclab_env

NUM_ENVS = 4
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9999

envs = {}
env_connections = {}  # Mapping: env_id -> client socket (once connected)
env_locks = {}

def env_worker(env_id):
    # Load the AnyMal C Rough environment (ensure task name matches your registration)
    env = load_isaaclab_env(
        task_name="Isaac-AnymalC-RoughEnv-v0",
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
            data = {"observation": obs.tolist() if isinstance(obs, np.ndarray) else obs}
            client_sock.sendall((json.dumps(data) + "\n").encode())
            # Wait for action from client.
            action_line = ""
            while "\n" not in action_line:
                chunk = client_sock.recv(4096).decode()
                if not chunk:
                    raise ConnectionError("Client disconnected.")
                action_line += chunk
            action = json.loads(action_line.strip())
            obs, reward, done, info = env.step(action)
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

def handle_client(conn, addr):
    print(f"[Server] Connection from {addr}")
    try:
        env_id_line = ""
        while "\n" not in env_id_line:
            chunk = conn.recv(1024).decode()
            if not chunk:
                raise ConnectionError("Client disconnected before sending env_id.")
            env_id_line += chunk
        env_id = int(env_id_line.strip())
        print(f"[Server] Client {addr} assigned to env {env_id}")
        env_connections[env_id] = conn
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[Server] Error with client {addr}: {e}")
    finally:
        print(f"[Server] Closing connection from {addr}")
        conn.close()
        env_connections[env_id] = None

def start_server():
    for env_id in range(NUM_ENVS):
        env_connections[env_id] = None
        t = threading.Thread(target=env_worker, args=(env_id,), daemon=True)
        t.start()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(NUM_ENVS)
    print(f"[Server] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    while True:
        conn, addr = server.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        client_thread.start()

if __name__ == "__main__":
    start_server()
