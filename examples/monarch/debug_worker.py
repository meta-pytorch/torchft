import os
import socket
import traceback

print(f"PID: {os.getpid()}")
print(f"Hostname: {socket.getfqdn()}")

port = os.environ.get("MONARCH_PORT", "26600")
hostname = socket.getfqdn()
address = f"tcp://{hostname}:{port}"
print(f"Starting worker at {address}")

try:
    from monarch.actor import run_worker_loop_forever
    run_worker_loop_forever(address=address, ca="trust_all_connections")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("Worker exited")
