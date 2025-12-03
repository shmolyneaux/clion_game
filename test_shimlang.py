import subprocess
import os
from pathlib import Path

from time import time

start_time = time()
subprocess.run("cd shimlang && cargo run --bin shm", shell=True)
end_time = time()

duration = end_time - start_time

print(f"Ran in {duration} seconds")