import subprocess
import os
from pathlib import Path

from time import time

test_path = Path(__file__).parent / "shimlang"

os.chdir(test_path)

start_time = time()
subprocess.run("cargo run --bin shm -- test_scripts/print.shm", shell=True)
end_time = time()

duration = end_time - start_time

print(f"Ran in {duration} seconds")