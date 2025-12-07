import itertools
import os
import subprocess
import sys
from pathlib import Path

from time import time

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[34m"
YELLOW  = "\033[33m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

test_path = Path(__file__).parent / "shimlang"

os.chdir(test_path)


start_time = time()
result = subprocess.run("cargo build --bin shm", shell=True)
end_time = time()
duration = end_time - start_time

if result.returncode:
    print("Failed to build")
    sys.exit()

print(f"Built executable in  {duration} seconds")

scripts = []

scripts.extend(Path("test_scripts").glob("*.shm"))

def print_diff(expected, actual):
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()

    expected_max_length = max(45, max((len(line) for line in expected_lines), default=0))
    diff_length = max(len(expected_lines), len(actual_lines))

    gutter_width = len(str(diff_length))

    output_lines = []

    print(
        f"   {' ':<{gutter_width}}   {'Expected':<{expected_max_length}}   Actual"
    )
    for line_count, expected_line, actual_line in itertools.zip_longest(range(1, diff_length + 1), expected_lines, actual_lines):
        # TODO: replace with better diffing than prefix matching
        if expected_line != actual_line:
            print(YELLOW, end='')

        expected_line = "<NONE>" if expected_line is None else expected_line
        actual_line = "<NONE>" if actual_line is None else actual_line

        print(
            f"   {line_count:<{gutter_width}} | {expected_line:<{expected_max_length}} | {actual_line}"
        )
        print(RESET, end='')


failures = []

for script in scripts:
    print(f"{script} ... ", end="")
    stdout_file = script.with_suffix(".stdout")
    stderr_file = script.with_suffix(".stderr")

    expected_stdout = ""
    if stdout_file.exists(): 
        expected_stdout = stdout_file.read_text().strip()

    expected_stderr = ""
    if stderr_file.exists(): 
        expected_stderr = stderr_file.read_text().strip()

    proc = subprocess.run(f"target\\debug\\shm.exe {script}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc_stdout = proc.stdout.strip()
    proc_stderr = proc.stderr.strip()

    if proc.returncode:
        msg = "FAILED (non-zero exit code)"
        print(f"{RED}{msg}{RESET}")
        print("")
        print("STDERR:")
        print(proc_stderr)
        print("")
        failures.append(f"{script} ... {msg}")

    elif proc_stderr != expected_stderr:
        msg = "FAILED (stderr mismatch)"
        print(f"{RED}{msg}{RESET}")
        print("")
        print("STDERR:")
        print_diff(expected=expected_stderr, actual=proc_stderr)
        print("")
        failures.append(f"{script} ... {msg}")

    elif proc_stdout != expected_stdout:
        msg = "FAILED (stdout mismatch)"
        print(f"{RED}{msg}{RESET}")
        print("")
        print("STDOUT:")
        print_diff(
            expected=expected_stdout,
            actual=proc_stdout,
        )
        print("")
        failures.append(f"{script} ... {msg}")

    else:
        print(f"{GREEN}PASSED{RESET}")

if failures:
    print("Failures:")
else:
    print(f"{GREEN}All testing passed!{RESET}")

for failure in failures:
    print(f"    {failure}")

print()