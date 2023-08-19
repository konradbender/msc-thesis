import pytest
from run_multiple_for_traces import Main
import sys

def test_main():

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 10

    options = f"--t={t} --n={n} --checkpoint={checkpoint} --n_int={n_int} --padding={padding}"
    
    try:
        main  = Main()
        main.main(options.split())
    except SystemExit:
        pass

if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [
                "-c",
                "pyproject.toml",
                "-k test_main ",
                "--durations=0"
            ]
        )
    )

