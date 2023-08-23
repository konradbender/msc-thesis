import pytest
from run_multiple_for_traces import Main
import sys
import numpy as np
import json

def test_main_1(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new"
    
    
    main  = Main(result_dir=tmpdir, arguments = options.split())
    main.main()

def test_main_2(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --dynamic"
    
    
    main  = Main(result_dir=tmpdir, arguments = options.split())
    main.main()


def test_main_random(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --dynamic --random_boundary"
    
    
    main  = Main(result_dir=tmpdir, arguments = options.split())
    main.main()


def test_main_1(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --mixed --fixed_steps=500"
    
    
    main  = Main(result_dir=tmpdir, arguments = options.split())
    main.main()

def test_main_2(tmpdir):
    tmpdir = str(tmpdir) +  "/"

    t = 200
    n = 1
    checkpoint = 1000
    n_int = 5
    padding = 1
    p = 0.7
    tol=0.98

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --tol={tol} --force_new"
    
    
    
    main  = Main(result_dir=tmpdir, arguments=options.split())
    main.main()
  

    result = json.load(open(f"{main.result_dir}/rep-0/result-dict.json", "r"))
    np.testing.assert_array_equal(
        result["vector"],
        np.array(
            [
                0.68,
                0.68,
                0.68,
                0.68,
                0.68,
                0.68,
                0.68,
                0.72,
                0.68,
                0.72,
                0.76,
                0.8,
                0.84,
                0.84,
                0.84,
                0.88,
                0.92,
                0.92,
                0.92,
                0.96,
                0.96,
                0.96,
                0.96,
                0.96,
                0.96,
                0.96,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ),
    )
if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [
                "-c",
                "pyproject.toml",
                "-k main and random",
                "--durations=0"
            ]
        )
    )

