"""You can not run these test from the terminal through python. Need to run pytest directly"""""
import pytest
from main import Main
import main
import sys
import numpy as np
import json
import os

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
    assert main.main() == 0

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
    
    
    main_instance  = Main(result_dir=tmpdir, arguments = options.split())
    assert main_instance.main() == 0


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
    
    
    main_instance  = Main(result_dir=tmpdir, arguments = options.split())
    assert main_instance.main() == 0


def test_main_3(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --mixed --fixed_steps=500"
    
    
    main_instance  = Main(result_dir=tmpdir, arguments = options.split())
    assert main_instance.main() == 0

def test_main_4(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 0
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --mixed --fixed_steps=500 --torus"
    
    
    main_instance  = Main(result_dir=tmpdir, arguments = options.split())
    assert main_instance.main() == 0

def test_main_5(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --torus"
    
    
    main_instance  = Main(result_dir=tmpdir, arguments = options.split())
    assert main_instance.main() == 0

def test_main_6(tmpdir):

    tmpdir = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new --dynamic --torus"
    
    
    main_instance  = Main(result_dir=tmpdir, arguments = options.split())
    assert main_instance.main() == 0

def test_checkpoint_discovery(tmpdir):

    tmp = main.RESULT_DIR

    main.RESULT_DIR = str(tmpdir) + "/"
    
    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t//2} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new"
    
    main_instance  = Main(arguments = options.split())
    assert main_instance.main() == 0


    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p}"
    
    main_instance_2  = Main(arguments = options.split())
    assert main_instance_2.main() == 0

    main.RESULT_DIR = tmp

def test_cp_beyond_stop(tmpdir):

    tmp = main.RESULT_DIR
    main.RESULT_DIR = str(tmpdir) + "/"

    t = 2000
    n = 4
    checkpoint = 1000
    n_int = 200
    padding = 1
    p = 0.505

    options = f"--t={t} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p} --force_new"
    
    main_instance  = Main(arguments = options.split())
    assert main_instance.main() == 0


    options = f"--t={t//2} --n={n} --checkpoint={checkpoint} " + \
        f"--n_int={n_int} --padding={padding} --p={p}"
    
    main_instance_2  = Main(arguments = options.split())
    assert main_instance_2.main() == 0

    main.RESULT_DIR = tmp    


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
    
    main_instance  = Main(result_dir=tmpdir, arguments=options.split())
    assert main_instance.main() == 0
  
    result = json.load(open(os.path.join(main_instance.result_dir, "rep-0", "result-dict.json"), "r"))
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



