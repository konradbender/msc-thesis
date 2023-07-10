"""
g++ -c -std=c++11 -fPIC glauber.cpp -o glauberC.o
g++ -shared -o libGlauberC.so  glauberC.o 
"""

# write a file for each glauber run, read it in python, delete it 
# and return numpy values as normal glauber python file

from ctypes import cdll, c_double, c_int, c_char_p, create_string_buffer
import datetime
import numpy as np
import os


def run_single_glauber(n_outer: int, n_interior: int, p:float, t:int, tol:float, call_id: int, verbose) -> dict:

    os.makedirs("temp", exist_ok=True)

    assert(call_id < 9999)

    now = datetime.datetime.now()
    time_string = now.strftime('%m%d%H%M%S')

    b_time_string = time_string.encode('utf-8')
    b_time_call = str(call_id).encode('utf-8')

    lib = cdll.LoadLibrary("./c++/libGlauberC.so")
    fixation = lib.run_single_c(c_int(n_outer), c_int(n_interior), c_double(p), c_int(t), c_double(tol), 
                                create_string_buffer(b_time_string),
                                create_string_buffer(b_time_call))
    
    trace = np.loadtxt(f'./temp/{time_string}_{call_id}.txt')

    iterations = np.argmin(trace) # in case of multiple occurences, the first is returned.


    return {'fixation': fixation, 'iterations': iterations, 'vector': trace}

    pass


def run_fixation_simulation(n_outer: int, n_inner: int, p: float, t:int, iter:int, tol: float) -> dict:
    pass
    """
    return {
        "fixation_rate": fixations / iter,
        "mean_iterations_when_fix": mean_iterations,
        "p": p,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "t": t,
    }
    """

def run_simulation(n_outer, n_inner, p, t, iter, tol):

    lib = cdll.LoadLibrary("./c++/libGlauberC.so")

    fixations = 0
    iterations = 0

    for i in range(iter):
        result = lib.run_single_c(n_outer, n_inner, c_double(p), t, 
                                  c_double(tol))

        if result:
            fixations += 1


    return {'fixation_rate': fixations / iter,
            "p": p, "n_outer": n_outer,"n_inner": n_inner, "t": t}




if __name__ == "__main__":
    n_outer = 2000
    n_interior = 1990
    p = 0.83
    t = 3000
    
    thres = 0.85

    run_single_glauber(n_outer, n_interior, p, t, thres, 69)