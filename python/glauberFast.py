"""
g++ -c -std=c++11 -fPIC glauber.cpp -o glauberC.o
g++ -shared -o libGlauberC.so  glauberC.o 
"""

# write a file for each glauber run

from ctypes import cdll, c_double, c_int



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