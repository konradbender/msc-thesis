import glauber2D
import numpy as np
import logging

def test_single_1():
    np.random.seed(0)
    result = glauber2D.run_single_glauber(100, 85, 0.2, 1e5, 0.85, 
                                          where_to_fixate='lower')
    assert(result["fixation"] == True )

def test_single_2():
    np.random.seed(0)
    result = glauber2D.run_single_glauber(1000, 800, 0.3, 1000)
    assert((result["fixation"] == False) and result["iterations"] == 1000)


def test_simulation():
    np.random.seed(0)
    result = glauber2D.run_simulation(300, 280, 0.7, 200000, 5, 0.8)
    print(result)
    assert(result["fixation_rate"] == 1.0 and result["mean_iterations"] == 134.11)

if __name__ == "__main__":
    test_single_1()