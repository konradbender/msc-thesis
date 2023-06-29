#include <iostream>
#include <vector>
#include "glauber.h"

int main() {
    // Example usage
    std::cout << "Hello world from main.cpp!" << std::endl;
    int n_outer = 300;
    int n_interior = 280;
    double p = 0.8;
    int t = 100000;
    double thres = 0.85;

    bool result = run_single_glauber(n_outer, n_interior, p, t, thres);

    std::cout << "fixation: " << std::to_string(result) << std::endl;

    return 0;

}
