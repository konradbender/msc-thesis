#include <iostream>
#include <vector>
#include "glauber.h"

int main() {
    // Example usage
    std::cout << "Hello world from main.cpp!" << std::endl;
    int n_outer = 2000;
    int n_interior = 1990;
    double p = 0.83;
    int t = 3000;
    char run_id[11] = "0706150001";
    char call_id[5] = "1";
    double thres = 0.85;

    bool result = run_single_glauber(n_outer, n_interior, p, t, thres, run_id, call_id);

    // std::cout << "fixation: " << std::to_string(result) << std::endl;

    return 0;

}
