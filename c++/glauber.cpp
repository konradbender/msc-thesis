#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <random>
#include <string>
#include "glauber.h"
#include <ctime>

const int logging_step = 1000;


extern "C" {
    bool run_single_c(const int n_outer, const int n_interior, const double p, int t, const double thres, 
    const char run_id[], const char call_id[]){
        return run_single_glauber(n_outer, n_interior, p, t, thres, run_id, call_id);}
}


std::vector<std::vector<bool> > squary_boundary_fix(int n, double p, int boundary = 1) {
    std::vector<std::vector<bool> > matrix(n, std::vector<bool>(n));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dis(p);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < n; i++) {
        matrix[0][i] = boundary;
        matrix[n-1][i] = boundary;
    }

    return matrix;
}

std::vector<std::vector<int> > get_random_indices(int n, int t) {
    std::vector<std::vector<int> > indices(t, std::vector<int>(2));

    std::random_device rd;
    std::mt19937 gen(rd());

    // here, the interval is closed on both sides
    std::uniform_int_distribution<> dis(1, n-2);

    for (int i = 0; i < t; i++) {
        indices[i][0] = dis(gen);
        indices[i][1] = dis(gen);
    }

    return indices;
}


bool run_single_glauber(const int n_outer, const int n_interior, const double p, const int t, 
    const double thres, const char run_id[], const char call_id[]) {
    
    clock_t start = clock();
    
    std::vector<std::vector<bool> > matrix = squary_boundary_fix(n_outer, p);

    std::cout << "Aftter matrix: " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;

    // the indices to iterate over
    std::vector<std::vector<int> > indices = get_random_indices(n_outer, t); 

    std::cout << "After indices:" << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;

    std::vector<double> trace = std::vector<double>(t);
    fill(trace.begin(), trace.end(), -1.0); // initialize to -1 to see where iteration stopped

    std::vector<std::vector<int> > interior_indices(n_interior*n_interior, std::vector<int>(2));

    double target = interior_indices.size();
    int buffer = (n_outer - n_interior) / 2;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dis(0.5);

    
    int index = 0;
    for (int i = 0; i < n_interior; i++) {
        for (int j = 0; j < n_interior; j++) {
            interior_indices[index][0] = buffer + i;
            interior_indices[index][1] = buffer + j;
            ++index;
        }
    }

    std::cout << "After interior indices: " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;

    int iterations = 0;
    bool fixation = false;

    for (const auto& index : indices) {
        // begin update vertex

        int nb_sum = matrix[index[0] + 0][index[1] + 1] + 
            matrix[index[0] + 1][index[1] + 0] + 
            matrix[index[0] - 1][index[1] + 0] + 
            matrix[index[0] + 0][index[1] - 1];

        if (nb_sum > 2) {
            matrix[index[0]][index[1]] = 1;
        }
        else if (nb_sum < 2) {
            matrix[index[0]][index[1]] = 0;
        }
        else if (nb_sum == 2) {
            matrix[index[0]][index[1]] = dis(gen);
        }

        // end update vertex
                
        double sum = 0;

        for (const auto& interior : interior_indices) {
            sum += matrix[interior[0]][interior[1]];
        }
        
        double share = sum / target;
        trace[iterations] = share;

        if (sum >= thres*target) {
            fixation = true;
            break;
        }
        else if (sum <= (1-thres)*target) {
            fixation = false;
            break;
        }
        
        iterations++;
        
        if (iterations % logging_step == 0) {
            std::cout << static_cast<double>(clock() - start) / CLOCKS_PER_SEC  << std::to_string(iterations) << " for probability " << std::to_string(p); 
            std::cout << " current share of 1: " << std::to_string(share) << std::endl;
        }

        
    }
    
    std::cout << "finished a single run of glauber dynamics with result: " <<  std::to_string(fixation) 
        << " after " << std::to_string(iterations) << " iterations." << std::endl;

    // save vector to file
    std::string filename = "temp/" + std::string(run_id) + "_" + std::string(call_id) + ".txt";
    std::ofstream output_file(filename);
    std::ostream_iterator<double> output_iterator(output_file, "\n");
    std::copy(trace.begin(), trace.end(), output_iterator);
    output_file.close();

    return fixation;
}
