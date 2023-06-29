#include <iostream>
#include <vector>
#include <random>
#include "glauber.h"

extern "C" {
    bool run_single_c(int n_outer, int n_interior, double p, int t, double thres){
        return run_single_glauber(n_outer, n_interior, p, t, thres);}
}

std::vector<std::vector<int> > squary_boundary_fix(int n, double p, int boundary = 1) {
    std::vector<std::vector<int> > matrix(n, std::vector<int>(n));

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

std::vector<std::vector<int> > squary_boundary_random(int n, double p, int boundary = 1) {
    std::vector<std::vector<int> > matrix(n, std::vector<int>(n));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dis(p);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = dis(gen);
        }
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

void update_vertex(std::vector<std::vector<int> > &lattice, std::vector<int>  index) {
    int deltas[4][2] = {
        {0, 1},
        {1, 0},
        {-1, 0},
        {0,- 1}
    };

    int sum = 0;

    // this iterates through neighbor_indices and creates the variable
    // neighbor, for each iteration
    for (const auto& delta : deltas) {
        sum += lattice[index[0] + delta[0]][index[1] + delta[1]];
    }

    if (sum > 2) {
        lattice[index[0]][index[1]] = 1;
    }
    else if (sum < 2) {
        lattice[index[0]][index[1]] = 0;
    }
    else if (sum == 2) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dis(0.5);
        lattice[index[0]][index[1]] = dis(gen);
    }
}

bool run_single_glauber(int n_outer, int n_interior, double p, int t, double thres) {


    std::vector<std::vector<int> > matrix = squary_boundary_fix(n_outer, p);
    std::vector<std::vector<int> > indices = get_random_indices(n_outer, t);

    int buffer = (n_outer - n_interior) / 2;
    std::vector<std::vector<int> > interior_indices(n_interior*n_interior, std::vector<int>(2));
    int index = 0;
    for (int i = 0; i < n_interior; i++) {
        for (int j = 0; j < n_interior; j++) {
            interior_indices[index][0] = buffer + i;
            interior_indices[index][1] = buffer + j;
            ++index;
        }
    }

    int iterations = 0;
    bool fixation = false;
    for (const auto& index : indices) {
        iterations++;
        update_vertex(matrix, index);
        
        double target = interior_indices.size();
        double sum = 0;

        for (const auto& interior : interior_indices) {
            sum += matrix[interior[0]][interior[1]];
        }

        if (sum >= thres*target || sum <= (1-thres)*target) {
            fixation = true;
            break;
        }
        double share = sum / target;
        
        if (iterations % 10000 == 0) {
            std::cout << "iteration: " << std::to_string(iterations) << " for probability " << std::to_string(p); 
            std::cout << " current share of 1: " << std::to_string(share) << std::endl;
        }
    }
    
    std::cout << "finished a single run of glauber dynamics with result: " <<  std::to_string(fixation) << std::endl;

    return fixation;
}
