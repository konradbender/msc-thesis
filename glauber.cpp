#include <iostream>
#include <vector>
#include <random>

std::vector<std::vector<int>> squary_boundary_fix(int n, double p, int boundary = 1) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));

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

std::vector<std::vector<int>> squary_boundary_random(int n, double p, int boundary = 1) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));

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

std::vector<std::vector<int>> get_random_indices(int n, int t) {
    std::vector<std::vector<int>> indices(t, std::vector<int>(2));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, n-2);

    for (int i = 0; i < t; i++) {
        indices[i][0] = dis(gen);
        indices[i][1] = dis(gen);
    }

    return indices;
}

void update_vertex(std::vector<std::vector<int>>& lattice, std::vector<int>& index) {
    std::vector<std::vector<int>> neighbor_indices = {
        {index[0], index[1] + 1},
        {index[0] + 1, index[1]},
        {index[0] - 1, index[1]},
        {index[0], index[1] - 1}
    };

    int sum = 0;
    for (const auto& neighbor : neighbor_indices) {
        sum += lattice[neighbor[0]][neighbor[1]];
    }

    if (sum > 2) {
        lattice[index[0]][index[1]] = 1;
    }
    else if (sum == 2) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dis(0.5);
        lattice[index[0]][index[1]] = dis(gen);
    }
}

std::vector<std::vector<int>> run_single_glauber(int n_outer, int n_interior, double p, int t, bool debug = false) {
    std::vector<std::vector<int>> matrix = squary_boundary_fix(n_outer, p);
    std::vector<std::vector<int>> indices = get_random_indices(n_outer, t);

    int buffer = (n_outer - n_interior) / 2;
    std::vector<std::vector<int>> interior_indices(n_interior, std::vector<int>(2));
    for (int i = 0; i < n_interior; i++) {
        interior_indices[i][0] = buffer + i;
