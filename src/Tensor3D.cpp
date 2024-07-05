#include "Tensor3D.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

Tensor3D::Tensor3D() : depth(0), rows(0), cols(0) {}

Tensor3D::Tensor3D(int d, int r, int c) : depth(d), rows(r), cols(c), data(d, vector<vector<double>>(r, vector<double>(c))) {}

Tensor3D::Tensor3D(const vector<vector<vector<double>>>& values) {
    depth = values.size();
    rows = depth > 0 ? values[0].size() : 0;
    cols = rows > 0 ? values[0][0].size() : 0;
    data = values;
}

void Tensor3D::randomize() {
    srand(time(0));
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[d][i][j] = (static_cast<double>(rand()) / (RAND_MAX)) - 0.5;
            }
        }
    }
}

Tensor3D Tensor3D::operator-(const Tensor3D& other) const {
    Tensor3D result(depth, rows, cols);
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[d][i][j] = data[d][i][j] - other.data[d][i][j];
            }
        }
    }
    return result;
}

void Tensor3D::print() const {
    for (int d = 0; d < depth; d++) {
        cout << "Depth " << d << ":\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << data[d][i][j] << " ";
            }
            cout << endl;
        }
    }
}
