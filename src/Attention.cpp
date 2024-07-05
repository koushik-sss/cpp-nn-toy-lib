#include "Attention.h"
#include <cmath>

Attention::Attention(int depth, int rows, int cols)
    : query(depth, rows, cols), key(depth, rows, cols), value(depth, rows, cols), output(depth, rows, cols) {}

void Attention::forward(const Tensor3D& inputQuery, const Tensor3D& inputKey, const Tensor3D& inputValue) {
    query = inputQuery;
    key = inputKey;
    value = inputValue;

    int depth = query.depth;
    int rows = query.rows;
    int cols = query.cols;

    // Compute the dot product of query and key
    Tensor3D scores(depth, rows, cols);
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                scores.data[d][i][j] = query.data[d][i][j] * key.data[d][i][j];
            }
        }
    }

    // Apply softmax to scores
    Tensor3D softmaxScores(depth, rows, cols);
    for (int d = 0; d < depth; d++) {
        double sumExp = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sumExp += exp(scores.data[d][i][j]);
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                softmaxScores.data[d][i][j] = exp(scores.data[d][i][j]) / sumExp;
            }
        }
    }

    // Compute the weighted sum of values
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[d][i][j] = 0.0;
                for (int k = 0; k < cols; k++) {
                    output.data[d][i][j] += softmaxScores.data[d][i][k] * value.data[d][k][j];
                }
            }
        }
    }
}

Tensor3D Attention::getOutput() const {
    return output;
}
