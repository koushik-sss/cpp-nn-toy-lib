#include "DenseLayer.h"

DenseLayer::DenseLayer(int inputDepth, int inputRows, int inputCols, int outputDepth, int outputRows, int outputCols)
    : weights(outputDepth, inputRows, inputCols), biases(outputDepth, outputRows, outputCols) {
    weights.randomize();
    biases.randomize();
}

void DenseLayer::forward(const Tensor3D& input) {
    inputCache = input;
    output = Tensor3D(weights.depth, input.rows, input.cols);
    for (int d = 0; d < weights.depth; d++) {
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[d][i][j] = 0.0;
                for (int k = 0; k < input.depth; k++) {
                    output.data[d][i][j] += weights.data[d][i][j] * input.data[k][i][j];
                }
                output.data[d][i][j] += biases.data[d][i][j];
            }
        }
    }
}

Tensor3D DenseLayer::backward(const Tensor3D& gradOutput, double learningRate) {
    Tensor3D gradInput(inputCache.depth, inputCache.rows, inputCache.cols);
    for (int d = 0; d < weights.depth; d++) {
        for (int i = 0; i < inputCache.rows; i++) {
            for (int j = 0; j < inputCache.cols; j++) {
                for (int k = 0; k < inputCache.depth; k++) {
                    gradInput.data[k][i][j] += gradOutput.data[d][i][j] * weights.data[d][i][j];
                    weights.data[d][i][j] -= learningRate * gradOutput.data[d][i][j] * inputCache.data[k][i][j];
                }
                biases.data[d][i][j] -= learningRate * gradOutput.data[d][i][j];
            }
        }
    }
    return gradInput;
}

Tensor3D DenseLayer::getOutput() const {
    return output;
}
