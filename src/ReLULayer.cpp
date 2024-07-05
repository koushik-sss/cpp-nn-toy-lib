#include "ReLULayer.h"
#include <algorithm>

void ReLULayer::forward(const Tensor3D& input) {
    inputCache = input;
    output = Tensor3D(input.depth, input.rows, input.cols);
    for (int d = 0; d < input.depth; d++) {
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[d][i][j] = max(0.0, input.data[d][i][j]);
            }
        }
    }
}

Tensor3D ReLULayer::backward(const Tensor3D& gradOutput, double learningRate) {
    Tensor3D gradInput(gradOutput.depth, gradOutput.rows, gradOutput.cols);
    for (int d = 0; d < gradOutput.depth; d++) {
        for (int i = 0; i < gradOutput.rows; i++) {
            for (int j = 0; j < gradOutput.cols; j++) {
                gradInput.data[d][i][j] = (inputCache.data[d][i][j] > 0) ? gradOutput.data[d][i][j] : 0;
            }
        }
    }
    return gradInput;
}

Tensor3D ReLULayer::getOutput() const {
    return output;
}
