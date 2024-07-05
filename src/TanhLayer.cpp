#include "TanhLayer.h"
#include <cmath>

void TanhLayer::forward(const Tensor3D& input) {
    inputCache = input;
    output = Tensor3D(input.depth, input.rows, input.cols);
    for (int d = 0; d < input.depth; d++) {
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[d][i][j] = tanh(input.data[d][i][j]);
            }
        }
    }
}

Tensor3D TanhLayer::backward(const Tensor3D& gradOutput, double learningRate) {
    Tensor3D gradInput(gradOutput.depth, gradOutput.rows, gradOutput.cols);
    for (int d = 0; d < gradOutput.depth; d++) {
        for (int i = 0; i < gradOutput.rows; i++) {
            for (int j = 0; j < gradOutput.cols; j++) {
                double tanhVal = output.data[d][i][j];
                gradInput.data[d][i][j] = gradOutput.data[d][i][j] * (1 - tanhVal * tanhVal);
            }
        }
    }
    return gradInput;
}

Tensor3D TanhLayer::getOutput() const {
    return output;
}
