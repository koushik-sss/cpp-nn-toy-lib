#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork(LossType lossType) : lossType(lossType) {}

void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void NeuralNetwork::forward(const Tensor3D& input) {
    Tensor3D currentInput = input;
    for (Layer* layer : layers) {
        layer->forward(currentInput);
        currentInput = layer->getOutput();
    }
}

void NeuralNetwork::backward(const Tensor3D& gradOutput, double learningRate) {
    Tensor3D currentGradOutput = gradOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        currentGradOutput = (*it)->backward(currentGradOutput, learningRate);
    }
}

Tensor3D NeuralNetwork::getOutput() const {
    return layers.back()->getOutput();
}

void NeuralNetwork::train(const std::vector<Tensor3D>& inputs, const std::vector<Tensor3D>& targets, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;
        for (size_t i = 0; i < inputs.size(); i++) {
            forward(inputs[i]);
            Tensor3D output = getOutput();
            totalLoss += computeLoss(output, targets[i]);
            Tensor3D lossGrad = computeLossGradient(output, targets[i]);
            backward(lossGrad, learningRate);
        }
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size() << std::endl;
    }
}

double NeuralNetwork::computeLoss(const Tensor3D& output, const Tensor3D& target) {
    double loss = 0.0;
    int n = output.depth * output.rows * output.cols;

    switch (lossType) {
        case LossType::MSE:
            for (int d = 0; d < output.depth; d++) {
                for (int i = 0; i < output.rows; i++) {
                    for (int j = 0; j < output.cols; j++) {
                        loss += std::pow(output.data[d][i][j] - target.data[d][i][j], 2);
                    }
                }
            }
            return loss / n;
        case LossType::MAE:
            for (int d = 0; d < output.depth; d++) {
                for (int i = 0; i < output.rows; i++) {
                    for (int j = 0; j < output.cols; j++) {
                        loss += std::abs(output.data[d][i][j] - target.data[d][i][j]);
                    }
                }
            }
            return loss / n;
    }
    return loss;
}

Tensor3D NeuralNetwork::computeLossGradient(const Tensor3D& output, const Tensor3D& target) {
    Tensor3D grad(output.depth, output.rows, output.cols);
    int n = output.depth * output.rows * output.cols;

    switch (lossType) {
        case LossType::MSE:
            for (int d = 0; d < output.depth; d++) {
                for (int i = 0; i < output.rows; i++) {
                    for (int j = 0; j < output.cols; j++) {
                        grad.data[d][i][j] = 2.0 * (output.data[d][i][j] - target.data[d][i][j]) / n;
                    }
                }
            }
            break;
        case LossType::MAE:
            for (int d = 0; d < output.depth; d++) {
                for (int i = 0; i < output.rows; i++) {
                    for (int j = 0; j < output.cols; j++) {
                        grad.data[d][i][j] = (output.data[d][i][j] > target.data[d][i][j]) ? 1.0 / n : -1.0 / n;
                    }
                }
            }
            break;
    }
    return grad;
}
