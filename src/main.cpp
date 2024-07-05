#include <iostream>
#include "Tensor3D.h"
#include "ReLULayer.h"
#include "SigmoidLayer.h"
#include "TanhLayer.h"
#include "DenseLayer.h"
#include "Attention.h"
#include "Transformer.h"
#include "LossType.h"

int main() {
    // Manually create a tensor with specific values
    vector<vector<vector<double>>> values = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};
    Tensor3D input(values);
    std::cout << "Input Tensor:" << std::endl;
    input.print();

    // Randomize another tensor for the target
    Tensor3D target(1, 3, 3);
    target.randomize();
    std::cout << "Target Tensor:" << std::endl;
    target.print();

    Transformer transformer(LossType::MSE);
    transformer.addLayer(new DenseLayer(1, 3, 3, 1, 3, 3));
    transformer.addLayer(new ReLULayer());
    transformer.addLayer(new DenseLayer(1, 3, 3, 1, 3, 3));
    transformer.addLayer(new SigmoidLayer());

    std::vector<Tensor3D> inputs = { input };
    std::vector<Tensor3D> targets = { target };
    int epochs = 1000;
    double learningRate = 0.01;

    transformer.train(inputs, targets, epochs, learningRate);

    return 0;
}
