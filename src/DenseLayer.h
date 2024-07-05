#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Layer.h"

class DenseLayer : public Layer {
public:
    Tensor3D weights;
    Tensor3D biases;
    Tensor3D inputCache;
    Tensor3D output;

    DenseLayer(int inputDepth, int inputRows, int inputCols, int outputDepth, int outputRows, int outputCols);
    void forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& gradOutput, double learningRate) override;
    Tensor3D getOutput() const override;
};

#endif // DENSELAYER_H
