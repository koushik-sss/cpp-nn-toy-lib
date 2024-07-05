#ifndef RELULAYER_H
#define RELULAYER_H

#include "Layer.h"

class ReLULayer : public Layer {
public:
    Tensor3D inputCache;
    Tensor3D output;

    void forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& gradOutput, double learningRate) override;
    Tensor3D getOutput() const override;
};

#endif // RELULAYER_H
