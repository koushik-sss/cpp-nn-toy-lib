#ifndef SIGMOIDLAYER_H
#define SIGMOIDLAYER_H

#include "Layer.h"

class SigmoidLayer : public Layer {
public:
    Tensor3D inputCache;
    Tensor3D output;

    void forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& gradOutput, double learningRate) override;
    Tensor3D getOutput() const override;
};

#endif // SIGMOIDLAYER_H
