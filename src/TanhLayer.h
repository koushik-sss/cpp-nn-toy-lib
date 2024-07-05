#ifndef TANHLAYER_H
#define TANHLAYER_H

#include "Layer.h"

class TanhLayer : public Layer {
public:
    Tensor3D inputCache;
    Tensor3D output;

    void forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& gradOutput, double learningRate) override;
    Tensor3D getOutput() const override;
};

#endif // TANHLAYER_H
