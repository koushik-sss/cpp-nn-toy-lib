#ifndef LAYER_H
#define LAYER_H

#include "Tensor3D.h"

class Layer {
public:
    virtual void forward(const Tensor3D& input) = 0;
    virtual Tensor3D backward(const Tensor3D& gradOutput, double learningRate) = 0;
    virtual Tensor3D getOutput() const = 0;
    virtual ~Layer() = default;
};

#endif // LAYER_H
