#ifndef ATTENTION_H
#define ATTENTION_H

#include "Tensor3D.h"

class Attention {
public:
    Tensor3D query, key, value;
    Tensor3D output;

    Attention(int depth, int rows, int cols);
    void forward(const Tensor3D& inputQuery, const Tensor3D& inputKey, const Tensor3D& inputValue);
    Tensor3D getOutput() const;
};

#endif // ATTENTION_H
