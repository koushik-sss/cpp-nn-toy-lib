#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include "Layer.h"
#include "Attention.h"
#include "Tensor3D.h"
#include "LossType.h"

class Transformer {
public:
    std::vector<Layer*> layers;
    Attention* attention;
    LossType lossType;

    Transformer(LossType lossType = LossType::MSE);
    void addLayer(Layer* layer);
    void forward(const Tensor3D& input);
    void backward(const Tensor3D& gradOutput, double learningRate);
    Tensor3D getOutput() const;
    void train(const std::vector<Tensor3D>& inputs, const std::vector<Tensor3D>& targets, int epochs, double learningRate);
    double computeLoss(const Tensor3D& output, const Tensor3D& target);
    Tensor3D computeLossGradient(const Tensor3D& output, const Tensor3D& target);
};

#endif // TRANSFORMER_H
