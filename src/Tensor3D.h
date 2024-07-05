#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
using namespace std;

class Tensor3D {
public:
    int depth, rows, cols;
    vector<vector<vector<double> > > data;

    Tensor3D();
    Tensor3D(int d, int r, int c);
    Tensor3D(const vector<vector<vector<double> > >& values);
    void randomize();
    Tensor3D operator-(const Tensor3D& other) const;
    void print() const;
};

#endif // TENSOR3D_H
