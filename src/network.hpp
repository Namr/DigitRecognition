#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Layer {
public:
  MatrixXd weights;
  VectorXd bias;

  VectorXd weightedValue;
  VectorXd activatedValue;

  MatrixXd sumDeltaW;
  VectorXd sumDeltaB;

  Layer(int neuronCount, int inputLength);
};

class Network {
public:
  int *layerShape;
  vector<Layer> layers;

  VectorXd input;
  VectorXd output;
  VectorXd desiredOutput;

  double cost;

  Network(int *shape, int layerCount, VectorXd inputs);
  Network(int *shape, int layerCount, VectorXd inputs, VectorXd labledOutputs);

  void setInput(VectorXd inputs);
  void setInput(VectorXd inputs, VectorXd outputs);

  void forwardProp(double (*activation)(double),
                   double (*costFunc)(VectorXd, VectorXd));
  void backProp(double (*activationPrime)(double),
                MatrixXd (*costFuncPrime)(VectorXd, VectorXd));
  void minibatch(VectorXd *inputs, VectorXd *labels, double learningRate,
                 int trainingStart, int trainingSize,
                 double (*activation)(double),
                 double (*costFunc)(VectorXd, VectorXd),
                 double (*activationPrime)(double),
                 MatrixXd (*costFuncPrime)(VectorXd, VectorXd));
};
