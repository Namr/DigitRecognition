#include <iostream>
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>


using namespace Eigen;
using namespace std;

class Layer
{
public:
  MatrixXd weights;
  VectorXd bias;
  VectorXd weightedValue;
  VectorXd activatedValue;
  
  Layer(int neuronCount, int inputLength);
};

class Network
{
public:
  int *layerShape;
  vector<Layer> layers;
  
  VectorXd input;
  VectorXd output;
  VectorXd desiredOutput;

  double cost;
  
  Network(int *shape, int layerCount, VectorXd inputs);
  Network(int *shape, int layerCount, VectorXd inputs, VectorXd labledOutputs);
  
  void forwardProp(double (*activation)(double), double (*costFunc)(VectorXd, VectorXd));
  void backProp(double (*activationPrime)(double), MatrixXd (*costFuncPrime)(VectorXd, VectorXd));
};
