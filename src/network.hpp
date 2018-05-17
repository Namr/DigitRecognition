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

  Layer(int neuronCount, int inputLength);
};

class Network
{
public:
  int *layerShape;
  vector<Layer> layers;
  VectorXd input;
  VectorXd output;
  
  Network(int *shape, int layerCount, VectorXd inputs);
  void forwardProp(double (*activation)(double));
  
};
