#include "network.hpp"

Layer::Layer(int neuronCount, int inputLength)
{
  weights = MatrixXd::Random(neuronCount, inputLength);
  bias = VectorXd::Random(neuronCount);
}


Network::Network(int *shape, int layerCount, VectorXd inputs)
{
  input = inputs;
  layerShape = shape;

  for(int i = 0; i < layerCount; i++)
  {
    
    Layer l(shape[i], input.rows());
    
    if(i > 0)
    {
      l = Layer(shape[i], shape[i-1]);
    }
    
    layers.push_back(l);
  }
  
}

void Network::forwardProp(double (*activation)(double))
{
  output = input;
  for(Layer layer : layers)
  {
    output = (layer.weights * output) + layer.bias;
    output = output.unaryExpr(activation);
  }
}
