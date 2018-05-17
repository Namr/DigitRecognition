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

//pushes data forward through the network, starts with the first layer
//and multiplies the data layer by the wieghts and adds the biases of the next layer
//that layer is now the new output, which is then multiplied by the weights and biases
//of the next layer and so on, each layer also goes through an activation function which is passed
void Network::forwardProp(double (*activation)(double))
{
  output = input;
  for(Layer layer : layers)
  {
    output = (layer.weights * output) + layer.bias;
    output = output.unaryExpr(activation);
  }
}
