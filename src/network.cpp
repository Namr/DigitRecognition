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

Network::Network(int *shape, int layerCount, VectorXd inputs, VectorXd labledOutputs)
{
  input = inputs;
  layerShape = shape;
  desiredOutput = labledOutputs;
  
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
void Network::forwardProp(double (*activation)(double), double (*costFunc)(VectorXd, VectorXd))
{
  output = input;
  for(int i = 0; i < layers.size(); i++)
  {
    layers[i].weightedValue = (layers[i].weights * output) + layers[i].bias;
    layers[i].activatedValue = layers[i].weightedValue.unaryExpr(activation);
    output = layers[i].activatedValue;
  }

  cost = costFunc(output, desiredOutput);
}

void Network::backProp(double (*activationPrime)(double), MatrixXd (*costFuncPrime)(VectorXd, VectorXd))
{
  MatrixXd outputError = costFuncPrime(output, desiredOutput).array() * layers.back().weightedValue.unaryExpr(activationPrime).array();
  
  MatrixXd lastError = outputError;
  
  //iterate back through the layers
  for(int i = layers.size() - 2; i > 0; i--)
  { 
    MatrixXd layerError = (layers[i + 1].weights.transpose() * lastError).array() * layers[i].weightedValue.unaryExpr(activationPrime).array();
    
    layers[i].bias -= layerError;
    layers[i].weights -= layerError * layers[i-1].activatedValue.transpose();
    lastError = layerError;
  }
  int i = 0;
  MatrixXd layerError = (layers[i + 1].weights.transpose() * lastError).array() * layers[i].weightedValue.unaryExpr(activationPrime).array();
  layers[i].bias -= layerError;
  layers[i].weights -= layerError * input.transpose();
    
}
