#include "network.hpp"

Layer::Layer(int neuronCount, int inputLength) {
  weights = MatrixXd::Random(neuronCount, inputLength);
  bias = VectorXd::Random(neuronCount);

  sumDeltaW = MatrixXd::Zero(neuronCount, inputLength);
  sumDeltaB = VectorXd::Zero(neuronCount);
}

Network::Network(int *shape, int layerCount, VectorXd inputs) {
  input = inputs;
  layerShape = shape;

  for (int i = 0; i < layerCount; i++) {

    Layer l(shape[i], input.rows());

    if (i > 0) {
      l = Layer(shape[i], shape[i - 1]);
    }

    layers.push_back(l);
  }
}

Network::Network(int *shape, int layerCount, VectorXd inputs,
                 VectorXd labledOutputs) {
  input = inputs;
  layerShape = shape;
  desiredOutput = labledOutputs;

  for (int i = 0; i < layerCount; i++) {

    Layer l(shape[i], input.rows());

    if (i > 0) {
      l = Layer(shape[i], shape[i - 1]);
    }

    layers.push_back(l);
  }
}

void Network::setInput(VectorXd inputs) { input = inputs; }

void Network::setInput(VectorXd inputs, VectorXd outputs) {
  input = inputs;
  desiredOutput = outputs;
}

// pushes data forward through the network, starts with the first layer
// and multiplies the data layer by the wieghts and adds the biases of the next
// layer that layer is now the new output, which is then multiplied by the
// weights and biases of the next layer and so on, each layer also goes through
// an activation function which is passed
void Network::forwardProp(double (*activation)(double),
                          double (*costFunc)(VectorXd, VectorXd)) {
  output = input;
  for (Layer &layer : layers) {
    layer.weightedValue = (layer.weights * output) + layer.bias;
    layer.activatedValue = layer.weightedValue.unaryExpr(activation);
    output = layer.activatedValue;
  }

  cost = costFunc(output, desiredOutput);
}

void Network::backProp(double (*activationPrime)(double),
                       MatrixXd (*costFuncPrime)(VectorXd, VectorXd)) {
  // calculate the error of the output layer,
  // based on the equation error = dirivitive of cost func * (element wise mult)
  // the derivative of activation function(original weighted value)
  MatrixXd error =
      costFuncPrime(output, desiredOutput).array() *
      layers.back().weightedValue.unaryExpr(activationPrime).array();
  layers.back().sumDeltaB += error;
  layers.back().sumDeltaW +=
      error * layers[layers.size() - 2].activatedValue.transpose();

  // iterate back through the layers, and calculate their error in a similar
  // way, based on the equation (transpose of the last weight matrix * last
  // error) *(element wise) the dirivitive of activation function(original
  // weighted value)
  for (int i = layers.size() - 2; i > 0; i--) {
    error = (layers[i + 1].weights.transpose() * error).array() *
            layers[i].weightedValue.unaryExpr(activationPrime).array();

    layers[i].sumDeltaB += error;
    layers[i].sumDeltaW += error * layers[i - 1].activatedValue.transpose();
  }

  // last case is special because input layer is not included in layers list so
  // it must be calculated alone
  int i = 0;
  error = (layers[i + 1].weights.transpose() * error).array() *
          layers[i].weightedValue.unaryExpr(activationPrime).array();

  layers[i].sumDeltaB += error;
  layers[i].sumDeltaW += error * input.transpose();
}

void Network::minibatch(VectorXd *inputs, VectorXd *labels, double learningRate,
                        int trainingStart, int trainingSize,
                        double (*activation)(double),
                        double (*costFunc)(VectorXd, VectorXd),
                        double (*activationPrime)(double),
                        MatrixXd (*costFuncPrime)(VectorXd, VectorXd)) {

  for (Layer &layer : layers) {
    layer.sumDeltaW *= 0;
    layer.sumDeltaB *= 0;
  }

  for (int i = trainingStart; i < trainingSize; i++) {
    setInput(inputs[i], labels[i]);
    forwardProp(activation, costFunc);
    backProp(activationPrime, costFuncPrime);
  }

  for (Layer &layer : layers) {
    layer.weights -= (learningRate / trainingSize) * layer.sumDeltaW;
    layer.bias -= (learningRate / trainingSize) * layer.sumDeltaB;
  }
}
