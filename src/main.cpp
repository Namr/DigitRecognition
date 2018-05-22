#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "network.hpp"

using namespace Eigen;
using namespace std;

double sigmoid(double x);
double quadraticCost(VectorXd givenOutput, VectorXd desiredOutput);

double sigmoidPrime(double x);
MatrixXd quadraticCostPrime(VectorXd givenOutput, VectorXd DesiredOutput);

vector<VectorXd> readDigitImages(const char *filepath);
vector<VectorXd> readDigitLabels(const char *filepath);

template <class T> void endswap(T *objp);

int main(int argc, char **argv) {

  vector<VectorXd> XORInput;
  vector<VectorXd> XOROutput;

  VectorXd I1(2);
  VectorXd O1(1);

  I1 << 0, 0;
  O1 << 0;
  XORInput.push_back(I1);
  XOROutput.push_back(O1);

  I1 << 0, 1;
  O1 << 1;
  XORInput.push_back(I1);
  XOROutput.push_back(O1);

  I1 << 1, 0;
  O1 << 1;
  XORInput.push_back(I1);
  XOROutput.push_back(O1);

  I1 << 1, 1;
  O1 << 0;
  XORInput.push_back(I1);
  XOROutput.push_back(O1);

  int shape[] = {2, 1};
  Network network(&shape[0], sizeof(shape) / sizeof(int), XORInput[0],
                  XOROutput[0]);

  network.minibatch(&XORInput[0], &XOROutput[0], 3.0, 0, 4, sigmoid,
                    quadraticCost, sigmoidPrime, quadraticCostPrime);

  network.setInput(XORInput[0], XOROutput[0]);
  network.forwardProp(&sigmoid, &quadraticCost);

  cout << "TEST PASS" << endl;
  cout << network.output << endl;

  cout << "COST:" << endl;
  cout << network.cost << endl;

  cout << "THE REAL DEAL" << endl;
  cout << network.desiredOutput << endl;

  /*
  //read digits from the database put them into Vectors
  vector<VectorXd> digitImages = readDigitImages("train-images.idx3-ubyte");
  vector<VectorXd> digitLabels = readDigitLabels("train-labels.idx1-ubyte");

  int shape[] = {16, 16, 10};
  Network network(&shape[0], sizeof(shape)/ sizeof(int), digitImages[0],
  digitLabels[0]);

  for(int i = 0; i < 600; i++)
  {
    network.minibatch(&digitImages[0], &digitLabels[0], 3.0, i*10,10, sigmoid,
  quadraticCost, sigmoidPrime, quadraticCostPrime);
  }

  network.setInput(digitImages[2001], digitLabels[2001]);
  network.forwardProp(&sigmoid, &quadraticCost);

  cout << "TEST PASS" << endl;
  cout << network.output << endl;

  cout << "COST:" << endl;
  cout << network.cost << endl;

  cout << "THE REAL DEAL" << endl;
  cout << digitLabels[2001] << endl;

  return 0;
  */
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

// same as, Cost = 1/2 * |y - aL|^2 where y is disired output and aL is actual
// output
double quadraticCost(VectorXd givenOutput, VectorXd desiredOutput) {
  MatrixXd difference = givenOutput - desiredOutput;
  return difference.array().abs2().sum() * 0.5;
}

double sigmoidPrime(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

MatrixXd quadraticCostPrime(VectorXd givenOutput, VectorXd desiredOutput) {
  return (givenOutput - desiredOutput);
}

vector<VectorXd> readDigitImages(const char *filepath) {
  // init file stream
  ifstream imageFile(filepath, ios::in | ios::binary);
  vector<VectorXd> digits;

  // if file not found, error out
  if (imageFile.fail()) {
    cout << "ERROR: Image not found" << std::endl;
    return digits;
  }

  // magic constant to ensure that endianness is correct
  int32_t magic;
  imageFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  endswap(&magic);

  uint32_t imageCount;
  imageFile.read(reinterpret_cast<char *>(&imageCount), sizeof(imageCount));
  endswap(&imageCount);

  uint32_t rowSize;
  imageFile.read(reinterpret_cast<char *>(&rowSize), sizeof(rowSize));
  endswap(&rowSize);

  uint32_t colSize;
  imageFile.read(reinterpret_cast<char *>(&colSize), sizeof(colSize));
  endswap(&colSize);

  // iterate over every digit image, then iterate over all their rows and cols
  // until a flattened vector is all that is left
  for (unsigned int i = 0; i < imageCount; i++) {
    VectorXd inputLayer(rowSize * colSize);
    for (unsigned int row = 0; row < rowSize; row++) {
      for (unsigned int col = 0; col < colSize; col++) {
        // place data into a char buffer and then reinterpret into an unsigned
        // char
        char pix[1];
        imageFile.read(pix, 1);
        endswap(&pix);

        // value inside the file is between 0 and 255, our net expected a value
        // between 0 and 1
        inputLayer((row * rowSize) + col) =
            (double)reinterpret_cast<unsigned char &>(pix[0]) / 255;
      }
    }
    digits.push_back(inputLayer);
  }
  return digits;
}

vector<VectorXd> readDigitLabels(const char *filepath) {
  // init file stream
  ifstream imageFile(filepath, ios::in | ios::binary);
  vector<VectorXd> labels;

  // if file not found, error out
  if (imageFile.fail()) {
    cout << "ERROR: Image not found" << std::endl;
    return labels;
  }

  // magic constant to ensure that endianness is correct
  int32_t magic;
  imageFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  endswap(&magic);

  uint32_t labelCount;
  imageFile.read(reinterpret_cast<char *>(&labelCount), sizeof(labelCount));
  endswap(&labelCount);

  for (unsigned int i = 0; i < labelCount; i++) {
    // place data into a char buffer and then reinterpret into an unsigned char
    char value[1];
    imageFile.read(value, 1);
    endswap(&value);

    unsigned char label = reinterpret_cast<unsigned char &>(value[0]);
    VectorXd output = VectorXd::Zero(10);
    output(label) = 1.0;

    labels.push_back(output);
  }
  return labels;
}

// reverses endianness
template <class T> void endswap(T *objp) {
  unsigned char *memp = reinterpret_cast<unsigned char *>(objp);
  std::reverse(memp, memp + sizeof(T));
}
