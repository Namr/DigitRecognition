#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "network.hpp"

using namespace Eigen;
using namespace std;

double sigmoid(double x);
double quadraticCost(VectorXd givenOutput, VectorXd desiredOutput);

double sigmoidPrime(double x);
MatrixXd quadraticCostPrime(VectorXd givenOutput, VectorXd DesiredOutput);

vector<VectorXd> readDigitImages(const char* filepath);
vector<VectorXd> readDigitLabels(const char* filepath);

template <class T>
void endswap(T *objp);

int main(int argc, char **argv)
{ 
  //read digits from the database put them into Vectors
  vector<VectorXd> digitImages = readDigitImages("train-images.idx3-ubyte");
  vector<VectorXd> digitLabels = readDigitLabels("train-labels.idx1-ubyte");
  
  int shape[] = {16, 16, 10};
  Network network(&shape[0], sizeof(shape)/ sizeof(int), digitImages[0], digitLabels[0]);
  
  network.forwardProp(&sigmoid, &quadraticCost);
  cout << "FIRST PASS" << endl;
  cout << network.output << endl;
  
  cout << "COST:" << endl;
  cout << network.cost << endl;

  network.backProp(&sigmoidPrime, &quadraticCostPrime);
  
  network.forwardProp(&sigmoid, &quadraticCost);
  cout << "SECOND PASS" << endl;
  cout << network.output << endl;
  
  cout << "COST:" << endl;
  cout << network.cost << endl;
  
  return 0;
}

double sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

//same as, Cost = 1/2 * |y - aL|^2 where y is disired output and aL is actual output
double quadraticCost(VectorXd givenOutput, VectorXd desiredOutput)
{
  MatrixXd difference = desiredOutput - givenOutput;
  return difference.array().abs2().sum() * 0.5;
}

double sigmoidPrime(double x)
{
  return sigmoid(x) * sigmoid(1 - x);
}

MatrixXd quadraticCostPrime(VectorXd givenOutput, VectorXd desiredOutput)
{
  return (givenOutput - desiredOutput);
}

vector<VectorXd> readDigitImages(const char* filepath)
{
  //init file stream
  ifstream imageFile (filepath, ios::in | ios::binary);
  vector<VectorXd> digits;

  //if file not found, error out
  if(imageFile.fail())
  {
    cout << "ERROR: Image not found" << std::endl;
    return digits;
  }

  //magic constant to ensure that endianness is correct
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

  //iterate over every digit image, then iterate over all their rows and cols until a flattened
  //vector is all that is left
  for(int i = 0; i < imageCount; i++)
  {
    VectorXd inputLayer(rowSize * colSize);
    for (int row = 0; row < rowSize; row++)
    {
      for (int col = 0; col < colSize; col++)
      {
        //place data into a char buffer and then reinterpret into an unsigned char
        char pix[1];
        imageFile.read(pix, 1);
        endswap(&pix);
	
	//value inside the file is between 0 and 255, our net expected a value between 0 and 1
        inputLayer((row * rowSize) + col) = (double) reinterpret_cast<unsigned char&>(pix[0]) / 255;
      }
    }
    digits.push_back(inputLayer);
  }
  return digits;
}

vector<VectorXd> readDigitLabels(const char* filepath)
{
  //init file stream
  ifstream imageFile (filepath, ios::in | ios::binary);
  vector<VectorXd> labels;

  //if file not found, error out
  if(imageFile.fail())
  {
    cout << "ERROR: Image not found" << std::endl;
    return labels;
  }

  //magic constant to ensure that endianness is correct
  int32_t magic;
  imageFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  endswap(&magic);

  uint32_t labelCount;
  imageFile.read(reinterpret_cast<char *>(&labelCount), sizeof(labelCount));
  endswap(&labelCount);

  for(int i = 0; i < labelCount; i++)
  {
    //place data into a char buffer and then reinterpret into an unsigned char
    char value[1];
    imageFile.read(value, 1);
    endswap(&value);

    unsigned char label = reinterpret_cast<unsigned char&>(value[0]);
    VectorXd output(10);
    output(label) = 1.0;
    
    labels.push_back(output);
  }
  return labels;
}

//reverses endianness
template <class T>
void endswap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}
