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

//reverses endianness
template <class T>
void endswap(T *objp);
vector<VectorXd> readDigitImages(const char* filepath);

double sigmoid(double x);
VectorXd forwardProp(MatrixXd *weightLayers, VectorXd *biasLayers, int *layerWidth, VectorXd startLayer, int layerCount);

int main(int argc, char **argv)
{
  
  //goal of this project is to have a standard NN with 2 hidden layers with 16 nodes in each layer
  //should be able to identify the digits from the database
  int layerCount = 3;
  int layerWidth[layerCount] = {16, 16, 10};

  //read digits from the database put them into Vectors
  vector<VectorXd> digitImages = readDigitImages("train-images.idx3-ubyte");
  
  //init arrays of hiddenlayer matricies based on the # of hidden layers
  MatrixXd weightLayers[layerCount];
  VectorXd biasLayers[layerCount];

  VectorXd output = forwardProp(&weightLayers[0], &biasLayers[0], &layerWidth[0], digitImages[0], layerCount);
  
  cout << "FIRST PASS" << endl;
  cout << output << endl;

  output = forwardProp(&weightLayers[0], &biasLayers[0], &layerWidth[0], digitImages[0], layerCount);

  cout << "SECOND PASS" << endl;
  cout << output << endl;
  return 0;
}

double sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

VectorXd forwardProp(MatrixXd *weightLayers, VectorXd *biasLayers, int *layerWidth, VectorXd startLayer, int layerCount)
{
  for(int i = 0; i < layerCount; i++)
  {
    if(weightLayers[i].isZero(0))
    {
      weightLayers[i] = MatrixXd::Random(layerWidth[i], startLayer.rows());
      biasLayers[i] = VectorXd::Random(layerWidth[i]);
    }
    startLayer = (weightLayers[i] * startLayer) + biasLayers[i];
    startLayer = startLayer.unaryExpr(&sigmoid);
  }
  return startLayer;
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

  //iterate over every digit image, then iterate over all their rows and cols until a flattened vector is all that is left
  for(int i = 0; i < imageCount; i++)
  {
    VectorXd inputLayer(rowSize * colSize);
    //loop through MRI file and seperate each pixel into its x,y,z cordinate
    for (int row = 0; row < rowSize; row++)
    {
      for (int col = 0; col < colSize; col++)
      {
        //place data into a char buffer and then reinterpret into an unsigned char
        char pix[1];
        imageFile.read(pix, 1);
        endswap(&pix);

        inputLayer((row * rowSize) + col) = (double) reinterpret_cast<unsigned char&>(pix[0]) / 255;
      }
    }
    digits.push_back(inputLayer);
  }
  return digits;
}

//reverses endianness
template <class T>
void endswap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}
