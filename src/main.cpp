#include <iostream>
#include <stdio.h>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{
  
    VectorXd inputLayer(784);
    MatrixXd HiddenLayer(16, 784);
    
    ifstream imageFile ("train-images-idx3-ubyte.gz", ios::in | ios::binary);

    if(imageFile.fail())
    {
      cout << "ERROR: Image not found" << std::endl;
      return 1;
    }
    
    char magic[4];
    imageFile.read(magic, 4);

    cout << magic << std::endl;
    
    char imageCount[4];
    imageFile.read(imageCount, 4);

    char rowSize[4];
    imageFile.read(rowSize, 4);

    char colSize[4];
    imageFile.read(colSize, 4);

    //loop through MRI file and seperate each pixel into its x,y,z cordinate
    for (int row = 0; row < (int) *(rowSize); row++)
    {
      for (int col = 0; col < (int) *(colSize); col++)
        {
            //place data into a char buffer and then reinterpret into an unsigned char
            char pix[1];
            imageFile.read(pix, 1);
	    inputLayer(col * row) = (double) reinterpret_cast<unsigned char&>(pix[0]) / 255;
        }
    }

    cout << "Matrix: " << inputLayer << std::endl;
    
    return 0;
}
