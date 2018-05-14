#include <iostream>
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <vector>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

//reverses endianness
template <class T>
void endswap(T *objp);


int main(int argc, char **argv)
{

    VectorXd inputLayer(784);
    MatrixXd HiddenLayer(16, 784);

    ifstream imageFile ("train-images.idx3-ubyte", ios::in | ios::binary);

    if(imageFile.fail())
    {
        cout << "ERROR: Image not found" << std::endl;
        return 1;
    }

    int32_t magic;
    imageFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    endswap(&magic);

    cout << magic << std::endl;
     
    uint32_t imageCount;
    imageFile.read(reinterpret_cast<char *>(&imageCount), sizeof(imageCount));
    endswap(&imageCount);

    uint32_t rowSize;
    imageFile.read(reinterpret_cast<char *>(&rowSize), sizeof(rowSize));
    endswap(&rowSize);

    uint32_t colSize;
    imageFile.read(reinterpret_cast<char *>(&colSize), sizeof(colSize));
    endswap(&colSize);

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

    cout << "Matrix: " << inputLayer << std::endl;

    return 0;
}


vector<VectorXd> readDigitImages()
{
}

//reverses endianness
template <class T>
void endswap(T *objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memp, memp + sizeof(T));
}
