#include "Net.h"
#include "MathOperations.h"
#include <iostream>

int main()
{
	Net * neuralNetwork = new Net();

	int numClasses = 2;
	int numFeatures = 2;
	int numSamples = 5;

	Eigen::MatrixXd X(numSamples, numFeatures);
	Eigen::VectorXd y(numSamples);

	X(0, 0) = 2.0; X(0, 1) = 3.0;
	X(1, 0) = 1.0; X(1, 1) = 0.0;
	X(2, 0) = 1.0; X(2, 1) = 2.0;
	X(3, 0) = 3.0; X(3, 1) = 3.0;
	X(4, 0) = 0.0; X(4, 1) = 0.0;

	y(0) = 1;
	y(1) = 0;
	y(2) = 1;
	y(3) = 1;
	y(4) = 0;

	Eigen::MatrixXd outerProduct = y * y.transpose();


	cout << "Outer product: " << endl << outerProduct << endl;
	

	vector<int> hiddenLayerSizes;

	//layerSizes.push_back(numFeatures);
	hiddenLayerSizes.push_back(4);
	hiddenLayerSizes.push_back(3);
	//layerSizes.push_back(numClasses);

	neuralNetwork->Build(numFeatures, hiddenLayerSizes, numClasses);

	neuralNetwork->Train(X, y);


	return 0;
}