#pragma once

#include "Layer.h"
#include "JacobiFunction.h"



class Net
{
public:
	Net();
	~Net();

	void Build(int numInput, vector<int> layerSizes);

	void Train(vector<vector<double>> X, vector<double> y, int numEpochs = 2000);

	void Train(Eigen::MatrixXd X, Eigen::VectorXd y, int numEpochs = 2000);

	double ComputeLossFunction(vector<vector<double>> X, vector<double> y);

	Layer* firstHiddenLayer = nullptr;
	CostFunction * CostFunc;
};

