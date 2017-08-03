#pragma once

#include "Layer.h"
#include "JacobiFunction.h"



class Net
{
public:
	Net();
	~Net();

	void ReadFromFile(string fileName);
	void SaveAsFile(string fileName);

	void Build(int numInput, vector<int> layerSizes, int numClasses);

	Eigen::VectorXd Evaluate(Eigen::VectorXd X);

	void Train(Eigen::MatrixXd X, Eigen::VectorXd y, int numEpochs = 10000);

	double ComputeLossFunction(Eigen::MatrixXd X, Eigen::VectorXd y, int numClasses);

	double ComputeLossFunction(vector<vector<double>> X, vector<double> y);

	void SetNumSamples(int numSamples);

	Layer* firstHiddenLayer = nullptr;
	CostFunction * CostFunc;

	int numClasses = 0;
};

