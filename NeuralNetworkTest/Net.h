#pragma once

#include "Layer.h"

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
	double Score(Eigen::MatrixXd X, Eigen::VectorXd y);

	double ComputeLossFunction(Eigen::MatrixXd X, Eigen::VectorXd y, int numClasses);

	void SetNumSamples(int numSamples);

	void SetLambda(double lambda);

	void SetEpsilon(double epsilon);

	Layer* firstHiddenLayer = nullptr;

	int numClasses = 0;
};

