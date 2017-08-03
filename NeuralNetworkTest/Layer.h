#pragma once

#include "Sigmoid.h"
#include "Neuron.h"

#include <iostream>

#include <Eigen/Dense>

using namespace std;

class Layer
{
public:
	Layer();
	~Layer();

	void Init(int numInput, int numOutput);

	vector<double> GetLayerInfo();

	void FillWeights(vector<double> weightBuffer, int currentIdx);

	void ForwardPropagate(Eigen::VectorXd input, double y);

	void BackwardPropagate(Eigen::VectorXd deltas);

	void SetDeltaToZero();

	void UpdateWeights();

	void UpdatePartialDerivative();

	Eigen::VectorXd Evaluate(Eigen::VectorXd X);

	double GetWeightSum();

	double ComputeLoss(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::MatrixXd yPred);

	void SetNumSamples(int numSamples);

	
	Eigen::MatrixXd WeightMatrix;

	Eigen::VectorXd VectorZ;
	Eigen::VectorXd ActivationVector;

	Eigen::MatrixXd DeltaMatrix;
	Eigen::MatrixXd PartialDerivatives;

	Layer* connectedLayer = nullptr;
	Layer * previousLayer = nullptr;

	ActivationFunction * Func = nullptr;

	double regLambda = 0.01; // Learning rate
	double epsilon = 0.01; // Regularization strength

	bool isHiddenLayer = true;

	int numSamples;
};

