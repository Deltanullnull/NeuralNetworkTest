#pragma once

#include "Sigmoid.h"
#include "Neuron.h"

#include <iostream>

#include <Eigen/Dense>

using namespace std;

class Layer
{
public:
	Layer(bool isHiddenLayer);
	~Layer();

	void Init(int numInput, int numOutput);
	void InitWeights();

	void ForwardPropagate(vector<double> input, double y);

	void ForwardPropagate(Eigen::VectorXd input, double y);

	void BackwardPropagate(vector<double> deltas);

	void BackwardPropagate(Eigen::VectorXd deltas);

	void SetDeltaToZero();

	void UpdateWeights(int numSamples);

	void ComputeWeightedSum(vector<double> input, vector<double>& outputActivation, vector<double>& outputZ);

	double ComputeLoss(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::MatrixXd yPred);

	double ComputeLoss(vector<vector<double>> X, vector<double> y, vector<vector<double>> yPred);

	void SetNumSamples(int numSamples);

	
	Eigen::MatrixXd WeightMatrix;
	Eigen::VectorXd BiasVector;

	Eigen::VectorXd VectorZ;
	Eigen::VectorXd ActivationVector;

	Eigen::MatrixXd DeltaMatrix;
	Eigen::MatrixXd PartialDerivatives;

	vector<vector<double>> weights;
	vector<double> bias;

	vector<double> zVector;
	vector<double> aVector;

	vector<vector<double>> Deltas;

	Layer* connectedLayer = nullptr;
	Layer * previousLayer = nullptr;

	ActivationFunction * Func = nullptr;

	double regLambda = 0.01; // Learning rate
	double epsilon = 0.01; // Regularization strength

	bool isHiddenLayer = true;
};

