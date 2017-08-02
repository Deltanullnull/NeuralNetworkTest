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
	//Eigen::VectorXd BiasVector;

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

	int numSamples;
};

