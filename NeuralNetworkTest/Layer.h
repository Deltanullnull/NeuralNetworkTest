#pragma once

#include "Sigmoid.h"
#include "Neuron.h"

using namespace std;

class Layer
{
public:
	Layer();
	~Layer();

	void Init(int numInput, int numOutput);
	void InitWeights();

	void ForwardPropagate(vector<double> input, double y);
	void BackwardPropagate(vector<double> deltas);

	vector<double> ComputeWeightedSum(vector<double> input);

	vector<vector<double>> weights;
	vector<double> bias;

	vector<double> zVector;

	vector<vector<double>> Deltas;

	Layer* connectedLayer = nullptr;
	Layer * previousLayer = nullptr;

	ActivationFunction * Func = nullptr;

	double regLambda = 0.01; // Learning rate
	double epsilon = 0.01; // Regularization strength
};

