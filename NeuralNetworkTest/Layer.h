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

	void ForwardPropagate(vector<double> input);
	void BackwardPropagate(vector<double> deltas);

	vector<double> ComputeWeightedSum(vector<double> input);

	vector<vector<double>> weights;
	vector<double> bias;

	Layer* connectedLayer;
	Layer * previousLayer;

	ActivationFunction * func = nullptr;
};

