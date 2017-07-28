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

	void ForwardPropagate(vector<double> input);
	void BackwardPropagate(vector<double> deltas);

	vector<double> ComputeWeightedSum(vector<double> input);

	vector<vector<double>> weights;
	vector<double> bias;

	vector<Neuron*> listNeurons;

	Layer* connectedLayer;

	ActivationFunction * func = nullptr;
};

