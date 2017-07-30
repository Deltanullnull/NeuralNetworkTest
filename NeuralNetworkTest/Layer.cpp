#include "Layer.h"
#include <random>
#include "MathOperations.h"

Layer::Layer()
{
	func = new Sigmoid();
}


Layer::~Layer()
{
}

void Layer::Init(int numInput, int numOutput)
{
	weights.resize(numInput);

	for (vector<double> weightVector : weights)
	{
		weightVector.resize(numOutput);
	}

	
	
}

void Layer::InitWeights()
{
	int numInput = weights.size();

	// randomly initialize values between 0 and 1
	random_device rd;
	mt19937 gen(rd());

	uniform_real_distribution<> dis(0, 1);

	for (int y = 0; y < numInput; y++)
	{
		int numOutput = weights.at(y).size();

		for (int x = 0; x < numOutput; x++)
		{
			weights.at(y).at(x) = dis(gen);
		}
	}

}

void Layer::ForwardPropagate(vector<double> input)
{
	// TODO compute values for this layer

	vector<double> output = ComputeWeightedSum(input);

	// TODO continue forward propagation on next layer
	if (connectedLayer != nullptr)
	{
		connectedLayer->ForwardPropagate(output);
	}

}

void Layer::BackwardPropagate(vector<double> deltas)
{
	vector<double> deltasCurrent;

	deltasCurrent = MathOperations::MatrixMultiplication(weights, deltas, true);

	if (previousLayer != nullptr)
	{
		previousLayer->BackwardPropagate(deltasCurrent);
	}

}

vector<double> Layer::ComputeWeightedSum(vector<double> input)
{
	vector<double> output;
	
	int numOutput = weights.size();
	int numCols = input.size();

	output.resize(numOutput);

	for (int row = 0; row < weights.size(); row++)
	{
		vector<double> currentWeights = weights.at(row);

		double val = 0;

		for (int col = 0; col < numCols; col++)
		{
			val += currentWeights.at(col) * input.at(col);
		}

		output.at(row) = func->Compute(val);
	}

	return output;
}
