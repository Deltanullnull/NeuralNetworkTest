#include "Layer.h"



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

	// TODO randomly initialize values between 0 and 1

	
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
