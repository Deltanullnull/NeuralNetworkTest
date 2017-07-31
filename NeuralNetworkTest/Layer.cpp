#include "Layer.h"
#include <random>
#include "MathOperations.h"

Layer::Layer()
{
	Func = new Sigmoid();
}


Layer::~Layer()
{
}

void Layer::Init(int numInput, int numOutput)
{
	weights.resize(numInput);
	Deltas.resize(numInput);

	for (vector<double>& weightVector : weights)
	{
		weightVector.resize(numOutput);
		Deltas.resize(numOutput);
	}

	bias.resize(numOutput);

	zVector.resize(numOutput);
	
	InitWeights();
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
			Deltas.at(y).at(x) = 0;

			if (y == 0)
			{
				bias.at(x) = dis(gen);
			}
		}
	}
}

void Layer::ForwardPropagate(vector<double> input, double y)
{
	// TODO compute values for this layer
	if (connectedLayer != nullptr)
	{
		vector<double> output = ComputeWeightedSum(input);

		// continue forward propagation on next layer
		connectedLayer->ForwardPropagate(output, y);
	}
	else
	{
		// This is the last layer

		// TODO compute cost function

		vector<double> h = ComputeWeightedSum(input);

		vector<double> yVector = vector<double>(h.size(), 0.0);
		int yIdx = y;
		yVector.at(yIdx) = 1.0;

		vector<double> deltas = MathOperations::VectorSubtraction(h, yVector);

		previousLayer->BackwardPropagate(deltas);
	}
	

}

void Layer::BackwardPropagate(vector<double> deltas)
{
	vector<double> deltasCurrent;

	deltasCurrent = MathOperations::MatrixMultiplication(weights, deltas, true);

	vector<double> gDeriv = Func->Derivative(zVector);

	deltasCurrent = MathOperations::ElementwiseMultiplication(deltasCurrent, gDeriv);

	
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

	// z = W * x + b
	zVector = MathOperations::MatrixMultiplication(weights, input, false);
	zVector = MathOperations::VectorAddition(zVector, bias);

	// a = g(z)
	output = Func->Compute(zVector);

	return output;
}
