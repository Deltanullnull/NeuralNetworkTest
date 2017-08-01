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
	
	//WeightMatrix = Eigen::ArrayXXd::Random(numInput, numOutput) * 0.5 + 0.5;
	WeightMatrix = Eigen::ArrayXXd::Random(numOutput, numInput) * 0.5 + 0.5;

	for (int v = 0; v < WeightMatrix.rows(); v++)
	{
		for (int u = 0; u < WeightMatrix.cols(); u++)
		{
			printf("%lf ", WeightMatrix(v, u));
		}

		printf("\n");
	}

	BiasVector = Eigen::ArrayXd::Random(numOutput) * 0.5 + 0.5;

	for (int v = 0; v < BiasVector.rows(); v++)
	{
		printf("%lf ", BiasVector(v));
	
		printf("\n");
	}

	//DeltaMatrix = Eigen::MatrixXd::Zero(numInput, numOutput);
	DeltaMatrix = Eigen::MatrixXd::Zero(numOutput, numInput);

	PartialDerivatives = Eigen::MatrixXd::Zero(numOutput, numInput);

	VectorZ = Eigen::VectorXd(numOutput);

	/*weights.resize(numInput);
	Deltas.resize(numInput);

	for (vector<double>& weightVector : weights)
	{
		weightVector.resize(numOutput);
		Deltas.resize(numOutput);
	}

	bias.resize(numOutput);

	zVector.resize(numOutput);*/
	
	//InitWeights();
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
		ComputeWeightedSum(input, aVector, zVector);

		// continue forward propagation on next layer
		connectedLayer->ForwardPropagate(aVector, y);
	}
	else
	{
		// This is the last layer

		// TODO compute cost function

		vector<double> h;
		
		ComputeWeightedSum(input, h, zVector);

		vector<double> yVector = vector<double>(h.size(), 0.0);
		int yIdx = y;
		yVector.at(yIdx) = 1.0;

		vector<double> deltas = MathOperations::VectorSubtraction(h, yVector);

		previousLayer->BackwardPropagate(deltas);
	}
	

}

void Layer::ForwardPropagate(Eigen::VectorXd input, double y)
{
	int numOutput = WeightMatrix.rows();
	int numCols = input.size();

	// z = W * x + b
	VectorZ = WeightMatrix * input + BiasVector;

	// a = g(z)
	ActivationVector = Func->Compute(VectorZ);

	if (connectedLayer != nullptr)
	{
		connectedLayer->ForwardPropagate(ActivationVector, y);
	}
	else
	{
		Eigen::VectorXd VectorY = Eigen::VectorXd::Zero(ActivationVector.rows());

		VectorY((int)y) = 1.0;

		Eigen::VectorXd deltaVector = ActivationVector - VectorY;

		previousLayer->BackwardPropagate(deltaVector);
	}
	
}

void Layer::BackwardPropagate(vector<double> deltas)
{
	vector<double> deltasCurrent;

	deltasCurrent = MathOperations::MatrixMultiplication(weights, deltas, true);

	vector<double> gDeriv = Func->Derivative(zVector);

	deltasCurrent = MathOperations::ElementwiseMultiplication(deltasCurrent, gDeriv);

	vector<vector<double>> Temp = MathOperations::VectorMultiplication(deltas, aVector);

	Deltas = MathOperations::MatrixAddition(Deltas, Temp);

	
	if (previousLayer != nullptr)
	{
		previousLayer->BackwardPropagate(deltasCurrent);
	}

}

void Layer::BackwardPropagate(Eigen::VectorXd deltas)
{
	Eigen::VectorXd DeltasCurrent = WeightMatrix.transpose() * deltas;

	Eigen::VectorXd gDeriv = Func->Derivative(DeltasCurrent);

	DeltasCurrent = DeltasCurrent.cwiseProduct(gDeriv);

	DeltaMatrix = DeltaMatrix + deltas * ActivationVector.transpose();

	PartialDerivatives = (DeltaMatrix + regLambda * WeightMatrix);
}

void Layer::SetDeltaToZero()
{
	DeltaMatrix.setZero();

	if (connectedLayer != nullptr)
	{
		connectedLayer->SetDeltaToZero();
	}
}

void Layer::UpdateWeights(int numSamples)
{
	PartialDerivatives = PartialDerivatives / (double) numSamples;

	WeightMatrix = WeightMatrix - epsilon * PartialDerivatives;

	if (connectedLayer != nullptr)
		connectedLayer->UpdateWeights(numSamples);
}

void Layer::ComputeWeightedSum(vector<double> input, vector<double>& outputActivation, vector<double>& outputZ)
{
	vector<double> output;
	
	int numOutput = weights.size();
	int numCols = input.size();

	output.resize(numOutput);

	// z = W * x + b
	outputZ = MathOperations::MatrixMultiplication(weights, input, false);
	outputZ = MathOperations::VectorAddition(outputZ, bias);

	// a = g(z)
	outputActivation = Func->Compute(outputZ);

}

double Layer::ComputeLoss(vector<vector<double>> X, vector<double> y, vector<vector<double>> yPred)
{
	double lambda = 0.01;

	double value = 0.0;

	for (int v = 0; v < weights.size(); v++)
	{
		for (int u = 0; u < weights.at(v).size(); u++)
		{
			double theta = weights.at(v).at(u);

			value += (lambda / 2.0) * pow(theta, 2.0);
		}
	}

	//yPred = MathOperations::MatrixMultiplication

	// compute prediction of y
	vector<vector<double>> newPred;

	for (int idx = 0; idx < yPred.size(); idx++)
	{
		vector<double> activation, z;

		ComputeWeightedSum(yPred.at(idx), activation, z);

		newPred.push_back(activation);
	}

	if (connectedLayer != nullptr)
	{
		value += connectedLayer->ComputeLoss(X, y, newPred);
	}
	else
	{
		for (int v = 0; v < newPred.size(); v++)
		{
			vector<double> yPredVector = newPred.at(v);

			int numClasses = newPred.size();

			vector<double> yTrueVector(numClasses, 0);
			yTrueVector.at(y.at(v)) = 1.0;

			for (int u = 0; u < numClasses; u++)
			{
				value += yTrueVector.at(u) * log(yPredVector.at(u)) + (1.0 - yTrueVector.at(u)) * log(1.0 - yPredVector.at(u));
			}
		}
	}

	

	return value;
}

void Layer::SetNumSamples(int numSamples)
{
	
}
