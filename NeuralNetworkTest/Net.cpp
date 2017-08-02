#include "Net.h"



Net::Net()
{
}


Net::~Net()
{
}

void Net::Build(int numInput, vector<int> layerSizes, int numClasses)
{
	Layer * current = nullptr;

	int numHiddenLayers = layerSizes.size();

	int numOutput = layerSizes.at(0);

	firstHiddenLayer = new Layer(true);
	firstHiddenLayer->Init(numInput, numOutput);

	current = firstHiddenLayer;

	for (int i = 0; i < numHiddenLayers - 1; i++)
	{
		Layer * connected = new Layer(true);
		numInput = layerSizes.at(i);
		numOutput = layerSizes.at(i + 1);
		connected->Init(numInput, numOutput);

		current->connectedLayer = connected;
		connected->previousLayer = current;

		current = connected;
	}

	Layer * finalHiddenLayer = new Layer(true);
	numInput = *(layerSizes.end() - 1);
	numOutput = numClasses;

	finalHiddenLayer->Init(numInput, numOutput);
	current->connectedLayer = finalHiddenLayer;
	finalHiddenLayer->previousLayer = current;


	Layer * finalLayer = new Layer(true);
	numInput = numClasses;
	numOutput = -1;

	finalLayer->Init(numInput, numOutput);
	finalHiddenLayer->connectedLayer = finalLayer;
	finalLayer->previousLayer = finalHiddenLayer;
}

void Net::Train(vector<vector<double>> X, vector<double> y, int numEpochs)
{
	if (firstHiddenLayer != nullptr)
	{
		for (int e = 0; e < numEpochs; e++)
		{
			int numTrainingData = X.size();

			firstHiddenLayer->SetDeltaToZero();

			for (int i = 0; i < numTrainingData; i++)
			{
				firstHiddenLayer->ForwardPropagate(X.at(i), y.at(i));

			}
		}

	}
}

void Net::Train(Eigen::MatrixXd X, Eigen::VectorXd y, int numEpochs)
{
	if (firstHiddenLayer != nullptr)
	{
		
		for (int e = 0; e < numEpochs; e++)
		{
			firstHiddenLayer->SetDeltaToZero();

			int numTrainingData = X.size();
			for (int i = 0; i < numTrainingData; i++)
			{
				firstHiddenLayer->ForwardPropagate(X.row(i), y(i));

				firstHiddenLayer->UpdateWeights(numTrainingData);
			}
		}

	}
}

double Net::ComputeLossFunction(vector<vector<double>> X, vector<double> y)
{
	double cost = 0.0;
	int numSamples = X.size();

	if (firstHiddenLayer != nullptr)
	{
		cost = firstHiddenLayer->ComputeLoss(X, y, X);
	}

	return cost / numSamples;
}
