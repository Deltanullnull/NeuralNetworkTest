#include "Net.h"



Net::Net()
{
}


Net::~Net()
{
}

void Net::Build(int numInput, vector<int> layerSizes)
{
	Layer * current = nullptr;

	int numHiddenLayers = layerSizes.size();

	int numOutput = layerSizes.at(0);

	firstHiddenLayer = new Layer();
	firstHiddenLayer->Init(numInput, numOutput);

	current = firstHiddenLayer;

	for (int i = 1; i < numHiddenLayers; i++)
	{
		Layer * connected = new Layer();
		numInput = numOutput;
		numOutput = layerSizes.at(i);
		connected->Init(numInput, numOutput);

		current->connectedLayer = connected;
		connected->previousLayer = current;

		current = connected;
	}
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
		firstHiddenLayer->SetNumSamples(X.rows());

		//for (int e = 0; e < numEpochs; e++)
		{
			int numTrainingData = X.size();
			for (int i = 0; i < numTrainingData; i++)
			{
				firstHiddenLayer->ForwardPropagate(X.row(i), y(i));
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
