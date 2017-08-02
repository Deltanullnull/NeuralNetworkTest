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

	this->numClasses = numClasses;

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

Eigen::VectorXd Net::Evaluate(Eigen::VectorXd X)
{
	if (firstHiddenLayer == nullptr)
	{
		cerr << "Error! No layers defined." << endl;
	}

	return firstHiddenLayer->Evaluate(X);
}


void Net::Train(Eigen::MatrixXd X, Eigen::VectorXd y, int numEpochs)
{
	if (firstHiddenLayer != nullptr)
	{
		
		int numTrainingData = X.rows();

		SetNumSamples(numTrainingData);

		for (int e = 0; e < numEpochs; e++)
		{
			firstHiddenLayer->SetDeltaToZero();

			
			for (int i = 0; i < numTrainingData; i++)
			{
				Eigen::VectorXd currentSample = X.row(i);
				double currentLabel = y(i);

				firstHiddenLayer->ForwardPropagate(currentSample, currentLabel);

				firstHiddenLayer->UpdateWeights();
			}

			if (e % (numEpochs / 100) == 0)
			{
				cout << "Loss: " << ComputeLossFunction(X, y, numClasses) << endl;
			}
		}

	}
}

double Net::ComputeLossFunction(Eigen::MatrixXd X, Eigen::VectorXd y, int numClasses)
{
	int numSamples = X.rows();

	

	if (firstHiddenLayer != nullptr)
	{
		double val = firstHiddenLayer->GetWeightSum() * firstHiddenLayer->regLambda / (2.0 * numSamples);

		//  get sum of all weights

		for (int i = 0; i < X.rows(); i++)
		{
			Eigen::VectorXd prediction = firstHiddenLayer->Evaluate(X.row(i));

			Eigen::VectorXd VectorY = Eigen::VectorXd::Zero(numClasses);

			int label = y[i];

			VectorY(label) = 1.0;

			for (int c = 0; c < numClasses; c++)
			{
				val += (VectorY(c) * log(prediction(c)) + (1 - VectorY(c)) * log(1 - prediction(c))) / (double) (-numSamples);
			}

		}

		return val;
	}

	return 1e10;
}

double Net::ComputeLossFunction(vector<vector<double>> X, vector<double> y)
{
	double cost = 0.0;
	int numSamples = X.size();

	if (firstHiddenLayer != nullptr)
	{
		//cost = firstHiddenLayer->ComputeLoss(X, y, X);
	}

	return cost / numSamples;
}

void Net::SetNumSamples(int numSamples)
{
	firstHiddenLayer->SetNumSamples(numSamples);
}
