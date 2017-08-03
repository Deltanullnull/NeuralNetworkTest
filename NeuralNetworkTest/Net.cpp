#include "Net.h"



Net::Net()
{
}


Net::~Net()
{
}

void Net::ReadFromFile(string fileName)
{
	FILE * file;

	fopen_s(&file, fileName.c_str(), "rb");

	if (file)
	{
		fseek(file, 0L, SEEK_END);
		size_t fileSize = ftell(file);

		rewind(file);

		int numValues = fileSize / sizeof(double);

		vector<double> values(numValues, 0);

		fread_s(values.data(), fileSize, sizeof(double), numValues, file);

		fclose(file);

		firstHiddenLayer = new Layer();
		firstHiddenLayer->FillWeights(values, 0);
	}
}

void Net::SaveAsFile(string fileName)
{
	FILE * file;
	
	fopen_s(&file, fileName.c_str(), "wb");

	if (file)
	{
		if (firstHiddenLayer != nullptr)
		{
			vector<double> layerInfo = firstHiddenLayer->GetLayerInfo();

			fwrite(layerInfo.data(), sizeof(double), layerInfo.size(), file);
		}

		fclose(file);
	}
}

void Net::Build(int numInput, vector<int> layerSizes, int numClasses)
{
	Layer * current = nullptr;

	this->numClasses = numClasses;

	int numHiddenLayers = layerSizes.size();

	int numOutput = layerSizes.at(0);

	firstHiddenLayer = new Layer();
	firstHiddenLayer->Init(numInput, numOutput);

	current = firstHiddenLayer;

	for (int i = 0; i < numHiddenLayers - 1; i++)
	{
		Layer * connected = new Layer();
		numInput = layerSizes.at(i);
		numOutput = layerSizes.at(i + 1);
		connected->Init(numInput, numOutput);

		current->connectedLayer = connected;
		connected->previousLayer = current;

		current = connected;
	}

	Layer * finalHiddenLayer = new Layer();
	numInput = *(layerSizes.end() - 1);
	numOutput = numClasses;

	finalHiddenLayer->Init(numInput, numOutput);
	current->connectedLayer = finalHiddenLayer;
	finalHiddenLayer->previousLayer = current;


	Layer * finalLayer = new Layer();
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

double Net::Score(Eigen::MatrixXd X, Eigen::VectorXd y)
{
	double score = 0.0;

	if (firstHiddenLayer != nullptr)
	{
		for (int i = 0; i < X.rows(); i++)
		{
			Eigen::VectorXd yPredVector = firstHiddenLayer->Evaluate(X.row(i));
			
			// Get index of max

			double yPred = 0.0;
			int maxIdx = 0;

			for (int j = 0; j < yPredVector.rows(); j++)
			{
				if (yPredVector(j) > yPred)
				{
					yPred = yPredVector(j);
					maxIdx = j;
				}
			}

			if (maxIdx == (int)(y(i)))
			{
				score += 1.0;
			}

		}
	}

	return score / X.rows();
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


void Net::SetNumSamples(int numSamples)
{
	firstHiddenLayer->SetNumSamples(numSamples);
}

void Net::SetLambda(double lambda)
{
	if (firstHiddenLayer != nullptr)
	{
		firstHiddenLayer->SetLambda(lambda);
	}
}

void Net::SetEpsilon(double epsilon)
{
	if (firstHiddenLayer != nullptr)
	{
		firstHiddenLayer->SetEpsilon(epsilon);
	}
}
