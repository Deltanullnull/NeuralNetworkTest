#include "Net.h"
#include <fstream>

int main(int argc, char ** argv)
{
	vector<double> trainingDataVector;
	vector<double> trainingLabels;

	int numClasses = 0;
	int numFeatures = 0;
	int numSamples = 0;

	bool train = true;

	double lambda = 0.01;
	double epsilon = 0.01;

	string modelPath = "";

	if (argc < 2)
	{
		cout << "No CSV file included. Evaluating example." << endl;


		numFeatures = 2;

		trainingDataVector.push_back(2.0); trainingDataVector.push_back(3.0);
		trainingDataVector.push_back(1.0); trainingDataVector.push_back(0.0);
		trainingDataVector.push_back(1.0); trainingDataVector.push_back(2.0);
		trainingDataVector.push_back(3.0); trainingDataVector.push_back(3.0);
		trainingDataVector.push_back(0.0); trainingDataVector.push_back(0.0);

		trainingLabels.push_back(1);
		trainingLabels.push_back(0);
		trainingLabels.push_back(1);
		trainingLabels.push_back(1);
		trainingLabels.push_back(0);
	}
	else
	{
		vector<string> trainingLabelNames;
		vector<string> labelNameMap;

		string mode(argv[1]);

		if (mode == "train")
			train = true;
		else if (mode == "test")
			train = false;
		else
		{
			cout << "Input goes like this: NeuralNetworkTest.ext train|test PATH_TO_YOUR_TRAINING_DATA [model PATH_TO_MODEL]" << endl;
			exit(0);
		}

		string fileName(argv[2]);

		cout << "Training data " << fileName << " loaded." << endl;

		if (argc > 3)
		{
			cout << argv[3] << endl;

			if (string(argv[3]) != "model")
			{
				cout << "Input goes like this: NeuralNetworkTest.ext train|test PATH_TO_YOUR_TRAINING_DATA [model PATH_TO_MODEL]" << endl;
				exit(0);
			}

			modelPath = argv[4];

			if (train && argc > 5)
			{ 
				
				if (string(argv[5]) == "lambda")
				{
					lambda = stod(argv[6]);
					cout << "Lambda: " << lambda << endl;
				}
				if (string(argv[7]) == "epsilon")
				{
					epsilon = stod(argv[8]);
					cout << "Epsilon: " << epsilon << endl;
				}
				
				
			}
		}

		ifstream csvFile(fileName);

		if (!csvFile)
		{
			cout << "File could not be opened." << endl;
			exit(-1);
		}

		string line;

		while (getline(csvFile, line))
		{
			string token;

			if (line.empty())
				break;

			istringstream stream(line);

			int i = 0;

			vector<string> sample;

			while (getline(stream, token, ','))
			{
				sample.push_back(token);

				i++;
			}

			for (int i = 0; i < sample.size() - 1; i++)
			{
				double val = stod(sample.at(i));

				trainingDataVector.push_back(val);
			}

			string labelName = *(sample.end() - 1);

			trainingLabelNames.push_back(labelName);

			if (find(labelNameMap.begin(), labelNameMap.end(), labelName) == labelNameMap.end())
			{
				labelNameMap.push_back(labelName);
			}


			if (numFeatures == 0)
				numFeatures = trainingDataVector.size();

		}

		for (int i = 0; i < trainingLabelNames.size(); i++)
		{
			for (int j = 0; j < labelNameMap.size(); j++)
			{
				if (trainingLabelNames.at(i) == labelNameMap.at(j))
				{
					trainingLabels.push_back(j);
					break;
				}
			}
		}
	}

	Net * neuralNetwork = new Net();


	vector<int> availableLabels;

	for (int i = 0; i < trainingLabels.size(); i++)
	{
		int label = trainingLabels.at(i);

		if (find(availableLabels.begin(), availableLabels.end(), label) == availableLabels.end())
		{
			availableLabels.push_back(label);
		}
	}

	numClasses = availableLabels.size();

	numSamples = trainingDataVector.size() / numFeatures;

	if (numSamples == 0)
	{
		cout << "Data contains no samples. Abort." << endl;
		exit(-1);
	}

	Eigen::MatrixXd X(numSamples, numFeatures);

	for (int i = 0; i < numSamples; i++)
	{
		for (int j = 0; j < numFeatures; j++)
		{
			X(i, j) = trainingDataVector.at(i * numFeatures + j);
		}
	}

	if (!train) // Testing data
	{
		neuralNetwork->ReadFromFile(modelPath);

		for (int i = 0; i < X.rows(); i++)
		{
			cout << "Result for " << X.row(i) << ":" << endl;
			cout << neuralNetwork->Evaluate(X.row(i)) << endl;
		}

		exit(0);
	}


	Eigen::Map<Eigen::VectorXd> y(trainingLabels.data(), numSamples);

	Eigen::MatrixXd shuffledData(numSamples, numFeatures + 1);

	for (int i = 0; i < numSamples; i++)
	{
		for (int j = 0; j < numFeatures; j++)
		{
			shuffledData(i, j) = X(i, j);
		}

		shuffledData(i, numFeatures) = y(i);
	}


	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.rows());
	perm.setIdentity();
	random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());

	shuffledData = perm * shuffledData;

	int numValidationSet = (int)(numSamples * 0.1);
	if (numValidationSet == 0)
		numValidationSet = 1;

	int numTrainSet = numSamples - numValidationSet;

	Eigen::MatrixXd X_training(numTrainSet, numFeatures);
	Eigen::VectorXd y_training(numTrainSet);

	Eigen::MatrixXd X_validation(numValidationSet, numFeatures);
	Eigen::VectorXd y_validation(numValidationSet);

	for (int i = 0; i < numTrainSet; i++)
	{
		for (int j = 0; j < numFeatures; j++)
		{
			X_training(i, j) = shuffledData(i, j);
		}


		y_training(i) = shuffledData(i, numFeatures);
	}

	for (int i = 0; i < numValidationSet; i++)
	{
		for (int j = 0; j < numFeatures; j++)
		{
			X_validation(i, j) = shuffledData(numTrainSet + i, j);
		}

		y_validation(i) = shuffledData(numTrainSet + i, numFeatures);
	}

	vector<int> hiddenLayerSizes;

	// LAYERS are defined here. More can be added (or reduced) by 
	//		hiddenLayerSizes.push_back(layer_size)

	hiddenLayerSizes.push_back(6);
	hiddenLayerSizes.push_back(6);

	neuralNetwork->Build(numFeatures, hiddenLayerSizes, numClasses);

	neuralNetwork->SetLambda(lambda);
	neuralNetwork->SetEpsilon(epsilon); 

	neuralNetwork->Train(X_training, y_training);

	cout << "Training finished. Calculating score..." << endl;

	cout << "Score: " << neuralNetwork->Score(X_validation, y_validation) << endl;

	if (!modelPath.empty())
		neuralNetwork->SaveAsFile(modelPath);

	return 0;
}