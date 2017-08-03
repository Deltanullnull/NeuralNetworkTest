#include "Net.h"
#include "MathOperations.h"
#include <fstream>

int main(int argc, char ** argv)
{
	//vector<vector<double>> trainingDataVector;
	vector<double> trainingDataVector;
	vector<double> trainingLabels;

	int numClasses = 0;
	int numFeatures = 0;
	int numSamples = 0;

	/*FILE * pipe = _popen("gnuplot", "w");

	if (pipe != nullptr)
	{
		fprintf(pipe, "set term win\n");
		fprintf(pipe, "plot(x, sin(x))\n");
		fprintf(pipe, "set term pngcairo\n");
		fprintf(pipe, "set output \"myFile.png\"\n");
		fprintf(pipe, "replot\n");
		fprintf(pipe, "set term win\n");
		fflush(pipe);
	}
	_pclose(pipe);

	exit(1);*/

	if (argc < 2)
	{
		cout << "No CSV file included. Evaluating example." << endl;


		numFeatures = 2;

		trainingDataVector.push_back(2.0); 
		trainingDataVector.push_back(3.0);
		trainingDataVector.push_back(1.0); 
		trainingDataVector.push_back(0.0);
		trainingDataVector.push_back(1.0);
		trainingDataVector.push_back(2.0);
		trainingDataVector.push_back(3.0);
		trainingDataVector.push_back(3.0);
		trainingDataVector.push_back(0.0);
		trainingDataVector.push_back(0.0);

		trainingLabels.push_back(1);
		trainingLabels.push_back(0);
		trainingLabels.push_back(1);
		trainingLabels.push_back(1);
		trainingLabels.push_back(0);
	}
	else
	{
		vector<string> trainingLabelNames;

		string csvFileName(argv[1]);

		if (csvFileName.find(".csv") == string::npos)
		{
			cout << "File is not a csv file. Please provide a csv file." << endl;
			exit(-1);
		}

		cout << "CSV file " << csvFileName << " loaded." << endl;

		ifstream csvFile(csvFileName);

		if (!csvFile)
		{
			cout << "File could not be opened." << endl;
			exit(-1);
		}

		string line;

		while (getline(csvFile, line))
		{
			string token;

			cout << "Reading line " << line << endl;

			istringstream stream(line);

			int i = 0;

			vector<string> sample;

			while (getline(stream, token, ','))
			{
				
				//double val = stod(token);

				cout << "Value: " << token << endl;

				sample.push_back(token);

				/*if (i == 0)
				{
					trainingLabels.push_back(val);
				}
				else
				{
					trainingDataVector.push_back(val);
				}*/

				i++;

			}

			for (int i = 0; i < sample.size() - 1; i++)
			{
				double val = stod(sample.at(i));

				trainingDataVector.push_back(val);
			}

			trainingLabelNames.push_back(*sample.end());


			if (numFeatures == 0)
				numFeatures = trainingDataVector.size();

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

	Eigen::Map<Eigen::VectorXd> y(trainingLabels.data(), trainingLabels.size());

	cout << X << endl;

	cout << y << endl;

	vector<int> hiddenLayerSizes;

	hiddenLayerSizes.push_back(5);
	hiddenLayerSizes.push_back(5);

	neuralNetwork->Build(numFeatures, hiddenLayerSizes, numClasses);

	neuralNetwork->Train(X, y);

	for (int i = 0; i < X.rows(); i++)
	{
		cout << "Result for " << X.row(i) << ":" << endl;
		cout << neuralNetwork->Evaluate(X.row(i)) << endl;
	}
	return 0;
}