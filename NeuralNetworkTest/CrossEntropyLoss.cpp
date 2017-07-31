#include "CrossEntropyLoss.h"
#include <math.h>


/*CrossEntropyLoss::CrossEntropyLoss()
{
}


CrossEntropyLoss::~CrossEntropyLoss()
{
}*/

double CrossEntropyLoss::Compute(double yPred, double yTrue)
{
	double val = yTrue * log(yPred);

	return val;
}

double CrossEntropyLoss::Derivative(double x, double y)
{
	return 0.0;
}

double CrossEntropyLoss::ComputeVectorized(vector<vector<double>> theta, vector<vector<double>> yPred, vector<vector<double>> yTrue)
{
	double loss = 0.0;

	int numTrainingData = yPred.size();
	int numClasses = yPred.at(0).size();

	for (int t = 0; t < numTrainingData; t++)
	{
		for (int c = 0; c < numClasses; c++)
		{
			loss += Compute(yPred.at(t).at(c), yTrue.at(t).at(c));
		}
	}

	return loss / numClasses;
}
