#pragma once
#include "CostFunction.h"
class CrossEntropyLoss :
	public CostFunction
{
public:
	//CrossEntropyLoss();
	//~CrossEntropyLoss();

	double Compute(double x, double y);
	double Derivative(double x, double y);

	double ComputeVectorized(vector<vector<double>> theta, vector<vector<double>> yPred, vector<vector<double>> yTrue);
};

