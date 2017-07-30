#pragma once

#include "CostFunction.h"

class JacobiFunction : 
	public CostFunction
{
public:
	JacobiFunction();
	~JacobiFunction();

	double Compute(double x);
	double Derivative(double x);

	double Compute(vector<vector<double>> theta, vector<double> x, vector<double> y);
};

