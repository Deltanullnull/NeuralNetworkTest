#pragma once

#include <vector>
#include <math.h>

using namespace std;

class ActivationFunction
{
public:
	
	virtual ~ActivationFunction() = 0;

	virtual double Compute(double x) = 0;
	virtual double Derivative(double x) = 0;

	//virtual vector<double> Compute(vector<double> x_vector) = 0;
};

