#pragma once

#include <vector>

using namespace std;

class CostFunction
{
public:

	virtual ~CostFunction() = 0;

	virtual double Compute(double x) = 0;
	virtual double Compute(vector<vector<double>> theta, vector<double> x, vector<double> y) = 0;
};

