#pragma once

#include <vector>

using namespace std;

class LossFunction
{
public:
	LossFunction();
	~LossFunction();

	double Compute(vector<double> y, vector<double> x, vector<double> theta);
};

