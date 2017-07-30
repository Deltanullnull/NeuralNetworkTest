#pragma once

#include <vector>

using namespace std;

class MathOperations
{
public:
	MathOperations();
	~MathOperations();

	static vector<double> MatrixMultiplication(vector<vector<double>> A, vector<double> x, bool transpose);
};

