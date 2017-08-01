#pragma once

#include <vector>


using namespace std;

class MathOperations
{
public:
	MathOperations();
	~MathOperations();

	static vector<double> VectorAddition(vector<double> A, vector<double> B);
	static vector<double> VectorSubtraction(vector<double> A, vector<double> B);

	static vector<double> ElementwiseMultiplication(vector<double> A, vector<double> B);

	static vector<vector<double>> MatrixAddition(vector<vector<double>> A, vector<vector<double>> B);

	static vector<double> MatrixMultiplication(vector<vector<double>> A, vector<double> x, bool transpose);

	static vector<vector<double>> VectorMultiplication(vector<double> a, vector<double> b);
};

