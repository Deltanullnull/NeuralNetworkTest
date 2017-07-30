#include "MathOperations.h"



MathOperations::MathOperations()
{
}


MathOperations::~MathOperations()
{
}

vector<double> MathOperations::MatrixMultiplication(vector<vector<double>> A, vector<double> x, bool transpose)
{
	vector<double> b;

	if (transpose)
	{
		// TODO change for transpose
		for (int r = 0; r < A.size(); r++)
		{
			int cols = A.at(r).size();

			double value = 0;

			for (int c = 0; c < cols; c++)
			{
				value += A.at(r).at(c) * x.at(c);
			}

			b.push_back(r);
		}
	}
	else
	{
		for (int r = 0; r < A.size(); r++)
		{
			int cols = A.at(r).size();

			double value = 0;

			for (int c = 0; c < cols; c++)
			{
				value += A.at(r).at(c) * x.at(c);
			}

			b.push_back(r);
		}
	}
	

	return b;
}
