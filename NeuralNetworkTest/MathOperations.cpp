#include "MathOperations.h"



MathOperations::MathOperations()
{
}


MathOperations::~MathOperations()
{
}

vector<double> MathOperations::VectorAddition(vector<double> A, vector<double> B)
{
	vector<double> x;

	for (int r = 0; r < A.size(); r++)
	{
		double val = A.at(r) + B.at(r);

		x.push_back(val);
	}
	
	return x;
}

vector<double> MathOperations::VectorSubtraction(vector<double> A, vector<double> B)
{
	vector<double> x;

	for (int r = 0; r < A.size(); r++)
	{
		double val = A.at(r) - B.at(r);

		x.push_back(val);
	}

	return x;
}

vector<double> MathOperations::ElementwiseMultiplication(vector<double> A, vector<double> B)
{
	vector<double> y(A.size(), 0);

	for (int i = 0; i < A.size(); i++)
	{
		y.at(i) = A.at(i) * B.at(i);
	}

	return y;
}

vector<double> MathOperations::MatrixMultiplication(vector<vector<double>> A, vector<double> x, bool transpose)
{
	vector<double> b;

	if (A.size() == 0)
		return b;

	

	if (transpose)
	{

		
		int rows = A.size();
		int cols = A.at(0).size();


		b = vector<double>(cols, 0.0);

		// change for transpose
		for (int c = 0; c < cols; c++)
		{
			for (int r = 0; r < rows; r++)
			{
				double valA = A.at(r).at(c);
				double valX = x.at(r);

				double value = valA * valX;

				b.at(c) += value;
			}

			
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

			b.push_back(value);
		}
	}
	

	return b;
}

vector<vector<double>> MathOperations::VectorMultiplication(vector<double> a, vector<double> b)
{
	vector<vector<double>> M;

	for (int y = 0; y < a.size(); y++)
	{
		vector<double> row(b.size(), 0);

		for (int x = 0; x < b.size(); x++)
		{
			row.at(x) = b.at(x) * a.at(y);
		}

		M.push_back(row);
	}


	return M;
}
