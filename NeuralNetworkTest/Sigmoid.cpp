#include "Sigmoid.h"



/*Sigmoid::Sigmoid()
{
}


Sigmoid::~Sigmoid()
{
}*/

double Sigmoid::Compute(double x)
{
	double value = 1 / (1 + exp(-x));

	return value;
}

double Sigmoid::Derivative(double x)
{
	double value = Compute(x) * (1.0 - Compute(x));

	return value;
}

vector<double> Sigmoid::Compute(vector<double> x_vector)
{
	vector<double> g(x_vector.size(), 0);

	for (int i = 0; i < g.size(); i++)
	{
		g.at(i) = Compute(x_vector.at(i));
	}

	return g;
}

vector<double> Sigmoid::Derivative(vector<double> x_vector)
{
	vector<double> g(x_vector.size(), 0);

	for (int i = 0; i < g.size(); i++)
	{
		g.at(i) = Derivative(x_vector.at(i));
	}

	return g;
}
