#include "Sigmoid.h"



Sigmoid::Sigmoid()
{
}


Sigmoid::~Sigmoid()
{
}

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
