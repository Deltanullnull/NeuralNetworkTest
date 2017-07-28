#pragma once
#include "ActivationFunctions.h"
class Sigmoid :
	public ActivationFunction
{
public:
	Sigmoid();
	~Sigmoid();

	double Compute(double x);
	double Derivative(double x);
};

