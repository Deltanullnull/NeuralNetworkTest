#pragma once
#include "ActivationFunctions.h"
class Sigmoid :
	public ActivationFunction
{
public:
	//Sigmoid();
	//~Sigmoid();

	double Compute(double x);
	double Derivative(double x);

	vector<double> Compute(vector<double> x_vector);
	vector<double> Derivative(vector<double> x_vector);

	Eigen::VectorXd Compute(Eigen::VectorXd X);
	Eigen::VectorXd Derivative(Eigen::VectorXd X);
};

