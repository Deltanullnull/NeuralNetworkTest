#pragma once

#include <vector>
#include <math.h>

#include <Eigen/Dense>

using namespace std;

class ActivationFunction
{
public:
	

	virtual double Compute(double x) = 0;
	virtual double Derivative(double x) = 0;

	virtual vector<double> Compute(vector<double> x_vector) = 0;
	virtual vector<double> Derivative(vector<double> x_vector) = 0;

	virtual Eigen::VectorXd Compute(Eigen::VectorXd X) = 0;
	virtual Eigen::VectorXd Derivative(Eigen::VectorXd X) = 0;
};

