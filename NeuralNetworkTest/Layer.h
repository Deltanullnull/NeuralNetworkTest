#pragma once

#include "Sigmoid.h"

#include <iostream>

#include <Eigen/Dense>

using namespace std;

class Layer
{
public:
	Layer();
	~Layer();
	
	/// <summary>
	/// Initializes the weight matrix.
	/// </summary>
	/// <param name="numInput">Input size</param>
	/// <param name="numOutput">Output size</param>
	void Init(int numInput, int numOutput);
	
	/// <summary>
	/// Stores all weights including input and output size in a vector.
	/// </summary>
	/// <returns>Layer info</returns>
	vector<double> GetLayerInfo();
	
	/// <summary>
	/// Fills weight buffer with values read from file.
	/// </summary>
	/// <param name="weightBuffer">Weight buffer.</param>
	/// <param name="currentIdx">Current index inside buffer.</param>
	void FillWeights(vector<double> weightBuffer, int start);
	
	/// <summary>
	/// Forward propagation.
	/// </summary>
	/// <param name="input">Input vector</param>
	/// <param name="y">Label</param>
	void ForwardPropagate(Eigen::VectorXd input, double y);
	
	/// <summary>
	/// Backward propagation
	/// </summary>
	/// <param name="deltas">Delta vector</param>
	void BackwardPropagate(Eigen::VectorXd deltas);
	
	/// <summary>
	/// Sets all components of delta vector to zero.
	/// </summary>
	void SetDeltaToZero();
	
	/// <summary>
	/// Updates the weights.
	/// </summary>
	void UpdateWeights();
	
	/// <summary>
	/// Updates the partial derivatives.
	/// </summary>
	void UpdatePartialDerivative();
	
	/// <summary>
	/// Evaluates the sample.
	/// </summary>
	/// <param name="X">Data sample.</param>
	/// <returns></returns>
	Eigen::VectorXd Evaluate(Eigen::VectorXd X);
	
	/// <summary>
	/// Gets sum of all weights on this layer and it's connected layer.
	/// </summary>
	/// <returns>Sum of weights</returns>
	double GetWeightSum();
	
	/// <summary>
	/// Computes loss.
	/// </summary>
	/// <param name="X">Training data set.</param>
	/// <param name="y">Training label.</param>
	/// <param name="yPred">Input vector for this layer.</param>
	/// <returns></returns>
	double ComputeLoss(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::MatrixXd yPred);
	
	/// <summary>
	/// Sets the number of samples.
	/// </summary>
	/// <param name="numSamples">The number samples.</param>
	void SetNumSamples(int numSamples);
	
	/// <summary>
	/// Sets lambda value.
	/// </summary>
	/// <param name="lambda">Lambda value.</param>
	void SetLambda(double lambda);
	
	/// <summary>
	/// Sets epsilon value.
	/// </summary>
	/// <param name="epsilon">Epsilon value.</param>
	void SetEpsilon(double epsilon);

	
	Eigen::MatrixXd WeightMatrix;

	Eigen::VectorXd VectorZ;
	Eigen::VectorXd ActivationVector;

	Eigen::MatrixXd DeltaMatrix;
	Eigen::MatrixXd PartialDerivatives;

	Layer* connectedLayer = nullptr;
	Layer * previousLayer = nullptr;

	ActivationFunction * Func = nullptr;

	double regLambda = 0.01; // Learning rate
	double epsilon = 0.01; // Regularization strength

	bool isHiddenLayer = true;

	int numSamples;
};

