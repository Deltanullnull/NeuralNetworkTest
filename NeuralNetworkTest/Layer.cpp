#include "Layer.h"
#include <random>
#include "MathOperations.h"

Layer::Layer(bool isHiddenLayer)
{
	Func = new Sigmoid();
	this->isHiddenLayer = isHiddenLayer;
}


Layer::~Layer()
{
}

void Layer::Init(int numInput, int numOutput)
{
	if (numOutput > 0)
	{
		//WeightMatrix = Eigen::ArrayXXd::Random(numOutput, numInput) * 0.5 + 0.5;
		WeightMatrix = Eigen::ArrayXXd::Random(numOutput, numInput + 1) * 0.5 + 0.5;

		cout << "WeightMatrix: " << endl;
		cout << WeightMatrix << endl;

		//BiasVector = Eigen::ArrayXd::Random(numOutput) * 0.5 + 0.5;


		/*cout << "BiasVector: " << endl;
		cout << BiasVector << endl;*/

		DeltaMatrix = Eigen::MatrixXd::Zero(numOutput, numInput + 1);
		//DeltaMatrix = Eigen::MatrixXd::Zero(numOutput, numInput);



		PartialDerivatives = Eigen::MatrixXd::Zero(numOutput, numInput + 1);
		//PartialDerivatives = Eigen::MatrixXd::Zero(numOutput, numInput);


		//VectorZ = Eigen::VectorXd(numOutput);
	}
}

void Layer::ForwardPropagate(Eigen::VectorXd input, double y)
{
	//cout << "Going to next layer" << endl;

	if (previousLayer == nullptr)
	{
		if (input.rows() + 1 != WeightMatrix.cols())
		{
			cerr << "Error! Data input size does not match layer input size. Abort!" << endl;
			exit(-1);
		}

		Eigen::VectorXd InputWithBias(input.rows() + 1);

		InputWithBias[0] = +1; // Set constant to 1

		for (int i = 0; i < input.rows(); i++)
		{
			InputWithBias[i + 1] = input[i];
		}

		ActivationVector = InputWithBias;


	}
	else
	{
		
		
		// z = W * x + b
		VectorZ = previousLayer->WeightMatrix * input;
		//VectorZ = previousLayer->WeightMatrix * input + previousLayer->BiasVector;

		// a = g(z)

		if (connectedLayer != nullptr)
		{
			Eigen::VectorXd TempActivation = Func->Compute(VectorZ);


			ActivationVector = Eigen::VectorXd(TempActivation.rows() + 1);


			ActivationVector[0] = +1;


			for (int i = 0; i < TempActivation.rows(); i++)
			{
				ActivationVector[i + 1] = TempActivation[i];
			}
		}
		else
		{
			ActivationVector = Func->Compute(VectorZ);
		}

	}

	

	if (connectedLayer != nullptr)
	{
		connectedLayer->ForwardPropagate(ActivationVector, y);
	}
	else
	{


		Eigen::VectorXd VectorY = Eigen::VectorXd::Zero(ActivationVector.rows());

		VectorY((int)y) = 1.0;

		Eigen::VectorXd deltaVector = ActivationVector - VectorY;

		previousLayer->BackwardPropagate(deltaVector);
	}
	
}

void Layer::BackwardPropagate(Eigen::VectorXd deltas)
{
	Eigen::MatrixXd TempMatrix = deltas * ActivationVector.transpose();

	DeltaMatrix = DeltaMatrix + TempMatrix;

	PartialDerivatives = (DeltaMatrix + regLambda * WeightMatrix);

	if (previousLayer != nullptr)
	{
		Eigen::VectorXd DeltasCurrent = WeightMatrix.transpose() * deltas;

		//Eigen::VectorXd gDeriv = Func->Derivative(VectorZ);
		Eigen::VectorXd gDeriv = ActivationVector.cwiseProduct(Eigen::VectorXd::Ones(ActivationVector.rows()) - ActivationVector);

		DeltasCurrent = DeltasCurrent.cwiseProduct(gDeriv);

		// TODO remove first value
		Eigen::VectorXd DeltasTemp(DeltasCurrent.rows() - 1);

		for (int i = 0; i < DeltasTemp.rows(); i++)
		{
			DeltasTemp[i] = DeltasCurrent[i + 1];
		}

		UpdatePartialDerivative();

		previousLayer->BackwardPropagate(DeltasTemp);
		//previousLayer->BackwardPropagate(DeltasCurrent);
	}
}

void Layer::SetDeltaToZero()
{
	DeltaMatrix.setZero();

	if (connectedLayer != nullptr)
	{
		connectedLayer->SetDeltaToZero();
	}
}

void Layer::UpdateWeights()
{
	WeightMatrix = WeightMatrix - epsilon * PartialDerivatives;

	if (connectedLayer != nullptr)
		connectedLayer->UpdateWeights();
}

void Layer::UpdatePartialDerivative()
{
	if (numSamples == 0)
	{
		cout << "Error! Num Samples <= 0!" << endl;
	}

	for (int i = 0; i < PartialDerivatives.rows(); i++)
	{
		for (int j = 0; j < PartialDerivatives.cols(); j++)
		{
			if (j == 0)
			{
				PartialDerivatives(i, j) = DeltaMatrix(i, j) / numSamples;
			}
			else
			{
				PartialDerivatives(i, j) = (DeltaMatrix(i, j) + regLambda * WeightMatrix(i, j))/ numSamples;
			}
		}
	}
}

Eigen::VectorXd Layer::Evaluate(Eigen::VectorXd input)
{
	Eigen::VectorXd Output;

	if (previousLayer == nullptr)
	{
		if (input.rows() + 1 != WeightMatrix.cols())
		{
			cerr << "Error! Data input size does not match layer input size. Abort!" << endl;
			exit(-1);
		}

		Eigen::VectorXd InputWithBias(input.rows() + 1);


		InputWithBias[0] = +1; // Set constant to 1

		for (int i = 0; i < input.rows(); i++)
		{
			InputWithBias[i + 1] = input[i];
		}

		Output = InputWithBias;

		if (connectedLayer != nullptr)
		{
			return connectedLayer->Evaluate(Output);
		}
	}
	else
	{


		// z = W * x + b
		Eigen::VectorXd Z = previousLayer->WeightMatrix * input;
		
		// a = g(z)

		if (connectedLayer != nullptr)
		{
			Eigen::VectorXd TempActivation = Func->Compute(Z);

			Output = Eigen::VectorXd(TempActivation.rows() + 1);

			Output[0] = +1;

			for (int i = 0; i < TempActivation.rows(); i++)
			{
				Output[i + 1] = TempActivation[i];
			}

			return connectedLayer->Evaluate(Output);
		}
		else
		{
			return Func->Compute(Z);
		}

	}

	return Output;
}

double Layer::GetWeightSum()
{
	double val = 0.0;

	for (int i = 0; i < WeightMatrix.rows(); i++)
	{
		for (int j = 0; j < WeightMatrix.cols(); j++)
		{
			val += pow(WeightMatrix(i, j), 2);
		}
	}

	if (connectedLayer != nullptr)
	{
		

		val += connectedLayer->GetWeightSum();
	}

	return val;
}


double Layer::ComputeLoss(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::MatrixXd yPred)
{
	double value = 0.0;

	for (int v = 0; v < WeightMatrix.rows(); v++)
	{
		for (int u = 0; u < WeightMatrix.cols(); u++)
		{
			double theta = WeightMatrix(v, u);

			value += (regLambda / 2.0) * pow(theta, 2.0);
		}
	}

	Eigen::MatrixXd newPred(yPred.rows(), yPred.cols());

	for (int i = 0; i < yPred.rows(); i++)
	{
		// a = g(z)
		newPred.row(i) = Func->Compute(WeightMatrix * yPred.row(i));
		//newPred.row(i) = Func->Compute(WeightMatrix * yPred.row(i) + BiasVector);

	}

	if (connectedLayer != nullptr)
	{
		value += connectedLayer->ComputeLoss(X, y, newPred);
	}
	else
	{
		for (int v = 0; v < newPred.size(); v++)
		{
			Eigen::VectorXd yPredVector = newPred.row(v);

			int numClasses = newPred.rows();

			Eigen::VectorXd YTrueVector = Eigen::VectorXd::Zero(numClasses);

			YTrueVector((int)v) = 1.0;

			for (int u = 0; u < numClasses; u++)
			{
				value += YTrueVector[u] * log(yPredVector[u]) + (1.0 - YTrueVector[u]) * log(1.0 - yPredVector[u]);
			}
		}
	}

	return value;
}


void Layer::SetNumSamples(int numSamples)
{
	this->numSamples = numSamples;
	if (connectedLayer != nullptr)
	{
		connectedLayer->SetNumSamples(numSamples);
	}
}
