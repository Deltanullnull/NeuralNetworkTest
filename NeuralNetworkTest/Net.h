#pragma once

#include "Layer.h"

class Net
{
public:
	Net();
	~Net();

	void Build(int numInput, vector<int> layerSizes);
	void Train(vector<vector<double>> X, vector<double> y, int numEpochs = 2000);

	Layer* firstHiddenLayer = nullptr;
};

