#pragma once

#include "Layer.h"

class Net
{
public:
	Net();
	~Net();

	void Build(int numHiddenLayers, vector<int> layerSizes);


	Layer* firstHiddenLayer;
};

