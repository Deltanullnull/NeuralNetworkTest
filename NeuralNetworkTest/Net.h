#pragma once

#include "Layer.h"

class Net
{
public:
	Net();
	~Net();

	void Build(int numHiddenLayers, int numInput);

	vector<Layer*> firstHiddenLayers;
};

