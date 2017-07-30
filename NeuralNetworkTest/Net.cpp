#include "Net.h"



Net::Net()
{
}


Net::~Net()
{
}

void Net::Build(int numInput, vector<int> layerSizes)
{
	Layer * current;

	int numHiddenLayers = layerSizes.size();

	int numOutput = layerSizes.at(0);

	firstHiddenLayer = new Layer();
	firstHiddenLayer->Init(numInput, numOutput);

	current = firstHiddenLayer;

	for (int i = 0; i < numHiddenLayers - 1; i++)
	{
		Layer * connected = new Layer();
		
		current->connectedLayer = connected;
		connected->previousLayer = current;

		current = connected;
	}
}
