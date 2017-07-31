#include "Net.h"
#include "MathOperations.h"

int main()
{
	Net * neuralNetwork = new Net();

	vector<vector<double>> x;
	vector<double> y;

	x.push_back({ 2.0, 3.0 });
	x.push_back({ 1.0, 0.0 });
	x.push_back({ 1.0, 2.0 });
	x.push_back({ 3.0, 3.0 });
	x.push_back({ 0.0, 0.0 });


	y.push_back(1);
	y.push_back(0);
	y.push_back(1);
	y.push_back(1);
	y.push_back(0);

	int numClasses = 2;
	int numFeatures = 2;

	vector<int> layerSizes;

	layerSizes.push_back(3);
	layerSizes.push_back(4);
	layerSizes.push_back(numClasses);

	neuralNetwork->Build(numFeatures, layerSizes);

	neuralNetwork->Train(x, y);

	/*vector<vector<double>> x_test;
	
	vector<double> row;
	row.push_back(1);
	row.push_back(2);
	row.push_back(3);
	row.push_back(2);
	x_test.push_back(row);

	row.clear();
	row.push_back(0);
	row.push_back(2);
	row.push_back(1);
	row.push_back(2);
	x_test.push_back(row);

	row.clear();
	row.push_back(0);
	row.push_back(0);
	row.push_back(1);
	row.push_back(3);
	x_test.push_back(row);

	vector<vector<double>> x_test_2;
	vector<double> row2;
	row2.push_back(1);
	row2.push_back(2);
	row2.push_back(3);
	
	x_test_2.push_back(row2);

	row2.clear();
	row2.push_back(0);
	row2.push_back(2);
	row2.push_back(1);
	
	x_test_2.push_back(row2);

	row2.clear();
	row2.push_back(0);
	row2.push_back(0);
	row2.push_back(1);
	
	x_test_2.push_back(row2);

	row2.clear();
	row2.push_back(2);
	row2.push_back(2);
	row2.push_back(3);

	x_test_2.push_back(row2);


	vector<double> b_test;

	b_test.push_back(1);
	b_test.push_back(2);
	b_test.push_back(0);
	b_test.push_back(3);

	vector<double> result = MathOperations::MatrixMultiplication(x_test_2, b_test, true);

	int n = 0;

	for (int y = 0; y < x_test_2.size(); y++)
	{
		int numRows = x_test_2.at(y).size();

		for (int x = 0; x < numRows; x++)
		{
			printf("%lf ", x_test_2.at(y).at(x));
		}

		if (n < b_test.size())
		{
			printf("   %lf\n", b_test.at(n));
			n++;
		}

	}

	for (int y = 0; y < result.size(); y++)
	{
		
		printf("%lf\n", result.at(y));
	}

	result = MathOperations::MatrixMultiplication(x_test, b_test, false);
	printf("\n");

	for (int y = 0; y < result.size(); y++)
	{
		printf("%lf\n", result.at(y));
	}*/

	return 0;
}