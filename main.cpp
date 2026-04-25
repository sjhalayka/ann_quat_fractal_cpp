#include "ffbpneuralnet.h"

#include <iostream>
#include <ctime>
#include <random>
using namespace std;


const unsigned int num_components = 4;


vector<double> qmul(const vector<double>& qaqb)
{
	// in case qA and qOut point to the same variable...
	double temp_a_x = qaqb[0];
	double temp_a_y = qaqb[1];
	double temp_a_z = qaqb[2];
	double temp_a_w = qaqb[3];

	double temp_b_x = qaqb[4];
	double temp_b_y = qaqb[5];
	double temp_b_z = qaqb[6];
	double temp_b_w = qaqb[7];

	vector<double> out;

	out.push_back(temp_a_x * temp_b_x - temp_a_y * temp_b_y - temp_a_z * temp_b_z - temp_a_w * temp_b_w);
	out.push_back(temp_a_x * temp_b_y + temp_a_y * temp_b_x + temp_a_z * temp_b_w - temp_a_w * temp_b_z);
	out.push_back(temp_a_x * temp_b_z - temp_a_y * temp_b_w + temp_a_z * temp_b_x + temp_a_w * temp_b_y);
	out.push_back(temp_a_x * temp_b_w + temp_a_y * temp_b_z - temp_a_z * temp_b_y + temp_a_w * temp_b_x);

	return out;
}






int main(void)
{
	std::mt19937 generator_real(static_cast<unsigned>(time(0)));
	std::uniform_real_distribution<float> dis_real(-1.0f, 1.0f);

	vector<size_t> HiddenLayers;
	HiddenLayers.push_back(32 * num_components);
	HiddenLayers.push_back(16 * num_components);
	HiddenLayers.push_back(32 * num_components);

	FFBPNeuralNet NNet(2 * num_components, HiddenLayers, num_components);

	NNet.SetLearningRate(0.01);
	NNet.SetMomentum(0.01);

	const double max_error_rate = 0.01;
	const long unsigned int max_training_sessions = 10000;

	double error_rate = 0.0;
	long unsigned int num_training_sessions = 0;

	const double threshold = 4.0;

	// train network until the error rate goes below the maximum error rate
	// or we reach the maximum number of training sessions (which could be considered as "giving up")
	do
	{
		vector<double> io;

		for (size_t i = 0; i < 2 * num_components; i++)
			io.push_back(threshold * dis_real(generator_real));

		NNet.FeedForward(io);
		io = qmul(io);
		error_rate = NNet.BackPropagate(io);
		error_rate = sqrt(error_rate);

		num_training_sessions++;

	} while (error_rate >= max_error_rate && num_training_sessions < max_training_sessions);


	// print out how many training sessions it took to arrive at whatever the final error rate was
	cout << "Final number of training sessions/epochs: " << num_training_sessions << endl;
	cout << "Final error rate: " << error_rate << endl << endl;
	cout << endl;


	// save the network to a file
	NNet.SaveToFile("network.bin");

	// load a network from a file
	FFBPNeuralNet NNet2("network.bin");

	// now use the pre-trained network
	vector<double> io;

	for (size_t i = 0; i < 2 * num_components; i++)
		io.push_back(threshold * dis_real(generator_real));

	NNet2.FeedForward(io);

	vector<double> io2;
	NNet2.GetOutputValues(io2);
	cout << io2[0] << " " << io2[1] << " " << io2[2] << " " << io2[3] << endl;

	vector<double> io3 = qmul(io);
	cout << io3[0] << " " << io3[1] << " " << io3[2] << " " << io3[3] << endl;

	return 0;
}