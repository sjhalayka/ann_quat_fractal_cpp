#include "ffbpneuralnet.h"

#include <iostream>
#include <ctime>
#include <random>
using namespace std;


const unsigned int num_components = 4;

vector<double> qmul(const vector<double>& qaqb)
{
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

vector<double> qmul_ann(const vector<double>& qaqb, FFBPNeuralNet &NNet2)
{
	NNet2.FeedForward(qaqb);

	vector<double> predicted;
	NNet2.GetOutputValues(predicted);

	return predicted;
}


int main(void)
{
	std::mt19937 generator_real(static_cast<unsigned>(time(0)));
	std::uniform_real_distribution<float> dis_real(-1.0f, 1.0f);

	const double threshold = 4.0;


	// Train network
	//
	//vector<size_t> HiddenLayers;
	//HiddenLayers.push_back(8 * num_components);
	//HiddenLayers.push_back(16 * num_components);
	//HiddenLayers.push_back(32 * num_components);

	//FFBPNeuralNet NNet(2 * num_components, HiddenLayers, num_components);

	//NNet.SetLearningRate(0.0001);
	//NNet.SetMomentum(1.0);

	//const double max_error_rate = 0.01;
	//const long unsigned int max_training_sessions = 1000000;

	//double error_rate = 0.0;
	//long unsigned int num_training_sessions = 0;

	//do
	//{
	//	if (num_training_sessions % 1000 == 0)
	//		cout << num_training_sessions / static_cast<float>(max_training_sessions) << endl;

	//	vector<double> io;

	//	for (size_t i = 0; i < 2 * num_components; i++)
	//		io.push_back(threshold * dis_real(generator_real));

	//	NNet.FeedForward(io);
	//	io = qmul(io);
	//	error_rate = NNet.BackPropagate(io);
	//	error_rate = sqrt(error_rate);

	//	num_training_sessions++;

	//} while (error_rate >= max_error_rate && num_training_sessions < max_training_sessions);

	//cout << "Final number of training sessions/epochs: " << num_training_sessions << endl;
	//cout << "Final error rate: " << error_rate << endl;
	//cout << endl;

	//NNet.SaveToFile("network.bin");






	// Load a network from a file
	//
	FFBPNeuralNet NNet2("network.bin");

	vector<double> io;

	for (size_t i = 0; i < 2 * num_components; i++)
		io.push_back(threshold * dis_real(generator_real));

	vector<double> predicted = qmul_ann(io, NNet2);
	vector<double> expected = qmul(io);

	cout << "Predicted: "
		<< predicted[0] << " " << predicted[1] << " "
		<< predicted[2] << " " << predicted[3] << endl;

	cout << "Expected:  "
		<< expected[0] << " " << expected[1] << " "
		<< expected[2] << " " << expected[3] << endl;

	return 0;
}