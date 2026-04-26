#include "ffbpneuralnet.h"
#include "weighted_neuron.h"

#include <sstream>
using std::ostringstream;

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <ios>
using std::ios;

#include <stdexcept>
using std::out_of_range;
using std::runtime_error;

#include <iostream>
using std::cout;
using std::endl;

#include <iomanip>

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>


FFBPNeuralNet::FFBPNeuralNet(const size_t& src_num_input_neurons, const vector<size_t>& src_num_hidden_layers_neurons, const size_t& src_num_output_neurons)
{
	// sanity checks
	if (src_num_input_neurons == 0)
		throw out_of_range("Invalid number of input neurons.");

	if (src_num_hidden_layers_neurons.size() == 0)
		throw out_of_range("Invalid number of hidden layers.");

	for (size_t i = 0; i < src_num_hidden_layers_neurons.size(); i++)
	{
		if (src_num_hidden_layers_neurons[i] == 0)
		{
			ostringstream out;
			out << "Invalid number of neurons in hidden layer #" << static_cast<long unsigned int>(i + 1) << ".";
			throw out_of_range(out.str().c_str());
		}
	}

	if (src_num_output_neurons == 0)
		throw out_of_range("Invalid number of output neurons.");


	// create input "neurons"
	InputLayer.resize(src_num_input_neurons, 0.0);


	// create hidden layers
	HiddenLayers.resize(src_num_hidden_layers_neurons.size());


	// init first hidden layer
	for (size_t i = 0; i < src_num_hidden_layers_neurons[0]; i++)
		HiddenLayers[0].push_back(WeightedNeuron(InputLayer.size()));

	// init subsequent hidden layers
	for (size_t i = 1; i < src_num_hidden_layers_neurons.size(); i++)
		for (size_t j = 0; j < src_num_hidden_layers_neurons[i]; j++)
			HiddenLayers[i].push_back(WeightedNeuron(HiddenLayers[i - 1].size()));


	// init output layer
	for (size_t i = 0; i < src_num_output_neurons; i++)
		OutputLayer.push_back(WeightedNeuron(HiddenLayers[HiddenLayers.size() - 1].size()));

	learning_rate = 1.0;
	momentum = 1.0;
}

FFBPNeuralNet::FFBPNeuralNet(const char* const src_filename)
{
	LoadFromFile(src_filename);
}

void FFBPNeuralNet::FeedForward(const vector<double>& src_inputs)
{
	// sanity check
	if (src_inputs.size() != InputLayer.size())
		throw out_of_range("Invalid input vector size.");

	InputLayer = src_inputs;

	// feed input values to first hidden layer's neurons
	for (size_t i = 0; i < HiddenLayers[0].size(); i++)
		HiddenLayers[0][i].SetInputValues(InputLayer);

	// for each subsequent hidden layer...
	for (size_t i = 1; i < HiddenLayers.size(); i++)
	{
		vector<double> PreviousLayerValues;

		// gather up previous hidden layer's values
		for (size_t j = 0; j < HiddenLayers[i - 1].size(); j++)
			PreviousLayerValues.push_back(HiddenLayers[i - 1][j].GetValue());

		// feed previous hidden layer's values to this hidden layer's neurons
		for (size_t j = 0; j < HiddenLayers[i].size(); j++)
			HiddenLayers[i][j].SetInputValues(PreviousLayerValues);
	}

	// feed final hidden layer's values to output layer's neurons
	vector<double> PreviousLayerValues;

	// gather up final hidden layer's values
	for (size_t i = 0; i < HiddenLayers[HiddenLayers.size() - 1].size(); i++)
		PreviousLayerValues.push_back(HiddenLayers[HiddenLayers.size() - 1][i].GetValue());

	// apply_activation=false gives linear (identity) output for regression
	for (size_t i = 0; i < OutputLayer.size(); i++)
		OutputLayer[i].SetInputValues(PreviousLayerValues, false);
}

void FFBPNeuralNet::GetOutputValues(vector<double>& src_outputs)
{
	src_outputs.clear();
	src_outputs.resize(OutputLayer.size());

	for (size_t i = 0; i < OutputLayer.size(); i++)
		src_outputs[i] = OutputLayer[i].GetValue();
}

size_t FFBPNeuralNet::GetMaximumOutputNeuron(void) const
{
	double temp_val = -DBL_MAX;
	size_t final_index = 0;

	for (size_t i = 0; i < OutputLayer.size(); i++)
	{
		if (OutputLayer[i].GetValue() > temp_val)
		{
			temp_val = OutputLayer[i].GetValue();
			final_index = i;
		}
	}

	return final_index;
}

double FFBPNeuralNet::BackPropagate(const vector<double>& src_desired_outputs)
{
	// generate output layer deltas
	// output neurons are linear (no activation), so derivative is 1 — delta = error directly
	vector<double> OutputLayerErrors(OutputLayer.size());

	for (size_t i = 0; i < OutputLayer.size(); i++)
		OutputLayerErrors[i] = src_desired_outputs[i] - OutputLayer[i].GetValue();


	// calculate mean squared error
	double error_rate = 0.0;

	for (size_t i = 0; i < OutputLayer.size(); i++)
		error_rate += (OutputLayer[i].GetValue() - src_desired_outputs[i]) * (OutputLayer[i].GetValue() - src_desired_outputs[i]);

	error_rate /= static_cast<double>(OutputLayer.size());


	// generate hidden layer errors (backpropagation)
	// for each node: error = (sum of next-layer errors * connecting weights) * activation derivative
	vector< vector<double> > HiddenLayerErrors;
	HiddenLayerErrors.resize(HiddenLayers.size());

	for (size_t i = 0; i < HiddenLayerErrors.size(); i++)
		HiddenLayerErrors[i].resize(HiddenLayers[i].size());

	// generate last hidden layer's errors from output layer
	for (size_t i = 0; i < HiddenLayers[HiddenLayers.size() - 1].size(); i++)
	{
		double sum = 0.0;

		for (size_t j = 0; j < OutputLayer.size(); j++)
			sum += OutputLayerErrors[j] * OutputLayer[j].GetWeight(i);

		HiddenLayerErrors[HiddenLayers.size() - 1][i] = sum * WeightedNeuron::DerivativeOfActivationFunction(HiddenLayers[HiddenLayers.size() - 1][i].GetValue());
	}

	// work backwards through remaining hidden layers
	for (size_t i = 1; i < HiddenLayers.size(); i++)
	{
		size_t ErrorLayerIndex = HiddenLayers.size() - i - 1;
		size_t NextLayerIndex = HiddenLayers.size() - i;

		for (size_t k = 0; k < HiddenLayers[ErrorLayerIndex].size(); k++)
		{
			double sum = 0.0;

			for (size_t j = 0; j < HiddenLayers[NextLayerIndex].size(); j++)
				sum += HiddenLayerErrors[NextLayerIndex][j] * HiddenLayers[NextLayerIndex][j].GetWeight(k);

			HiddenLayerErrors[ErrorLayerIndex][k] = sum * WeightedNeuron::DerivativeOfActivationFunction(HiddenLayers[ErrorLayerIndex][k].GetValue());
		}
	}


	// ---- adjust weights ----

	// adjust output layer weights (inputs come from last hidden layer)
	for (size_t i = 0; i < HiddenLayers[HiddenLayers.size() - 1].size(); i++)
	{
		double neuron_value = HiddenLayers[HiddenLayers.size() - 1][i].GetValue();

		for (size_t j = 0; j < OutputLayer.size(); j++)
		{
			double delta_weight = learning_rate * OutputLayerErrors[j] * neuron_value;
			OutputLayer[j].SetWeight(i, OutputLayer[j].GetWeight(i) + delta_weight + momentum * OutputLayer[j].GetPreviousWeightAdjustment(i));
			OutputLayer[j].SetPreviousWeightAdjustment(i, delta_weight);
		}
	}

	// adjust output layer biases (with momentum)
	// FIX: was missing momentum term; now tracks and applies previous bias adjustment
	for (size_t j = 0; j < OutputLayer.size(); j++)
	{
		double delta_bias = learning_rate * OutputLayerErrors[j] * OutputLayer[j].GetBias();
		double new_bias_weight = OutputLayer[j].GetBiasWeight()
			+ delta_bias
			+ momentum * OutputLayer[j].GetPreviousBiasWeightAdjustment();
		OutputLayer[j].SetBiasWeight(new_bias_weight);
		OutputLayer[j].SetPreviousBiasWeightAdjustment(delta_bias);
	}

	// adjust hidden layers (all except the first), working backwards
	for (size_t i = 1; i < HiddenLayers.size(); i++)
	{
		size_t AdjustmentLayerIndex = HiddenLayers.size() - i;
		size_t PreviousLayerIndex = HiddenLayers.size() - i - 1;

		for (size_t k = 0; k < HiddenLayers[PreviousLayerIndex].size(); k++)
		{
			double neuron_value = HiddenLayers[PreviousLayerIndex][k].GetValue();

			for (size_t j = 0; j < HiddenLayers[AdjustmentLayerIndex].size(); j++)
			{
				double delta_weight = learning_rate * HiddenLayerErrors[AdjustmentLayerIndex][j] * neuron_value;
				HiddenLayers[AdjustmentLayerIndex][j].SetWeight(k, HiddenLayers[AdjustmentLayerIndex][j].GetWeight(k) + delta_weight + momentum * HiddenLayers[AdjustmentLayerIndex][j].GetPreviousWeightAdjustment(k));
				HiddenLayers[AdjustmentLayerIndex][j].SetPreviousWeightAdjustment(k, delta_weight);
			}
		}

		// adjust biases for this hidden layer (with momentum)
		// FIX: was missing momentum term
		for (size_t j = 0; j < HiddenLayers[AdjustmentLayerIndex].size(); j++)
		{
			double delta_bias = learning_rate * HiddenLayerErrors[AdjustmentLayerIndex][j] * HiddenLayers[AdjustmentLayerIndex][j].GetBias();
			double new_bias_weight = HiddenLayers[AdjustmentLayerIndex][j].GetBiasWeight()
				+ delta_bias
				+ momentum * HiddenLayers[AdjustmentLayerIndex][j].GetPreviousBiasWeightAdjustment();
			HiddenLayers[AdjustmentLayerIndex][j].SetBiasWeight(new_bias_weight);
			HiddenLayers[AdjustmentLayerIndex][j].SetPreviousBiasWeightAdjustment(delta_bias);
		}
	}


	// adjust first hidden layer weights (inputs come from input layer)
	for (size_t i = 0; i < InputLayer.size(); i++)
	{
		double neuron_value = InputLayer[i];

		for (size_t j = 0; j < HiddenLayers[0].size(); j++)
		{
			double delta_weight = learning_rate * HiddenLayerErrors[0][j] * neuron_value;
			HiddenLayers[0][j].SetWeight(i, HiddenLayers[0][j].GetWeight(i) + delta_weight + momentum * HiddenLayers[0][j].GetPreviousWeightAdjustment(i));
			HiddenLayers[0][j].SetPreviousWeightAdjustment(i, delta_weight);
		}
	}

	// adjust first hidden layer biases (with momentum)
	// FIX: was missing momentum term
	for (size_t j = 0; j < HiddenLayers[0].size(); j++)
	{
		double delta_bias = learning_rate * HiddenLayerErrors[0][j] * HiddenLayers[0][j].GetBias();
		double new_bias_weight = HiddenLayers[0][j].GetBiasWeight()
			+ delta_bias
			+ momentum * HiddenLayers[0][j].GetPreviousBiasWeightAdjustment();
		HiddenLayers[0][j].SetBiasWeight(new_bias_weight);
		HiddenLayers[0][j].SetPreviousBiasWeightAdjustment(delta_bias);
	}

	return error_rate;
}

size_t FFBPNeuralNet::GetNumInputLayerNeurons(void) const
{
	return InputLayer.size();
}

void FFBPNeuralNet::ResetNumInputLayerNeurons(const size_t& src_num_input_neurons)
{
	if (src_num_input_neurons == 0)
		throw out_of_range("Invalid number of input neurons.");

	InputLayer.resize(src_num_input_neurons, 0.0);

	// in case we are calling this from LoadFromFile via the second constructor
	if (0 != HiddenLayers.size())
		ResetNumHiddenLayerNeurons(0, InputLayer.size());
}

size_t FFBPNeuralNet::GetNumHiddenLayers(void) const
{
	return HiddenLayers.size();
}

void FFBPNeuralNet::AddHiddenLayer(const size_t& insert_before_index, const size_t& src_num_hidden_layer_neurons)
{
	vector<WeightedNeuron> NewHiddenLayer;

	if (insert_before_index == 0) // insert before first layer
	{
		for (size_t i = 0; i < src_num_hidden_layer_neurons; i++)
			NewHiddenLayer.push_back(WeightedNeuron(InputLayer.size()));

		HiddenLayers.insert(HiddenLayers.begin(), NewHiddenLayer);

		ResetNumHiddenLayerNeurons(1, src_num_hidden_layer_neurons);
	}
	else if (insert_before_index >= HiddenLayers.size()) // insert after last layer
	{
		for (size_t i = 0; i < src_num_hidden_layer_neurons; i++)
			NewHiddenLayer.push_back(WeightedNeuron(HiddenLayers[HiddenLayers.size() - 1].size()));

		HiddenLayers.insert(HiddenLayers.begin() + HiddenLayers.size(), NewHiddenLayer);

		ResetNumOutputLayerNeurons(src_num_hidden_layer_neurons);
	}
	else
	{
		for (size_t i = 0; i < src_num_hidden_layer_neurons; i++)
			NewHiddenLayer.push_back(WeightedNeuron(HiddenLayers[insert_before_index].size()));

		HiddenLayers.insert(HiddenLayers.begin() + insert_before_index, NewHiddenLayer);

		ResetNumHiddenLayerNeurons(insert_before_index - 1, src_num_hidden_layer_neurons);
	}
}

void FFBPNeuralNet::RemoveHiddenLayer(const size_t& index)
{
	if (index >= HiddenLayers.size())
		throw out_of_range("Invalid hidden layer index.");

	if (HiddenLayers.size() == 1)
		throw out_of_range("Invalid number of hidden layers.");

	if (index == 0)
		ResetNumHiddenLayerNeurons(1, InputLayer.size());
	else if (index == HiddenLayers.size() - 1)
		ResetNumOutputLayerNeurons(HiddenLayers[index - 1].size());
	else
		ResetNumHiddenLayerNeurons(index + 1, HiddenLayers[index - 1].size());

	HiddenLayers.erase(HiddenLayers.begin() + index);
}


size_t FFBPNeuralNet::GetNumHiddenLayerNeurons(const size_t& index) const
{
	if (index >= HiddenLayers.size())
		throw out_of_range("Invalid hidden layer index.");

	return HiddenLayers[index].size();
}

void FFBPNeuralNet::ResetNumHiddenLayerNeurons(const size_t& index, const size_t& src_num_hidden_layer_neurons)
{
	if (index >= HiddenLayers.size())
		throw out_of_range("Invalid hidden layer index.");

	size_t temp_layer_size = HiddenLayers[index].size();

	if (index == 0) // is this the first hidden layer?
	{
		if (src_num_hidden_layer_neurons < temp_layer_size)
		{
			for (size_t i = 0; i < temp_layer_size - src_num_hidden_layer_neurons; i++)
				HiddenLayers[index].pop_back();
		}
		else if (src_num_hidden_layer_neurons > temp_layer_size)
		{
			for (size_t i = 0; i < src_num_hidden_layer_neurons - temp_layer_size; i++)
				HiddenLayers[index].push_back(WeightedNeuron(InputLayer.size()));
		}

		// is it also the last hidden layer?
		if (index == HiddenLayers.size() - 1)
			ResetNumOutputLayerNeurons(src_num_hidden_layer_neurons);
		else
			ResetNumHiddenLayerNeurons(index + 1, src_num_hidden_layer_neurons);
	}
	else if (index == HiddenLayers.size() - 1) // the last layer (but obviously not the first)
	{
		if (src_num_hidden_layer_neurons < temp_layer_size)
		{
			for (size_t i = 0; i < temp_layer_size - src_num_hidden_layer_neurons; i++)
				HiddenLayers[index].pop_back();
		}
		else if (src_num_hidden_layer_neurons > temp_layer_size)
		{
			for (size_t i = 0; i < src_num_hidden_layer_neurons - temp_layer_size; i++)
				HiddenLayers[index].push_back(WeightedNeuron(HiddenLayers[index - 1].size()));
		}

		ResetNumOutputLayerNeurons(src_num_hidden_layer_neurons);
	}
	else // a layer in the middle
	{
		if (src_num_hidden_layer_neurons < temp_layer_size)
		{
			for (size_t i = 0; i < temp_layer_size - src_num_hidden_layer_neurons; i++)
				HiddenLayers[index].pop_back();
		}
		else if (src_num_hidden_layer_neurons > temp_layer_size)
		{
			for (size_t i = 0; i < src_num_hidden_layer_neurons - temp_layer_size; i++)
				HiddenLayers[index].push_back(WeightedNeuron(HiddenLayers[index - 1].size()));
		}

		ResetNumHiddenLayerNeurons(index + 1, src_num_hidden_layer_neurons);
	}
}

size_t FFBPNeuralNet::GetNumOutputLayerNeurons(void) const
{
	return OutputLayer.size();
}

void FFBPNeuralNet::ResetNumOutputLayerNeurons(const size_t& src_num_output_neurons)
{
	if (src_num_output_neurons == 0)
		throw out_of_range("Invalid number of output neurons.");

	size_t temp_output_size = OutputLayer.size();

	if (src_num_output_neurons < temp_output_size)
	{
		for (size_t i = 0; i < temp_output_size - src_num_output_neurons; i++)
			OutputLayer.pop_back();
	}
	else if (src_num_output_neurons > temp_output_size)
	{
		for (size_t i = 0; i < src_num_output_neurons - temp_output_size; i++)
			OutputLayer.push_back(WeightedNeuron(HiddenLayers[HiddenLayers.size() - 1].size()));
	}
}

double FFBPNeuralNet::GetLearningRate(void) const
{
	return learning_rate;
}

void FFBPNeuralNet::SetLearningRate(const double& src_learning_rate)
{
	learning_rate = src_learning_rate;
}

double FFBPNeuralNet::GetMomentum(void) const
{
	return momentum;
}

void FFBPNeuralNet::SetMomentum(const double& src_momentum)
{
	momentum = src_momentum;
}

void FFBPNeuralNet::SaveToFile(const char* const filename) const
{
	ofstream out(filename, ios::binary);

	if (out.fail())
		throw runtime_error("Error creating/opening file.");

	size_t temp_size_t = 0;
	double temp_double = 0.0;

	// write num input neurons
	temp_size_t = InputLayer.size();
	out.write((const char*)&temp_size_t, sizeof(size_t));
	if (out.fail())
		throw runtime_error("Error writing to file.");

	// write num hidden layers
	temp_size_t = HiddenLayers.size();
	out.write((const char*)&temp_size_t, sizeof(size_t));
	if (out.fail())
		throw runtime_error("Error writing to file.");

	// write num neurons per hidden layer
	for (size_t i = 0; i < HiddenLayers.size(); i++)
	{
		temp_size_t = HiddenLayers[i].size();
		out.write((const char*)&temp_size_t, sizeof(size_t));
		if (out.fail())
			throw runtime_error("Error writing to file.");
	}

	// write num output neurons
	temp_size_t = OutputLayer.size();
	out.write((const char*)&temp_size_t, sizeof(size_t));
	if (out.fail())
		throw runtime_error("Error writing to file.");

	// write learning_rate
	out.write((const char*)&learning_rate, sizeof(double));
	if (out.fail())
		throw runtime_error("Error writing to file.");

	// write momentum
	out.write((const char*)&momentum, sizeof(double));
	if (out.fail())
		throw runtime_error("Error writing to file.");

	// for each hidden layer
	for (size_t i = 0; i < HiddenLayers.size(); i++)
	{
		// for each neuron
		for (size_t j = 0; j < HiddenLayers[i].size(); j++)
		{
			// write num input weights
			temp_size_t = HiddenLayers[i][j].GetNumInputs();
			out.write((const char*)&temp_size_t, sizeof(size_t));
			if (out.fail())
				throw runtime_error("Error writing to file.");

			// for each input
			for (size_t k = 0; k < HiddenLayers[i][j].GetNumInputs(); k++)
			{
				// write input weight
				temp_double = HiddenLayers[i][j].GetWeight(k);
				out.write((const char*)&temp_double, sizeof(double));
				if (out.fail())
					throw runtime_error("Error writing to file.");

				// write previous weight adjustment
				temp_double = HiddenLayers[i][j].GetPreviousWeightAdjustment(k);
				out.write((const char*)&temp_double, sizeof(double));
				if (out.fail())
					throw runtime_error("Error writing to file.");
			}

			// write bias
			temp_double = HiddenLayers[i][j].GetBias();
			out.write((const char*)&temp_double, sizeof(double));
			if (out.fail())
				throw runtime_error("Error writing to file.");

			// write bias weight
			temp_double = HiddenLayers[i][j].GetBiasWeight();
			out.write((const char*)&temp_double, sizeof(double));
			if (out.fail())
				throw runtime_error("Error writing to file.");

			// write previous bias weight adjustment
			temp_double = HiddenLayers[i][j].GetPreviousBiasWeightAdjustment();
			out.write((const char*)&temp_double, sizeof(double));
			if (out.fail())
				throw runtime_error("Error writing to file.");
		}
	}

	// for each output neuron
	for (size_t i = 0; i < OutputLayer.size(); i++)
	{
		// write num input weights
		temp_size_t = OutputLayer[i].GetNumInputs();
		out.write((const char*)&temp_size_t, sizeof(size_t));
		if (out.fail())
			throw runtime_error("Error writing to file.");

		// for each input
		for (size_t j = 0; j < OutputLayer[i].GetNumInputs(); j++)
		{
			// write input weight
			temp_double = OutputLayer[i].GetWeight(j);
			out.write((const char*)&temp_double, sizeof(double));
			if (out.fail())
				throw runtime_error("Error writing to file.");

			// write previous weight adjustment
			temp_double = OutputLayer[i].GetPreviousWeightAdjustment(j);
			out.write((const char*)&temp_double, sizeof(double));
			if (out.fail())
				throw runtime_error("Error writing to file.");
		}

		// write bias
		temp_double = OutputLayer[i].GetBias();
		out.write((const char*)&temp_double, sizeof(double));
		if (out.fail())
			throw runtime_error("Error writing to file.");

		// write bias weight
		temp_double = OutputLayer[i].GetBiasWeight();
		out.write((const char*)&temp_double, sizeof(double));
		if (out.fail())
			throw runtime_error("Error writing to file.");

		// write previous bias weight adjustment
		temp_double = OutputLayer[i].GetPreviousBiasWeightAdjustment();
		out.write((const char*)&temp_double, sizeof(double));
		if (out.fail())
			throw runtime_error("Error writing to file.");
	}
}

void FFBPNeuralNet::LoadFromFile(const char* const filename)
{
	ifstream in(filename, ios::binary);

	if (in.fail() || in.eof())
		throw runtime_error("Error opening file.");

	size_t temp_size_t = 0;
	double temp_double = 0.0;
	WeightedNeuron temp_weighted_neuron(1);

	// read num input neurons
	in.read((char*)&temp_size_t, sizeof(size_t));
	if (in.fail() || in.eof())
		throw runtime_error("Error reading from file.");

	ResetNumInputLayerNeurons(temp_size_t);

	// read num hidden layers
	in.read((char*)&temp_size_t, sizeof(size_t));
	if (in.fail() || in.eof())
		throw runtime_error("Error reading from file.");

	HiddenLayers.resize(temp_size_t);

	// read num neurons per hidden layer
	for (size_t i = 0; i < HiddenLayers.size(); i++)
	{
		in.read((char*)&temp_size_t, sizeof(size_t));
		if (in.fail() || in.eof())
			throw runtime_error("Error reading from file.");

		HiddenLayers[i].resize(temp_size_t, temp_weighted_neuron);
	}


	// read num output neurons
	in.read((char*)&temp_size_t, sizeof(size_t));
	if (in.fail() || in.eof())
		throw runtime_error("Error reading from file.");

	OutputLayer.resize(temp_size_t, temp_weighted_neuron);


	// read learning_rate
	in.read((char*)&learning_rate, sizeof(double));
	if (in.fail() || in.eof())
		throw runtime_error("Error reading from file.");

	// read momentum
	in.read((char*)&momentum, sizeof(double));
	if (in.fail() || in.eof())
		throw runtime_error("Error reading from file.");


	// for each hidden layer
	for (size_t i = 0; i < HiddenLayers.size(); i++)
	{
		// for each neuron
		for (size_t j = 0; j < HiddenLayers[i].size(); j++)
		{
			// read num input weights
			in.read((char*)&temp_size_t, sizeof(size_t));
			if (in.fail() || in.eof())
				throw runtime_error("Error reading from file.");

			HiddenLayers[i][j].ResetNumInputs(temp_size_t);

			// for each input
			for (size_t k = 0; k < HiddenLayers[i][j].GetNumInputs(); k++)
			{
				// read input weight
				in.read((char*)&temp_double, sizeof(double));
				if (in.fail() || in.eof())
					throw runtime_error("Error reading from file.");

				HiddenLayers[i][j].SetWeight(k, temp_double);

				// read previous weight adjustment
				in.read((char*)&temp_double, sizeof(double));
				if (in.fail() || in.eof())
					throw runtime_error("Error reading from file.");

				HiddenLayers[i][j].SetPreviousWeightAdjustment(k, temp_double);
			}

			// read bias
			in.read((char*)&temp_double, sizeof(double));
			if (in.fail() || in.eof())
				throw runtime_error("Error reading from file.");

			HiddenLayers[i][j].SetBias(temp_double);

			// read bias weight
			in.read((char*)&temp_double, sizeof(double));
			if (in.fail() || in.eof())
				throw runtime_error("Error reading from file.");

			HiddenLayers[i][j].SetBiasWeight(temp_double);

			// read previous bias weight adjustment
			in.read((char*)&temp_double, sizeof(double));
			if (in.fail() || in.eof())
				throw runtime_error("Error reading from file.");

			HiddenLayers[i][j].SetPreviousBiasWeightAdjustment(temp_double);
		}
	}

	// for each output neuron
	for (size_t i = 0; i < OutputLayer.size(); i++)
	{
		// read num input weights
		in.read((char*)&temp_size_t, sizeof(size_t));
		if (in.fail() || in.eof())
			throw runtime_error("Error reading from file.");

		OutputLayer[i].ResetNumInputs(temp_size_t);

		// for each input
		for (size_t j = 0; j < OutputLayer[i].GetNumInputs(); j++)
		{
			// read input weight
			in.read((char*)&temp_double, sizeof(double));
			if (in.fail() || in.eof())
				throw runtime_error("Error reading from file.");

			OutputLayer[i].SetWeight(j, temp_double);

			// read previous weight adjustment
			in.read((char*)&temp_double, sizeof(double));
			if (in.fail() || in.eof())
				throw runtime_error("Error reading from file.");

			OutputLayer[i].SetPreviousWeightAdjustment(j, temp_double);
		}

		// read bias
		in.read((char*)&temp_double, sizeof(double));
		if (in.fail() || in.eof())
			throw runtime_error("Error reading from file.");

		OutputLayer[i].SetBias(temp_double);

		// read bias weight
		in.read((char*)&temp_double, sizeof(double));
		if (in.fail() || in.eof())
			throw runtime_error("Error reading from file.");

		OutputLayer[i].SetBiasWeight(temp_double);

		// read previous bias weight adjustment
		in.read((char*)&temp_double, sizeof(double));
		if (in.fail())
			throw runtime_error("Error reading from file.");

		OutputLayer[i].SetPreviousBiasWeightAdjustment(temp_double);
	}
}