#ifndef FFBPNEURALNET_H
#define FFBPNEURALNET_H


#include "weighted_neuron.h"


#include <vector>
using std::vector;


class FFBPNeuralNet
{
public:
	FFBPNeuralNet(const size_t& src_num_input_neurons, const vector<size_t>& src_num_hidden_layers_neurons, const size_t& src_num_output_neurons);
	FFBPNeuralNet(const char* const src_filename);

	// to feed data into network
	void FeedForward(const vector<double>& src_inputs);

	// to obtain the outputs based on previously fed data
	void GetOutputValues(vector<double>& src_outputs);

	// which output neuron holds the highest value?
	size_t GetMaximumOutputNeuron(void) const;

	// provide desired outputs, which will be compared against the currently set output values
	double BackPropagate(const vector<double>& src_desired_outputs);

	// layer manipulation functions
	size_t GetNumInputLayerNeurons(void) const;
	void ResetNumInputLayerNeurons(const size_t& src_num_input_neurons);

	size_t GetNumHiddenLayers(void) const;
	void AddHiddenLayer(const size_t& insert_before_index, const size_t& src_num_hidden_layer_neurons);
	void RemoveHiddenLayer(const size_t& index);

	size_t GetNumHiddenLayerNeurons(const size_t& index) const;
	void ResetNumHiddenLayerNeurons(const size_t& index, const size_t& src_num_hidden_layer_neurons);

	size_t GetNumOutputLayerNeurons(void) const;
	void ResetNumOutputLayerNeurons(const size_t& src_num_output_neurons);

	double GetLearningRate(void) const;
	void SetLearningRate(const double& src_learning_rate);
	double GetMomentum(void) const;
	void SetMomentum(const double& src_momentum);

	void SaveToFile(const char* const filename) const;
	void LoadFromFile(const char* const filename);

protected:
	vector<double> InputLayer;
	vector< vector<WeightedNeuron> > HiddenLayers;
	vector<WeightedNeuron> OutputLayer;

	double learning_rate;
	double momentum;
};




#endif