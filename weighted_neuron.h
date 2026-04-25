#ifndef WEIGHTED_NEURON_H
#define WEIGHTED_NEURON_H


#include <vector>
using std::vector;

#include <cmath>
#include <cstdlib>

class WeightedNeuron
{
public:
	WeightedNeuron(const size_t& src_num_inputs);
	size_t GetNumInputs(void) const;
	void ResetNumInputs(const size_t& src_num_inputs);
	void SetInputValues(const vector<double>& src_inputs, bool apply_activation = true);
	void SetWeight(const size_t& index, const double& src_weight);
	double GetWeight(const size_t& index) const;
	void SetPreviousWeightAdjustment(const size_t& index, const double& src_weight_adjustment);
	double GetPreviousWeightAdjustment(const size_t& index) const;
	void SetBiasWeight(const double& src_bias_weight);
	double GetBiasWeight(void) const;
	// FIX: added previous bias weight adjustment getter/setter for momentum on bias updates
	void SetPreviousBiasWeightAdjustment(const double& src_adjustment);
	double GetPreviousBiasWeightAdjustment(void) const;
	double GetValue(void) const;
	void SetBias(const double& src_bias);
	double GetBias(void) const;
	void RandomizeWeights(void);
	void PerturbWeights(const double scale);

	// tanh activation function
	static inline double ActivationFunction(const double& x)
	{
		return tanh(x);
	}

	// derivative of tanh: tanh'(x) = 1 - tanh(x)^2 = 1 - f(x)^2
	// note: input is f(x), not x
	static inline double DerivativeOfActivationFunction(const double& f_x)
	{
		return 1.0 - f_x * f_x;
	}

	inline double GetRandWeight(void) const
	{
		// get value from -1.0 to 1.0
		return (static_cast<double>(rand() % 2001) / 1000.0) - 1.0;
	}

protected:
	double bias_weight, bias;
	double previous_bias_weight_adjustment;  // FIX: tracks previous bias adjustment for momentum
	vector<double> weights;
	vector<double> previous_weight_adjustments;
	double value;
};


#endif