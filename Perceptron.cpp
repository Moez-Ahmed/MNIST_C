#include "Perceptron.h"
#include <cmath>
#include <cstdlib>

//random value generator
float Perceptron::randomWeight() {
	return static_cast<float>(rand()) / RAND_MAX - 0.5f;
}
//set up constructor
Perceptron::Perceptron(int num) {
	//set up bias weight
	bias = randomWeight();
	changeBias = 0.0;
	//set up all weights
	for (int i = 0; i < num; i++) {
		weight.push_back(randomWeight());
		changeWeight.push_back(0.0);
	}
}

float Perceptron::sigmoid(float num) {
	return 1.0f / (1.0f + exp(-num));
}
//return change in sigmoid
float Perceptron::sigmoidDelta(float num) {
	return num * (1.0f - num);
}

float Perceptron::forwardPropagation(const std::vector<float>& nums) {
	float sum = bias;
	for (int i = 0; i < weight.size(); i++) {
		sum = sum + (weight[i] * nums[i]);
	}
	output = sigmoid(sum);
	return output;
}
//error check to see how off the actual value is output layer -> use gradient descent
void Perceptron::outputErrorGradient(float actual, float L, float H) {
	float desiredOutput;
	if (actual == 1.0f) {
		desiredOutput = H;
	}
	else {
		desiredOutput = L;
	}
	grad = (desiredOutput - output) * sigmoidDelta(output);
}
//error check to see how off the actual value is in hidden layer
void Perceptron::hiddenErrorGradient(const std::vector<Perceptron>& forward, int numWeight) {
	float totalNeuronWeight = 0.0;
	for (int i = 0; i < forward.size(); i++) {
		totalNeuronWeight = totalNeuronWeight + forward[i].weight[numWeight] * forward[i].grad;
	}
	grad = totalNeuronWeight * sigmoidDelta(output);

}

void Perceptron::updateWeights(const std::vector<float>& values, float numLearn, float momentum) {
	for (int i = 0; i < weight.size(); i++) {
		float changeOld = (numLearn * grad * values[i]) + (momentum * changeWeight[i]);
		//change all weights with the above value
		weight[i] = weight[i] + changeOld;
		changeWeight[i] = changeOld;
	}
	//change bias neuron
	float deltaBias = (numLearn * grad) + (momentum * changeBias);
	bias = bias + deltaBias;
	changeBias = deltaBias;

}