#pragma once
#include <vector>

class Perceptron {
private:
	float randomWeight();
	float changeBias;
	
	std::vector<float> changeWeight;
	float sigmoid(float num);
	float sigmoidDelta(float num);

public:
	std::vector<float> weight;
	float bias;
	float grad; //
	float output; //sum
	//constructor
	Perceptron(int num);
	//forward propagation
	float forwardPropagation(const std::vector<float>& num);
	//gradient calcs
	void outputErrorGradient(float num, float L, float H);
	void hiddenErrorGradient(const std::vector<Perceptron>& num, int num2);
	//weight calcs
	void updateWeights(const std::vector<float>& num, float numLearn, float change);

};
