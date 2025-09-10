#pragma once
#include "Perceptron.h"
#include <vector>

class PerceptronLayer {
public: 
	std::vector<Perceptron> layer;
	//constructor for data structor that holds numInputs per perceptron in each layer
	PerceptronLayer(int numPerceptrons, int numPerceptronInputs) {
		for (int i = 0; i < numPerceptrons; i++) {
			layer.emplace_back(numPerceptronInputs);
		}
	}
};