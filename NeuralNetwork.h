#pragma once
#include "PerceptronLayer.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
class NeuralNetwork {
private:
	float error;
	//backpropagation storage values
	std::vector<float> bpStorage;

public:
	//constructor for entire PerceptronNetwork
	NeuralNetwork(int numInputs, const std::vector<int>& NetworkCharacteristics, int numOutputs) {
		int numPerceptronInputs = numInputs;
		for (int i = 0; i < NetworkCharacteristics.size(); i++) {
			int numPerceptrons = NetworkCharacteristics[i];
			pLayers.emplace_back(numPerceptrons, numPerceptronInputs);
			numPerceptronInputs = numPerceptrons;
		}
		pLayers.emplace_back(numOutputs, numPerceptronInputs);
	}

	//count of layers
	std::vector<PerceptronLayer> pLayers;
	//train 
	void train(const std::vector<std::vector<float>>& trainImages, const std::vector<int>& trainLabels, int batchSize, float LearningRate, float momentum, float L, float H, bool caseSelect) {
		std::vector<int> indices(trainImages.size());
		for (int i = 0; i < indices.size(); i++) {
			indices[i] = i;
		}
		std::random_shuffle(indices.begin(), indices.end());

		for (int i = 0; i < batchSize; i++) {
			int index = indices[i];
			const std::vector<float>& image = trainImages[index];
			int label = trainLabels[index];
			std::vector<float> outputs = forwardPropagation(image);
			std::vector<float> target(outputs.size(), 0.0f);
			target[label] = 1.0f;

			backwardPropagation(target, LearningRate, momentum, L, H, caseSelect);
		}
	}
	//forward propagation algo
	std::vector<float> forwardPropagation(const std::vector<float>& Inputs) {
		std::vector<float> Responses = Inputs;
		bpStorage = Inputs; //needed for back propagation
		for (int i = 0; i < pLayers.size(); i++) {
			std::vector<float> newResponse;

			for (int j = 0; j < pLayers[i].layer.size(); j++) {
				Perceptron& perceptron = pLayers[i].layer[j];
				float Response = perceptron.forwardPropagation(Responses);
				newResponse.push_back(Response);
			}
			Responses = newResponse;
		}
		return Responses;
	}
	//backwards propagation algo
	void backwardPropagation(const std::vector<float>& actual, float numLearn, float Momentum, float L, float H, bool csel) {
		int numLayers = pLayers.size();
		//find output error gradients
		PerceptronLayer& outputLayer = pLayers.back();
		for (int i = 0; i < outputLayer.layer.size(); i++) {
			outputLayer.layer[i].outputErrorGradient(actual[i], L, H);
		}
		//find hidden layer gradients
		for (int j = numLayers - 2; j >= 0; j--) {
			PerceptronLayer& hiddenLayer = pLayers[j];
			PerceptronLayer& nextLayer = pLayers[j + 1];

			for (int k = 0; k < hiddenLayer.layer.size(); k++) {
				hiddenLayer.layer[k].hiddenErrorGradient(nextLayer.layer, k);
			}
		}
		//update every layer weight
		for (int i = 0; i < numLayers; i++) {
			if (csel && i == 0) {
				continue;
			}
			std::vector<float> inputs;
			if (i == 0) {
				inputs = bpStorage;
			}
			else {
				PerceptronLayer& prevLayer = pLayers[i - 1];
				for (int j = 0; j < prevLayer.layer.size(); j++) {
					inputs.push_back(prevLayer.layer[j].output);
				}
			}
			
			PerceptronLayer& layer = pLayers[i];
			for (int j = 0; j < layer.layer.size(); j++) {
				layer.layer[j].updateWeights(inputs, numLearn, Momentum);
			}
		}
		 
	}

	float evaluate(const std::vector<std::vector<float>>& images, const std::vector<int>& labels) {
		int wrong = 0;
		//implement winner takes all
		for (int i = 0; i < images.size(); i++) {
			std::vector<float> outputs = forwardPropagation(images[i]);
			int predicted = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

			if (predicted != labels[i]) {
				wrong++;
			}
		}
		return static_cast<float>(wrong) / images.size();
	}

	std::vector<int> predict(const std::vector<std::vector<float>>& images) {
		std::vector<int> predictions;
		for (int i = 0; i < images.size(); i++) {
			const std::vector<float>& image = images[i];
			std::vector<float> outputs = forwardPropagation(image);
			int prediction = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
			predictions.push_back(prediction);
		}
		return predictions;
	}

	//save weights to file
	void saveWeights(const std::string& filename) {
		std::ofstream outFile(filename);
		if (!outFile.is_open()) {
			std::cerr << "Error opening file \n";
			return;
		}
		for (int i = 0; i < pLayers.size(); i++) {
			PerceptronLayer& layer = pLayers[i];
			outFile << "Layer " << i << ":\n";
			for (int j = 0; j < layer.layer.size(); j++) {
				Perceptron& perceptron = layer.layer[j];
				outFile << "Perceptron " << j << ": \n";
				outFile << "Bias: " << perceptron.bias << "\n";
				outFile << "Weights:";
				for (int k = 0; k < perceptron.weight.size(); k++) {
					outFile << " " << perceptron.weight[k];
				}
				outFile << "\n";
			}
			outFile << "\n";
		}
		outFile.close();
		std::cout << "Content in: " << filename << std::endl;
	}

	//load weights
	void loadWeights(const std::string& filename) {
		std::ifstream weightsFile(filename);
		std::string text;
		int layerIndex = -1;
		int neuronIndex = -1;
		bool changehidden = false;
		while (std::getline(weightsFile, text)) {
			text.erase(0, text.find_first_not_of(" \t"));
			text.erase(text.find_last_not_of(" \t") + 1);
			if (text.empty()) { continue; }

			if (text.find("Hidden Layer") == 0) {
				layerIndex = 0;
				neuronIndex = -1;
				changehidden = true;
				continue;
			}
			if (text.find("Output Layer") == 0) {
				changehidden = false;
				continue;
			}
			if (changehidden) {
				if (text.find("Neuron") == 0) {
					int temp = text.find(" ");
					neuronIndex = std::stoi(text.substr(temp + 1)) - 1;
					continue;
				}
				if (text.find("Bias:") == 0) {
					float bias = std::stoi(text.substr(5));
					pLayers[layerIndex].layer[neuronIndex].bias = bias;
					continue;
				}
				if (text.find("Weights:") == 0) {
					std::istringstream iss(text.substr(8));
					std::vector<float> weights;
					float weight;
					while (iss >> weight) {
						weights.push_back(weight);
					}
					pLayers[layerIndex].layer[neuronIndex].weight = weights;
					continue;
				}
			}
		}
		weightsFile.close();
	}
};