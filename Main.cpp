#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <map>

#include "FileIO.h"
#include "NeuralNetwork.h"
#include "Perceptron.h"
#include "PerceptronLayer.h"

void savePredictions(const std::string& filename,
    const std::vector<int>& actualLbl,
    const std::vector<int>& predictlbl) {
    // Check that the actual and predicted labels have the same size
    if (actualLbl.size() != predictlbl.size()) {
        std::cerr << "Error: actual labels and predicted labels have different sizes.\n";
        return;
    }

    // Data structure to hold mispredictions
    // Map from actual label to map of predicted label to count
    std::map<int, std::map<int, int>> mispredictions;

    // Iterate over the labels and count mispredictions
    for (size_t i = 0; i < actualLbl.size(); ++i) {
        if (actualLbl[i] != predictlbl[i]) {
            mispredictions[actualLbl[i]][predictlbl[i]]++;
        }
    }

    // Open the output file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    // Write the header
    outFile << "Actual Label\tMispredicted As\tCount\n";

    // Iterate through the mispredictions map and write to the file
    for (const auto& actualPair : mispredictions) {
        int actual = actualPair.first;
        const auto& predMap = actualPair.second;
        for (const auto& predPair : predMap) {
            int predicted = predPair.first;
            int count = predPair.second;
            outFile << actual << "\t\t" << predicted << "\t\t" << count << "\n";
        }
    }

    // Close the file
    outFile.close();
    std::cout << "Mispredictions written to: " << filename << std::endl;
}
int main()  {
    //setup rand function for perceptron weights
    srand(static_cast<unsigned int>(time(0)));

    //num input neurons
    int numInput = 784;
    int numOutput = 10;

    //threshold
    float L = 0.25f;
    float H = 0.75f;

    //parameters
    float learningRate = 0.05f;
    float momentum = 0.9f;
    int epochs = 475;
    int batchSize = 100;

    //grab num of layers
    int numLayers;
    std::cout << "Input the desired amount of layers: ";
    std::cin >> numLayers;
    std::cout << std::endl;
    std::vector<int> layerNeuronCount;

    //grab number of neurons per layer
    for (int i = 0; i < numLayers; i++) {
        int numNeurons;
        std::cout << "Input the desired amount of neurons: ";
        std::cin >> numNeurons;
        std::cout << std::endl;
        layerNeuronCount.push_back(numNeurons);
    }

    //grab data from MNIST set
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    collectImgData(images, labels);

    //create train and test data
    std::vector<std::vector<float>> trainImages;
    std::vector<int> trainLabels;
    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;
    
    splitData(images, labels, trainImages, trainLabels, testImages, testLabels);
    //construct neural network
    NeuralNetwork PerceptronNetwork(numInput, layerNeuronCount, numOutput);

    PerceptronNetwork.loadWeights(R"(C:\Users\Moez_\source\repos\ahmed2mz_HW4\ahmed2mz_HW4\AutoEncoder_Weights.txt)");
    bool caseSelectin = false;

    //train
    float initialTrainError = PerceptronNetwork.evaluate(trainImages, trainLabels);
    std::cout << "Initial Training Eror: " << initialTrainError << "\n";
    float initialTestError = PerceptronNetwork.evaluate(testImages, testLabels);
    std::cout << "Initial Testing Eror: " << initialTestError << "\n";

    //training for epoch count
    for (int i = 1; i < epochs; i++) {
        PerceptronNetwork.train(trainImages, trainLabels, batchSize, learningRate, momentum, L, H, caseSelectin);

        //every 10 epochs output error
        if (i % 10 == 0) {
            std::cout << "At Epoch: " << i << " \n";
            float trainError = PerceptronNetwork.evaluate(trainImages, trainLabels);
            std::cout << "Training Eror: " << trainError << "\n";
            float testError = PerceptronNetwork.evaluate(testImages, testLabels);
            std::cout << "Testing Eror: " << testError << "\n";
        }
    }

    //done training ->save weights
    PerceptronNetwork.saveWeights("Final_Weights.txt");

    //find final predictions
    std::vector<int> trainPredictions = PerceptronNetwork.predict(trainImages);
    savePredictions("Train_Predictions.txt", trainLabels, trainPredictions);

    std::vector<int> testPredictions = PerceptronNetwork.predict(testImages);
    savePredictions("Test_Predictions.txt", testLabels, testPredictions);

    return 0;

}


