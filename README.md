# MNIST_CPP

A **from-scratch C++ implementation** of a perceptron network trained on the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/).  
This project avoids traditional machine learning libraries and frameworks; everything from file I/O to the perceptron learning rule is implemented manually in C++.

## Features

- Pure C++ implementation (no TensorFlow, PyTorch, scikit-learn, or similar libraries).  
- Custom code for:
  - Reading MNIST image and label files.  
  - Defining perceptrons and layers.  
  - Training a simple neural network via the perceptron learning rule.  

## Why No ML Libraries?

This project demonstrates how neural networks can be built **from first principles**.  
Every step, loading raw MNIST data, computing activations, applying weight updates, is explicitly written in C++, without delegating to high-level ML abstractions.  

## Repository Structure

~ `Main.cpp` – Entry point: sets up and runs training/testing.  
~ `FileIO.cpp / FileIO.h` – Raw file handling for MNIST images/labels.  
~ `Perceptron.cpp / Perceptron.h` – Perceptron logic and learning updates.  
~ `PerceptronLayer.h` – Defines layers of perceptrons (e.g., one per digit).  
~ `NeuralNetwork.h` – Network wrapper combining layers and training routines.  

## Build & Run

```bash
# Clone the repository
git clone https://github.com/Moez-Ahmed/MNIST_C.git
cd MNIST_C

# Compile (example with g++)
g++ Main.cpp FileIO.cpp Perceptron.cpp -o mnist_c -std=c++11

# Run (ensure MNIST .idx files are in the working directory)
./mnist_c
