import numpy as np
from utils import sigmoid, sigmoid_derivative

class NeuralNetwork():
    
    def __init__(self, n=3, seed=1):
        # Seed the random number generator
        self.seed = seed
        np.random.seed(self.seed)

        # Set synaptic weights to a nx1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1

    def train(self, training_inputs : np.ndarray, training_outputs : np.ndarray, training_iterations : int):
        """
            Ajusta los synaptic_weights de la SNN dada una cantidad dada de training_iterations.
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.predict(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def predict(self, inputs : np.ndarray, thresh=None) -> float:
        """
            Evalua la SNN con el array de inputs dado, redondea a 0 o 1 dado el threshold
        """
        
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.synaptic_weights))
        return output if not thresh else 1 if output > thresh else 0

    def __repr__(self):
        return "NeuralNetwork(input_neurons={}, seed={})".format(len(self.synaptic_weights), self.seed)

if __name__ == "__main__":
    
    # The training set
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1],
                                [1,1,0],
                                [0,1,0]])

    training_outputs = np.array([[0,1,1,0,1,0]]).T

    # Initialize the single neuron neural network
    n = training_inputs.shape[1]
    nn = NeuralNetwork(n)
    print(nn)
    
    print("Random starting synaptic weights: ")
    print(nn.synaptic_weights)

    # Train the neural network
    nn.train(training_inputs, training_outputs, epochs = 100000)

    print("Synaptic weights after training: ")
    print(nn.synaptic_weights)

    # Try the neural network
    in_vals = [input("Enter input "+str(i+1)+": ") for i in range(n)]
    out_vals = nn.predict(np.array([int(i) for i in in_vals]), thresh=0.5)
    print("Input data: ", ", ".join(in_vals))
    print("Output data: ", out_vals)
