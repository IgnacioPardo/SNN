import numpy as np

class NeuralNetwork():
    
    def __init__(self, n=3, seed=1):
        # Seed the random number generator
        self.seed = seed
        np.random.seed(self.seed)

        # Set synaptic weights to a nx1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1

    def sigmoid(self, x : float) -> float:
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x : float) -> float:
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)

    def train(self, training_inputs : np.ndarray, training_outputs : np.ndarray, training_iterations : int):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs : np.ndarray, thresh=None) -> float:
        """
        Pass inputs through the neural network to get output
        """
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
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
    epochs = 100000
    nn.train(training_inputs, training_outputs, epochs)

    print("Synaptic weights after training: ")
    print(nn.synaptic_weights)

    # Try the neural network
    in_vals = [input("Enter input "+str(i+1)+": ") for i in range(n)]
    out_vals = nn.think(np.array([int(i) for i in in_vals]), thresh=0.1)
    print("Input data: ", ", ".join(in_vals))
    print("Output data: ", out_vals)