import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_layer = X
hidden_weights = np.random.uniform(size=(2, 3))
output_weights = np.random.uniform(size=(3, 1))

for _ in range(10000):
    hidden_input = np.dot(input_layer, hidden_weights)
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, output_weights)
    predicted_output = sigmoid(final_input)

    error = y - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(output_weights.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    output_weights += hidden_output.T.dot(d_output)
    hidden_weights += input_layer.T.dot(d_hidden)

print("Predicted Output:\n", predicted_output)
