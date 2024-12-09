import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam


class MyRNN:
    loss_history: np.ndarray

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, input_size) * 0.01
        self.wb = np.zeros((hidden_size, 1))
        self.U = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(output_size, hidden_size) * 0.01
        self.vb = np.zeros((output_size, 1))

    def forward(self, x):
        num_inputs, _ = x.shape
        h = np.zeros((num_inputs, self.hidden_size))
        ys = np.zeros((num_inputs, self.V.shape[0]))
        h0 = np.zeros((hidden_size, 1))
        for t in range(num_inputs):
            h0 = np.tanh(np.dot(self.W, x[t].reshape(-1, 1)) + np.dot(self.U, h0) + self.wb)
            h[t] = h0.ravel()
            y = np.dot(self.V, h0) + self.vb
            ys[t] = y.ravel()
        return ys, h

    def backward(self, x, y, targets, h, learning_rate):
        num_inputs, _ = x.shape
        dW, dU, dwb = np.zeros_like(self.W), np.zeros_like(self.U), np.zeros_like(self.wb)
        dV, dVb = np.zeros_like(self.V), np.zeros_like(self.vb)
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(num_inputs)):
            pred = y[t].reshape(-1, 1)
            target = targets[t].reshape(-1, 1)
            dy = pred - target

            dV += np.dot(dy, h[t].reshape(1, -1))
            dVb += dy

            dh = np.dot(self.V.T, dy) + dh_next
            dh_raw = (1 - h[t].reshape(-1, 1) ** 2) * dh
            dW += np.dot(dh_raw, x[t].reshape(1, -1))
            dU += np.dot(dh_raw, h[t-1].reshape(1, -1)) if t > 0 else 0
            dwb += dh_raw
            dh_next = np.dot(self.U.T, dh_raw)

        self.W -= learning_rate * dW
        self.U -= learning_rate * dU
        self.wb -= learning_rate * dwb
        self.V -= learning_rate * dV
        self.vb -= learning_rate * dVb

    def train(self, inputs, targets, epochs, learning_rate):
        self.loss_history = np.zeros(epochs)
        for epoch in range(epochs):
            outputs, hidden_states = self.forward(inputs)
            loss = np.mean((outputs - targets) ** 2)
            self.loss_history[epoch] = loss
            self.backward(inputs, outputs, targets, hidden_states, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def overall_loss(self):
        return np.mean(self.loss_history)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    seq_len, input_size, hidden_size, output_size, epochs = 10, 1, 20, 1, 2000
    inputs = np.sin(np.linspace(0, 2 * np.pi, seq_len)).reshape(-1, 1)  # Input
    targets = np.roll(inputs, -1)  # Target

    # My RNN
    print("\n+------------+\nMy RNN Model\n+------------+\n")
    rnn = MyRNN(input_size, hidden_size, output_size)
    rnn.train(inputs, targets, epochs=epochs, learning_rate=0.1)

    # Built-in RNN
    builtin_rnn = RNNModel(input_size, hidden_size, output_size, num_layers=1)
    optimizer = Adam(builtin_rnn.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    inputs_tensor = torch.from_numpy(inputs).float()
    targets_tensor = torch.from_numpy(targets).float()


    print("\n\n+------------+\nBuilt-in RNN Model\n+------------+\n")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = builtin_rnn(inputs_tensor)
        loss = criterion(output, targets_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    my_output, _ = rnn.forward(inputs)
    builtin_output = builtin_rnn.forward(inputs_tensor)
    print("\n\nTarget\t|\tByHand\t|\tBuiltin")
    for x, y, z in zip(targets.tolist(), my_output.tolist(), builtin_output.tolist()):
        print(f"{x[0]:.4f}\t|\t{y[0]:.4f}\t|\t{z[0]:.4f}")
        print("--------+-----------+----------")
