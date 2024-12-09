import numpy as np
import scipy as sp

import torch
import torch.nn as nn
from torch.optim import Adam


class MyLSTM:
    loss_history: np.ndarray

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(hidden_size, input_size) * 0.01
        self.b2 = np.zeros((hidden_size, 1))
        self.W3 = np.random.randn(hidden_size, input_size) * 0.01
        self.b3 = np.zeros((hidden_size, 1))
        self.W4 = np.random.randn(hidden_size, input_size) * 0.01
        self.b4 = np.zeros((hidden_size, 1))
        self.U1 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U3 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U4 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(output_size, hidden_size) * 0.01
        self.Vb = np.zeros((output_size, 1))

    def forward(self, x):
        num_inputs, _ = x.shape

        long_term = np.zeros((self.hidden_size, 1))
        short_term = np.zeros((self.hidden_size, 1))

        # Records
        short_record = np.zeros((num_inputs, self.hidden_size))
        long_record = np.zeros((num_inputs, self.hidden_size))
        f_record = np.zeros((num_inputs, self.hidden_size))
        g_record = np.zeros((num_inputs, self.hidden_size))
        i_record = np.zeros((num_inputs, self.hidden_size))
        o_record = np.zeros((num_inputs, self.hidden_size))
        ys = np.zeros((num_inputs, self.output_size))

        for t in range(num_inputs):
            # Gate layer
            forget_output = sp.special.expit(np.dot(self.W1, x[t].reshape(-1, 1)) + np.dot(self.U1, short_term) + self.b1)
            long_term *= forget_output

            # Input layer
            percentage_input = sp.special.expit(np.dot(self.W2, x[t].reshape(-1, 1)) + np.dot(self.U2, short_term) + self.b2)
            state_output = np.tanh(np.dot(self.W3, x[t].reshape(-1, 1)) + np.dot(self.U3, short_term) + self.b3)
            long_term = state_output * percentage_input

            # Output layer
            percentage_output = sp.special.expit(np.dot(self.W4, x[t].reshape(-1, 1)) + np.dot(self.U4, short_term) + self.b4)
            short_term = np.tanh(long_term)
            short_term *=  percentage_output

            # Fully connected layer
            y = np.dot(self.V, sp.special.softmax(short_term)) + self.Vb

            # Record values:
            short_record[t] = short_term.ravel()
            long_record[t] = long_term.ravel()
            f_record[t] = forget_output.ravel()
            g_record[t] = state_output.ravel()
            i_record[t] = percentage_input.ravel()
            o_record[t] = percentage_output.ravel()
            ys[t] = y.ravel()
        return ys, short_record, long_record, f_record, g_record, i_record, o_record

    def backward(self, x, y, targets, short_record, long_record, f_record, g_record, i_record, o_record, learning_rate):
        num_inputs, _ = x.shape
        dW1, dW2, dW3, dW4 = np.zeros_like(self.W1), np.zeros_like(self.W2), np.zeros_like(self.W3), np.zeros_like(self.W4)
        db1, db2, db3, db4 = np.zeros_like(self.b1), np.zeros_like(self.b2), np.zeros_like(self.b3), np.zeros_like(self.b4)
        dU1, dU2, dU3, dU4 = np.zeros_like(self.U1), np.zeros_like(self.U2), np.zeros_like(self.U3), np.zeros_like(self.U4)
        dV, dVb = np.zeros_like(self.V), np.zeros_like(self.Vb)

        for t in reversed(range(num_inputs)):
            pred = y[t].reshape(-1, 1)
            target = targets[t].reshape(-1, 1)

            # Error
            dy = pred - target

            # Forward Layer derivatives
            dV += np.dot(dy, short_record[t].reshape(1, -1))
            dVb += dy

            # General derivatives
            dshort = np.dot(self.V.T, dy)
            dlong = (1 - np.tanh(long_record[t].reshape(-1, 1)) ** 2) * o_record[t].reshape(-1, 1) * dshort

            # Layer derivatives
            do = dshort * np.tanh(long_record[t].reshape(-1, 1)) * o_record[t].reshape(-1, 1) * (1 - o_record[t].reshape(-1, 1))
            di = dlong * g_record[t].reshape(-1, 1) * i_record[t].reshape(-1, 1) * (1 - i_record[t].reshape(-1, 1))
            dg = dlong * i_record[t].reshape(-1, 1) * (1 - g_record[t].reshape(-1, 1) ** 2)
            df = (dlong * long_record[t - 1].reshape(-1, 1) * f_record[t].reshape(-1, 1) * (1 - f_record[t].reshape(-1, 1))) if t > 0 else 0

            # Parameters derivatives
            db4 += do
            dW4 += np.dot(do, x[t].reshape(1, -1))
            dU4 += np.dot(do, short_record[t-1].reshape(1, -1)) if t > 0 else 0
            db3 += dg
            dW3 += np.dot(dg, x[t].reshape(1, -1))
            dU3 += np.dot(dg, short_record[t-1].reshape(1, -1)) if t > 0 else 0
            db2 += di
            dW2 += np.dot(di, x[t].reshape(1, -1))
            dU2 += np.dot(di, short_record[t-1].reshape(1, -1)) if t > 0 else 0
            db1 += df
            dW1 += np.dot(df, x[t].reshape(1, -1))
            dU1 += np.dot(df, short_record[t-1].reshape(1, -1)) if t > 0 else 0

        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
        self.W3 -= learning_rate * dW3
        self.W4 -= learning_rate * dW4
        self.b1 -= learning_rate * db1
        self.b2 -= learning_rate * db2
        self.b3 -= learning_rate * db3
        self.b4 -= learning_rate * db4
        self.U1 -= learning_rate * dU1
        self.U2 -= learning_rate * dU2
        self.U3 -= learning_rate * dU3
        self.U4 -= learning_rate * dU4
        self.V -= learning_rate * dV
        self.Vb -= learning_rate * dVb

    def train(self, inputs, targets, epochs, learning_rate):
        self.loss_history = np.zeros(epochs)
        for epoch in range(epochs):
            outputs, short_record, long_record, f_record, g_record, i_record, o_record = self.forward(inputs)
            loss = np.mean((outputs - targets) ** 2)
            self.loss_history[epoch] = loss
            self.backward(inputs, outputs, targets, short_record, long_record, f_record, g_record, i_record, o_record, learning_rate)
            if epoch % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def overall_loss(self):
        return np.mean(self.loss_history)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    seq_len, input_size, hidden_size, output_size, epochs = 10, 1, 20, 1, 2000
    inputs = np.sin(np.linspace(0, 2 * np.pi, seq_len)).reshape(-1, 1)  # Input
    targets = np.roll(inputs, -1)  # Target

    # My LSTM
    print("\n+------------+\nMy LSTM Model\n+------------+\n")
    lstm = MyLSTM(input_size, hidden_size, output_size)
    lstm.train(inputs, targets, epochs=epochs + 2000, learning_rate=0.01)

    # Built-in LSTM
    builtin_lstm = LSTMModel(input_size, hidden_size, output_size, num_layers=1)
    optimizer = Adam(builtin_lstm.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    inputs_tensor = torch.from_numpy(inputs).float()
    targets_tensor = torch.from_numpy(targets).float()

    print("\n\n+------------+\nBuilt-in LSTM Model\n+------------+\n")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = builtin_lstm(inputs_tensor)
        loss = criterion(output, targets_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{2000}, Loss: {loss.item():.4f}")

    my_output, *other = lstm.forward(inputs)
    builtin_output = builtin_lstm.forward(inputs_tensor)
    print("\n\nTarget\t|\tByHand\t|\tBuiltin")
    for x, y, z in zip(targets.tolist(), my_output.tolist(), builtin_output.tolist()):
        print(f"{x[0]:.4f}\t|\t{y[0]:.4f}\t|\t{z[0]:.4f}")
        print("--------+-----------+----------")
