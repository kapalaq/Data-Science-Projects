import numpy as np
import scipy as sp

import torch
import torch.nn as nn
from torch.optim import Adam


class MyGRU:
    loss_history: np.ndarray

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wr = np.random.randn(hidden_size, input_size) * 0.01
        self.Ur = np.random.randn(hidden_size, hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1))
        self.Wz = np.random.randn(hidden_size, input_size) * 0.01
        self.Uz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        self.V = np.random.randn(output_size, hidden_size) * 0.01
        self.Vb = np.zeros((output_size, 1))

    def forward(self, x):
        num_inputs, _ = x.shape

        h = np.zeros((self.hidden_size, 1))

        # Records
        h_record = np.zeros((num_inputs, self.hidden_size))
        z_record = np.zeros((num_inputs, self.hidden_size))
        c_record = np.zeros((num_inputs, self.hidden_size))
        r_record = np.zeros((num_inputs, self.hidden_size))
        ys = np.zeros((num_inputs, self.output_size))

        for t in range(num_inputs):
            # Reset Gate layer
            r = sp.special.expit(np.dot(self.Wr, x[t].reshape(-1, 1)) + np.dot(self.Ur, h) + self.br)
            z = sp.special.expit(np.dot(self.Wz, x[t].reshape(-1, 1)) + np.dot(self.Uz, h) + self.bz)

            # Input Gate layer
            c = np.tanh(np.dot(self.Wc, x[t].reshape(-1, 1)) + r * h + self.bc)
            h = (1 - z) * c + z * h

            # Fully connected layer
            y = np.dot(self.V, sp.special.softmax(h)) + self.Vb

            # Record values:
            h_record[t] = h.ravel()
            z_record[t] = z.ravel()
            c_record[t] = c.ravel()
            r_record[t] = r.ravel()
            ys[t] = y.ravel()
        return ys, h_record, z_record, c_record, r_record

    def backward(self, x, y, targets, h_record, z_record, c_record, r_record, learning_rate):
        num_inputs, _ = x.shape
        dWr, dWz, dWc = np.zeros_like(self.Wr), np.zeros_like(self.Wz), np.zeros_like(self.Wc)
        dbr, dbz, dbc = np.zeros_like(self.br), np.zeros_like(self.bz), np.zeros_like(self.bc)
        dUr, dUz = np.zeros_like(self.Ur), np.zeros_like(self.Uz)
        dV, dVb = np.zeros_like(self.V), np.zeros_like(self.Vb)

        for t in reversed(range(num_inputs)):
            pred = y[t].reshape(-1, 1)
            target = targets[t].reshape(-1, 1)

            # Error
            dy = pred - target

            # Forward Layer derivatives
            dV += np.dot(dy, h_record[t].reshape(1, -1))
            dVb += dy

            # General derivatives
            dh = np.dot(self.V.T, dy)

            # Layer derivatives
            dc = dh * (1 - z_record[t].reshape(-1, 1)) * (1 - c_record[t].reshape(-1, 1) ** 2)
            dz = dh * (h_record[t].reshape(-1, 1) - c_record[t].reshape(-1, 1)) * (1 - z_record[t].reshape(-1, 1)) * z_record[t].reshape(-1, 1)
            dr = dc * h_record[t - 1].reshape(-1, 1) * (1 - r_record[t].reshape(-1, 1)) * r_record[t].reshape(-1, 1) if t > 0 else 0

            # Parameters derivatives
            dbr += dr
            dWr += np.dot(dr, x[t].reshape(1, -1))
            dUr += np.dot(dr, h_record[t - 1].reshape(1, -1)) if t > 0 else 0
            dbz += dz
            dWz += np.dot(dz, x[t].reshape(1, -1))
            dUz += np.dot(dz, h_record[t - 1].reshape(1, -1)) if t > 0 else 0
            dbc += dc
            dWc += np.dot(dc, x[t].reshape(1, -1))

        self.Wr -= learning_rate * dWr
        self.Wz -= learning_rate * dWz
        self.Wc -= learning_rate * dWc
        self.br -= learning_rate * dbr
        self.bz -= learning_rate * dbz
        self.bc -= learning_rate * dbc
        self.Ur -= learning_rate * dUr
        self.Uz -= learning_rate * dUz
        self.V -= learning_rate * dV
        self.Vb -= learning_rate * dVb

    def train(self, inputs, targets, epochs, learning_rate):
        self.loss_history = np.zeros(epochs)
        for epoch in range(epochs):
            outputs, h_record, z_record, c_record, r_record = self.forward(inputs)
            loss = np.mean((outputs - targets) ** 2)
            self.loss_history[epoch] = loss
            self.backward(inputs, outputs, targets, h_record, z_record, c_record, r_record, learning_rate)
            if epoch % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def overall_loss(self):
        return np.mean(self.loss_history)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    seq_len, input_size, hidden_size, output_size, epochs = 10, 1, 20, 1, 2000
    inputs = np.sin(np.linspace(0, 2 * np.pi, seq_len)).reshape(-1, 1)  # Input
    targets = np.roll(inputs, -1)  # Target

    # My LSTM
    print("\n+------------+\nMy GRU Model\n+------------+\n")
    gru = MyGRU(input_size, hidden_size, output_size)
    gru.train(inputs, targets, epochs=epochs, learning_rate=0.1)

    # Built-in LSTM
    builtin_gru = GRUModel(input_size, hidden_size, output_size, num_layers=1)
    optimizer = Adam(builtin_gru.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    inputs_tensor = torch.from_numpy(inputs).float()
    targets_tensor = torch.from_numpy(targets).float()

    print("\n\n+------------+\nBuilt-in GRU Model\n+------------+\n")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = builtin_gru(inputs_tensor)
        loss = criterion(output, targets_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch + 1}/{2000}, Loss: {loss.item():.4f}")

    my_output, *other = gru.forward(inputs)
    builtin_output = builtin_gru.forward(inputs_tensor)
    print("\n\nTarget\t|\tByHand\t|\tBuiltin")
    for x, y, z in zip(targets.tolist(), my_output.tolist(), builtin_output.tolist()):
        print(f"{x[0]:.4f}\t|\t{y[0]:.4f}\t|\t{z[0]:.4f}")
        print("--------+-----------+----------")
