import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        out = self.fc1(X)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = torch.unsqueeze(X, 0)
        out, _ = self.rnn(X)  # out.size() = (1, seq_length, hidden_dim)
        out = self.fc(out[0, :, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = torch.unsqueeze(X, 0)
        out, _ = self.rnn(X)  # out.size() = (1, seq_length, hidden_dim)
        out = self.fc(out[0, :, :])
        return out