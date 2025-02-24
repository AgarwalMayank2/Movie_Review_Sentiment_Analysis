import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, vocab_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)

        self.embedding = nn.Embedding(vocab_size, input_size)

        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        self.fc1 = nn.Linear(hidden_size*sequence_length, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x = self.embedding(x)
        batch_size = x.size(0)
        (out, (h1, c1)) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = F.sigmoid(self.fc(out))
        return out