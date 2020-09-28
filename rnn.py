import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 setup='BCEwithLL'):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        if setup == 'BCEwithLL':
            self.activation = None
            self.criterion = nn.BCEWithLogitsLoss()
        elif setup == 'NLLoss':
            self.activation = nn.LogSoftmax(dim=1)
            self.criterion = nn.NLLLoss()

    def forward(self, input_tensor, hidden):
        combined = torch.cat((input_tensor, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        if self.activation:
            output = self.activation(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def train_iteration(self, category_tensor, sequence_tensor,
                        optimizer):
        hidden = self.init_hidden()

        self.zero_grad()

        for i in range(sequence_tensor.size()[0]):
            output, hidden = self(sequence_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output, loss.item()

    def evaluate(self, sequence_tensor):
        hidden = self.init_hidden()

        for i in range(sequence_tensor.size()[0]):
            output, hidden = self(sequence_tensor[i], hidden)

        return output
