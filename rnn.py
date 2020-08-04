import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def train(self, category_tensor, sentence_tensor, learning_rate,
              criterion):
        hidden = self.init_hidden()

        self.zero_grad()

        for i in range(sentence_tensor.size()[0]):
            output, hidden = self(sentence_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values,
        # multiplied by learning rate
        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()

    def evaluate(self, sentence_tensor):
        hidden = self.init_hidden()

        for i in range(sentence_tensor.size()[0]):
            output, hidden = self(sentence_tensor[i], hidden)

        return output
