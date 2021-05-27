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

    def train_iteration(self, category_tensor, batch_tensor, lengths_tensor,
                        optimizer):
        self.zero_grad()

        outputs = []
        for i in range(batch_tensor.size()[0]):  # batch size
            hidden = self.init_hidden()
            for j in range(lengths_tensor[i]):
                output, hidden = self(batch_tensor[i][j], hidden)
            outputs.append(output.clone())
        if self.activation:
            loss = self.criterion(torch.unbind(torch.stack(outputs), 1)[0],
                                  torch.unbind(category_tensor, 1)[0])
        else:
            loss = self.criterion(torch.stack(outputs), category_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return output, loss.item()

    def evaluate(self, sequence_tensor, is_batch=False):
        hidden = self.init_hidden()
        if is_batch:
            for i in range(sequence_tensor.size()[0]):
                for j in range(sequence_tensor.size()[1]):
                    output, hidden = self(sequence_tensor[i][j], hidden)
        else:
            for i in range(sequence_tensor.size()[0]):
                output, hidden = self(sequence_tensor[i], hidden)

        return output
