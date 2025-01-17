import torch
from torch import nn


class CurtainsLSTMModel(nn.Module):
    def __init__(self):
        super(CurtainsLSTMModel, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=2)
        self.relu_cnn1 = nn.ReLU()
        self.cnn2 = nn.Conv1d(
            in_channels=256, out_channels=256, padding=1, kernel_size=2
        )
        self.relu_cnn2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2, stride=1)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2)
        self.relu_cnn3 = nn.ReLU()
        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2)
        self.relu_cnn4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2, stride=1)
        self.dense1 = nn.Linear(3, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.dense_cic1 = nn.Linear(76, 200)
        self.batch_norm_cic1 = nn.BatchNorm1d(200)
        self.relu_cic1 = nn.LeakyReLU()
        self.dropout_cic = nn.Dropout(0.2)
        self.dense_cic2 = nn.Linear(200, 200)
        self.batch_norm_cic2 = nn.BatchNorm1d(200)
        self.relu_cic2 = nn.LeakyReLU()
        self.dropout_cic2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(456, 128)
        self.relu_concat1 = nn.LeakyReLU()
        self.dropout_concat1 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(128, 128)
        self.relu_concat2 = nn.LeakyReLU()
        self.dropout_concat2 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(128, 11)
        self.softmax = nn.Softmax(dim=1)

    def getReps(self, X, X2):
        # X3 = X2[:,-24:]
        # X2 = X2[:, :-24]
        # X3 = X3.unsqueeze(1)
        # output_cnn = self.cnn1(X3)
        # output_cnn = self.relu_cnn1(output_cnn)
        # output_cnn = self.cnn2(output_cnn)
        # output_cnn = self.relu_cnn2(output_cnn)
        # output_cnn = self.maxpool1(output_cnn)
        # output_cnn = self.cnn3(output_cnn)
        # output_cnn = self.relu_cnn3(output_cnn)
        # output_cnn = self.cnn4(output_cnn)
        # output_cnn = self.relu_cnn4(output_cnn)
        # output_cnn = self.maxpool2(output_cnn)
        # output_cnn = output_cnn.flatten(start_dim = 1)
        output = self.dense1(X)
        output = torch.transpose(output, 1, 2)
        output = self.batch_norm(output)
        output = torch.transpose(output, 1, 2)  # reverting transpose
        output, hidden = self.lstm1(output)
        output = torch.flip(input=output, dims=[1])
        output, hidden = self.lstm2(output)
        output = torch.flip(input=output, dims=[1])
        output, hidden = self.lstm3(output)
        output = output[:, -1, :]
        output_lstm = self.dropout(output)
        output = self.dense_cic1(X2)
        output = self.batch_norm_cic1(output)
        output = self.relu_cic1(output)
        output = self.dense_cic2(output)
        output = self.batch_norm_cic2(output)
        output = self.relu_cic2(output)
        output = self.dropout_cic(output)
        # output = torch.concatenate([output_lstm, output, output_cnn], dim=1)  # , output_cnn], dim = 1)
        output = torch.concatenate([output_lstm, output], dim=1)
        output = self.dense2(output)
        output = self.relu_concat1(output)
        output = self.dropout_concat1(output)
        output = self.dense3(output)
        output = self.relu_concat2(output)
        return output

    def forward(self, X, X2):
        output = self.getReps(X, X2)
        output = self.dropout_concat2(output)
        output = self.dense4(output)
        return self.softmax(output)
