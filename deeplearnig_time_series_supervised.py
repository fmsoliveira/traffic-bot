import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# LSTM neural network +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

scaler = MinMaxScaler()
demanda_normalizada = scaler.fit_transform(df['Demanda'].values.reshape(-1, 1))

def criar_sequencias(data, seq_length):
    X, Y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
      """
      Other recurrent layers
      self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
      # Or
      self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
      """
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        #self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), torch.zeros(1,1,self.hidden_layer_size))

  

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

seq_length = 12
X, Y = criar_sequencias(demanda_normalizada, seq_length)
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
Y_train, Y_test = Y[:int(len(Y)*0.8)], Y[int(len(Y)*0.8):]

# setup loss function
criterion = nn.MSELoss()  # For regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    outputs = model(x)
    optimizer.zero_grad()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(x)
    print("Predictions:", predictions[:5].squeeze().numpy())
    print("Targets:", y[:5].squeeze().numpy())

# Save and load the model
# Save
torch.save(model.state_dict(), 'lstm_model.pth')

# Load
model.load_state_dict(torch.load('lstm_model.pth'))
