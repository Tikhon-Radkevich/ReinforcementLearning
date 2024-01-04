import torch
import torch.nn as nn
import torch.optim as optim


class LSTMWithHistory(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWithHistory, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Additional attribute to store intermediate values
        self.intermediate_values = []

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        intermediate_values = []  # to store values at each step

        for i in range(x.size(1)):
            # Forward pass through the LSTM layer for each time step
            out, (h0, c0) = self.lstm(x[:, i:i + 1, :], (h0, c0))

            # Save intermediate values
            intermediate_values.append({
                'output': out.clone().detach(),  # clone and detach to avoid gradient tracking
                'hidden_state': h0.clone().detach(),
                'cell_state': c0.clone().detach()
            })

        # Concatenate outputs from all time steps
        out = self.fc(out[:, -1, :])

        # Save intermediate values for this batch
        self.intermediate_values.append(intermediate_values)

        return out


# Dummy input data
input_size = 10
hidden_size = 20
num_layers = 1
output_size = 1
seq_length = 5
batch_size = 3

x = torch.rand((batch_size, seq_length, input_size))
y = torch.rand((batch_size, output_size))  # Dummy target

# Create an instance of the LSTMWithHistory model
model = LSTMWithHistory(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    output = model(x)

    # Compute the loss
    loss = criterion(output, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
