import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_size] (decoder state)
        # encoder_outputs: [batch, seq_len, hidden_size]
        
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden]
        
        energy = self.energy(torch.cat((hidden, encoder_outputs), dim=2))  # [batch, seq_len, 1]
        attention = torch.softmax(energy.squeeze(2), dim=1)  # [batch, seq_len]
        
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
        return context.squeeze(1), attention