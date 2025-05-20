import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from attention import *  # Assuming attention.py is in the same directory
class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 embed_dim=64, hidden_size=512,
                 enc_layers=1, dec_layers=2,
                 cell_type='gru', dropout=0.3,
                 attention=True):
        super().__init__()
        
        self.cell_type = cell_type.lower()
        self.attention = attention
        self.hidden_size = hidden_size
        
        # Embedding layers
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_dim)
        
        # RNN cell selection
        rnn_dict = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }
        rnn_class = rnn_dict[self.cell_type]
        
        # Encoder
        self.encoder = rnn_class(
            embed_dim, hidden_size, enc_layers,
            dropout=dropout if enc_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder with attention
        decoder_input_size = embed_dim + (hidden_size if attention else 0)
        self.decoder = rnn_class(
            decoder_input_size, hidden_size, dec_layers,
            dropout=dropout if dec_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention module
        self.attention_layer = Attention(hidden_size) if attention else None
        
        # Final projection
        self.fc = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        # Encoder forward
        src_embedded = self.src_embed(src)
        encoder_outputs, hidden = self._run_encoder(src_embedded)
        
        # Adapt hidden states for decoder
        hidden = self._adapt_hidden(hidden, self.decoder.num_layers)
        
        # Decoder initialization
        inputs = trg[:, 0]
        outputs = torch.zeros(batch_size, trg_len, self.fc.out_features).to(src.device)
        
        for t in range(1, trg_len):
            # Attention context
            if self.attention:
                context, _ = self._get_context(hidden, encoder_outputs)
                embedded = torch.cat([self.trg_embed(inputs), context], dim=1)
            else:
                embedded = self.trg_embed(inputs)
            
            embedded = embedded.unsqueeze(1)  # Add sequence dimension
            
            # RNN step
            output, hidden = self._decoder_step(embedded, hidden)
            
            # Store predictions
            outputs[:, t] = output
            
            # Teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            inputs = trg[:, t] if use_teacher else output.argmax(1)
            
        return outputs

    def _run_encoder(self, src_embedded):
        if self.cell_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
            return encoder_outputs, (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(src_embedded)
            return encoder_outputs, hidden

    def _decoder_step(self, embedded, hidden):
        if self.cell_type == 'lstm':
            hidden, cell = hidden
            output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            return self.fc(output.squeeze(1)), (hidden, cell)
        else:
            output, hidden = self.decoder(embedded, hidden)
            return self.fc(output.squeeze(1)), hidden

    def _adapt_hidden(self, hidden, target_layers):
        """Handle variable encoder/decoder layers"""
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            h = self._adjust_layers(h, target_layers)
            c = self._adjust_layers(c, target_layers)
            return (h, c)
        else:  # RNN/GRU
            return self._adjust_layers(hidden, target_layers)

    def _adjust_layers(self, state, target_layers):
        """Replicate or slice hidden states to match target layers"""
        current_layers = state.size(0)
        if current_layers < target_layers:
            return torch.cat([state] + [state[-1:]]*(target_layers - current_layers))
        return state[-target_layers:]

    def _get_context(self, hidden, encoder_outputs):
        """Compute attention context vector"""
        if self.cell_type == 'lstm':
            hidden = hidden[0]  # Use hidden state, not cell state
        return self.attention_layer(hidden[-1], encoder_outputs)