import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import numpy as np

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 embed_dim=256, hidden_size=512,
                 enc_layers=2, dec_layers=2,
                 cell_type='lstm', dropout=0.3):
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type.lower()
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers

        # Embedding layers
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_dim)
        
        # RNN cell selection and initialization
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
        
        # Decoder 
        self.decoder = rnn_class(
            embed_dim, hidden_size, dec_layers,
            dropout=dropout if dec_layers > 1 else 0,
            batch_first=True
        )
        
        # Final projection layer
        self.fc = nn.Linear(hidden_size, trg_vocab_size)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        # Encoder forward
        src_embedded = self.src_embed(src)
        encoder_outputs, hidden = self._run_encoder(src_embedded)
        
        # Adjust hidden states for decoder
        if self.cell_type == 'lstm':
            hidden = self._adapt_hidden(hidden, self.dec_layers)
        else:
            if self.enc_layers > 1:  # Handle multi-layer RNN/GRU encoder
                hidden = hidden[-self.dec_layers:]
            else:  # Single-layer encoder → expand for multi-layer decoder
                hidden = hidden.repeat(self.dec_layers, 1, 1)
        
        # Decoder initialization
        inputs = trg[:, 0]
        outputs = torch.zeros(batch_size, trg_len, self.trg_vocab_size).to(src.device)
        
        # Decoder steps
        for t in range(1, trg_len):
            trg_embedded = self.trg_embed(inputs).unsqueeze(1)
            
            if self.cell_type == 'lstm':
                out, (hidden, cell) = self.decoder(trg_embedded, hidden)
                hidden = (hidden, cell)
            else:
                out, hidden = self.decoder(trg_embedded, hidden)
            
            output = self.fc(out.squeeze(1))
            outputs[:, t] = output
            
            # Teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inputs = trg[:, t] if use_teacher else top1
            
        return outputs


    def _adapt_hidden(self, hidden, target_layers):
        """Adjust hidden states to match target layer count"""
        if isinstance(hidden, tuple):  # LSTM case
            return (hidden[0][-target_layers:], 
                    hidden[1][-target_layers:])
        else:  # RNN/GRU case
            return hidden[-target_layers:]

    
    def _run_encoder(self, src_embedded):
        if self.cell_type == 'lstm':
            outputs, (hidden, cell) = self.encoder(src_embedded)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.encoder(src_embedded)
            return outputs, hidden
    
    def _decoder_step(self, inputs, hidden):
        trg_embedded = self.trg_embed(inputs).unsqueeze(1)
        
        if self.cell_type == 'lstm':
            hidden, cell = hidden
            out, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))
            hidden_state = (hidden, cell)
        else:
            out, hidden = self.decoder(trg_embedded, hidden)
            hidden_state = hidden
            
        output = self.fc(out.squeeze(1))
        return output, hidden_state
    
    def beam_search_decode(self, src, beam_width=3, max_len=20):
        self.eval()
        with torch.no_grad():
            src_embedded = self.src_embed(src)
            encoder_outputs, hidden = self._run_encoder(src_embedded)
            
            # Initialize beam
            start_token = self.trg_embed.weight.shape[0] - 4  # <start> index
            beams = [([start_token], 0, hidden)]
            
            for _ in range(max_len):
                candidates = []
                for seq, score, hidden_state in beams:
                    if seq[-1] == self.trg_embed.weight.shape[0] - 3:  # <end>
                        candidates.append((seq, score, hidden_state))
                        continue
                        
                    inputs = torch.LongTensor([seq[-1]]).to(src.device)
                    output, new_hidden = self._decoder_step(inputs, hidden_state)
                    topk_probs, topk_ids = torch.topk(torch.log_softmax(output, dim=1), beam_width)
                    
                    for i in range(beam_width):
                        candidates.append((
                            seq + [topk_ids[0, i].item()],
                            score + topk_probs[0, i].item(),
                            new_hidden
                        ))
                
                # Keep top-k candidates
                candidates.sort(key=lambda x: x[1]/len(x[0]), reverse=True)
                beams = candidates[:beam_width]
                
                # Check if all beams end with <end>
                if all([seq[-1] == self.trg_embed.weight.shape[0] - 3 for seq, _, _ in beams]):
                    break
                    
            # Return best sequence (strip <start> and <end>)
            best_seq = beams[0][0][1:-1]
            return torch.LongTensor(best_seq).unsqueeze(0)