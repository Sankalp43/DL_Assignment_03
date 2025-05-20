import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_lstm_activations(model, data, sample_idx=0, layer=0, max_units=50):
    """
    Plot the activations of LSTM hidden units for a specific input sequence.
    
    Args:
        model: Your Seq2SeqAttention model
        data: Dictionary containing your data loaders and vocabularies
        sample_idx: Index of the sample in the test set to visualize
        layer: Which encoder layer to visualize (0 for first layer)
        max_units: Maximum number of hidden units to display (for readability)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get a sample from the test set
    batch = next(iter(data['test_loader']))
    src = batch['source'][sample_idx:sample_idx+1].to(device)  # Get just one sample
    
    # Record the input text for reference
    src_text = ''.join([data['src_idx2char'][idx.item()] for idx in src[0] 
               if idx.item() not in [data['src_vocab']['<start>'], 
                                    data['src_vocab']['<end>'], 
                                    data['src_vocab']['<pad>']]])
    
    with torch.no_grad():
        # Get embeddings and encoder outputs
        src_embedded = model.src_embed(src)
        encoder_outputs, _ = model._run_encoder(src_embedded)
        # encoder_outputs shape: [batch_size, seq_len, hidden_size]
        
        # Extract activations (remove batch dimension since batch=1)
        activations = encoder_outputs[0].cpu().numpy()  # [seq_len, hidden_size]
        
        # Only keep non-padding tokens
        pad_idx = data['src_vocab']['<pad>']
        seq_length = 0
        for i, token_id in enumerate(src[0].cpu().numpy()):
            if token_id == pad_idx:
                break
            seq_length += 1
        
        activations = activations[:seq_length]
        
        # Create label list for x-axis ticks (characters)
        char_labels = [data['src_idx2char'][idx.item()] for idx in src[0][:seq_length]]
        
    # Limit number of units to display
    if activations.shape[1] > max_units:
        activations = activations[:, :max_units]
    
    # Plot the activations
    plt.figure(figsize=(14, 8))
    plt.imshow(activations.T, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Activation Value')
    plt.xlabel('Input Character Position')
    plt.ylabel('LSTM Hidden Unit')
    plt.title(f'LSTM Encoder Activations for Input: "{src_text}"')
    
    # Set x ticks to show characters
    plt.xticks(range(len(char_labels)), char_labels, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('lstm_activations.png')
    plt.show()
    
    return activations
