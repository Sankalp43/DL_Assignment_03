import matplotlib
import numpy as np
import torch
from dataloader import *
from attention import *
from attention_model import *
# This script visualizes the attention heatmaps for a trained Seq2SeqAttention model
# First install PyTorch if needed
# !pip install torch

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def prepare_data(args):
    data = prepare_dakshina_data(
        lang_code=args.lang_code,
        batch_size=args.batch_size,
        base_dir=args.data_path
    )
    logging.info(f"Source vocab size: {len(data['src_vocab'])}")
    logging.info(f"Target vocab size: {len(data['trg_vocab'])}")
    logging.info(f"Train batches: {len(data['train_loader'])}")
    logging.info(f"Dev batches: {len(data['dev_loader'])}")
    logging.info(f"Test batches: {len(data['test_loader'])}")
    return data

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = [
    'DejaVu Sans',           # Good for Latin
    'Noto Sans Devanagari',  # Good for Devanagari
    'Arial Unicode MS',      # Good fallback for both
    'sans-serif'
]
matplotlib.rcParams['font.family'] = [
    'DejaVu Sans',           # Good for Latin
    'Noto Sans Devanagari',  # Good for Devanagari
    'Arial Unicode MS',      # Good fallback for both
    'sans-serif'
]
def plot_attention_heatmaps_with_labels(model, data, num_samples=9):
    model.eval()
    device = next(model.parameters()).device
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Attention Heatmaps", fontsize=22)
    
    test_loader = data['test_loader']
    src_vocab = data['src_vocab']
    trg_vocab = data['trg_vocab']
    src_idx2char = data['src_idx2char']
    trg_idx2char = data['trg_idx2char']
    
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            for i in range(src.size(0)):
                if count >= num_samples:
                    break

                # Get attention weights for each target step
                attention_weights = []
                # To capture attention weights, you need to modify your Attention module to save weights per forward pass
                # Here we assume model.attention_layer returns (context, weights) and you collect weights per step
                output_attn = []
                def hook(module, input, output):
                    # output[1] is the attention weights
                    output_attn.append(output[1].squeeze(0).cpu().numpy())
                handle = model.attention_layer.register_forward_hook(hook)
                _ = model(src[i:i+1], trg[i:i+1], teacher_forcing_ratio=0)
                handle.remove()
                attn_matrix = np.array(output_attn)  # shape: [tgt_len, src_len]

                # Prepare labels
                src_seq = [src_idx2char[idx] for idx in src[i].cpu().tolist()
                           if idx not in [src_vocab['<start>'], src_vocab['<end>'], src_vocab['<pad>']]]
                trg_seq = [trg_idx2char[idx] for idx in trg[i].cpu().tolist()
                           if idx not in [trg_vocab['<start>'], trg_vocab['<end>'], trg_vocab['<pad>']]]
                # Model prediction
                pred_seq = model(src[i:i+1], trg[i:i+1], teacher_forcing_ratio=0)
                pred_ids = pred_seq.argmax(2)[0].cpu().tolist()
                pred_chars = [trg_idx2char[idx] for idx in pred_ids
                              if idx not in [trg_vocab['<start>'], trg_vocab['<end>'], trg_vocab['<pad>']]]
                
                # Plot
                ax = axes[count // 3, count % 3]
                ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(src_seq)))
                ax.set_xticklabels(src_seq, fontsize=14,)
                ax.set_yticks(range(len(trg_seq)))
                ax.set_yticklabels(trg_seq, fontsize=14, fontname='Noto Sans Devanagari')
                ax.tick_params(axis='both', which='both', length=0)
                
                # Show input, target, and output below the plot
                label_text = (
                    f"Input: {''.join(src_seq)}\n"
                    f"Target: {''.join(trg_seq)}\n"
                    f"Output: {''.join(pred_chars)}"
                )
                ax.text(0.5, -0.25, label_text, fontsize=14, ha='center', va='top', transform=ax.transAxes,)
                ax.set_xlabel("Input (Latin)", fontsize=14)
                ax.set_ylabel("Output (Devanagari)", fontsize=14)
                count += 1
            if count >= num_samples:
                break
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('attention_heatmaps.png')

    plt.show()

# if __name__ == "__main__":
#     

#     args = {
#         'data_path': '/path/to/dakshina/data',
#         'lang_code': 'hi',
#         'batch_size': 64,
#         'epochs': 10,
#         'teacher_forcing': 0.5
#     }

  
#     data = prepare_data(args)
#     best_model = Seq2SeqAttention(
#         len(data['src_vocab']),
#         len(data['trg_vocab']),
#         embed_dim=256,  # Replace with best params from sweep
#         hidden_size=512
#     )
#     best_model.load_state_dict(torch.load("best_model.pth"))
    
#     # Assuming you have a trained model
#     model = AttentionModel(src_vocab_size=len(data['src_vocab']),
#                            trg_vocab_size=len(data['trg_vocab']),
#                            hidden_size=256,
#                            cell_type='lstm',
#                            attention=True)
    
#     # Load your trained model weights here
#     model.load_state_dict(torch.load('path_to_your_model.pth'))
    
#     plot_attention_heatmaps_with_labels(model, data)