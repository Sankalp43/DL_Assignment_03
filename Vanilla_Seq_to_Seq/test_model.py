import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from dataloader import *
from vanilla_model import *
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


def test_model_and_save_predictions(model, test_loader, 
                                  src_vocab, src_idx2char,
                                  trg_vocab, trg_idx2char, 
                                  filename='vanilla_prediction.tsv'):
    model.eval()
    correct = 0
    total = 0
    pad_idx = trg_vocab['<pad>']
    device = next(model.parameters()).device
    
    inputs_list = []
    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            # Greedy decoding
            outputs = model(src, trg, teacher_forcing_ratio=0)
            outputs_reshaped = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets = batch['target_output'].to(device).reshape(-1)
            
            _, predicted = torch.max(outputs_reshaped, 1)
            mask = targets != pad_idx
            correct += ((predicted == targets) * mask).sum().item()
            total += mask.sum().item()
            
            # Convert indices to strings
            for i in range(src.size(0)):
                # Decode SOURCE (Latin) using source vocab
                src_seq = src[i].cpu().tolist()
                src_chars = [src_idx2char[idx] for idx in src_seq 
                           if idx not in [src_vocab['<start>'], src_vocab['<end>'], src_vocab['<pad>']]]
                latin_input = ''.join(src_chars)
                
                # Decode PREDICTION (Devanagari) using target vocab
                pred_seq = outputs[i].argmax(dim=1).cpu().tolist()
                pred_chars = [trg_idx2char[idx] for idx in pred_seq 
                            if idx not in [trg_vocab['<start>'], trg_vocab['<end>'], trg_vocab['<pad>']]]
                devanagari_pred = ''.join(pred_chars)
                
                # Decode TARGET (Devanagari) using target vocab
                trg_seq = trg[i].cpu().tolist()
                trg_chars = [trg_idx2char[idx] for idx in trg_seq 
                           if idx not in [trg_vocab['<start>'], trg_vocab['<end>'], trg_vocab['<pad>']]]
                devanagari_target = ''.join(trg_chars)
                
                inputs_list.append(latin_input)
                predictions_list.append(devanagari_pred)
                targets_list.append(devanagari_target)
    
    # Save to TSV
    import pandas as pd
    df = pd.DataFrame({
        'latin_input': inputs_list,
        'devanagari_prediction': predictions_list,
        'devanagari_target': targets_list
    })
    df.to_csv(filename, sep='\t', index=False, encoding='utf-8')
    
    return correct / total

args = {} # Define your arguments here
args['lang_code'] = 'hi'
args['data_path'] = 'path/to/data'  # Replace with your data path
args['batch_size'] = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = prepare_data(args)
best_model = Seq2Seq(
    len(data['src_vocab']),
    len(data['trg_vocab']),
    embed_dim=256,  # Replace with best params from sweep
    hidden_size=512
).to(device)
best_model.load_state_dict(torch.load("best_model.pth"))

# Usage
test_acc = test_model_and_save_predictions(
    best_model, 
    data['test_loader'],
    data['src_vocab'],  # Source vocab (Latin)
    data['src_idx2char'],
    data['trg_vocab'],  # Target vocab (Devanagari)
    data['trg_idx2char'],
    'vanilla_prediction.tsv'
)
