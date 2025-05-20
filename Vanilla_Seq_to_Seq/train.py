from dataloader import *
import torch
import torch.nn as nn
from vanilla_model import *
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from vanilla_model import *


PATH = "/kaggle/input/dakshina/dakshina_dataset_v1.0/"
data = prepare_dakshina_data(lang_code='hi', batch_size=64,base_dir = PATH)
print(f"Source vocab size: {len(data['src_vocab'])}")
print(f"Target vocab size: {len(data['trg_vocab'])}")
print(f"Train batches: {len(data['train_loader'])}")
print(f"Dev batches: {len(data['dev_loader'])}")
print(f"Test batches: {len(data['test_loader'])}")

# Check a batch
batch = next(iter(data['train_loader']))
print("Source batch shape:", batch['source'].shape)
print("Target input batch shape:", batch['target_input'].shape)
print("Target output batch shape:", batch['target_output'].shape)

def train(config=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with wandb.init(config=config):
        config = wandb.config
        pad_idx = data['trg_vocab']['<pad>']
        
        # Model initialization
        model = Seq2Seq(
            src_vocab_size=len(data['src_vocab']),
            trg_vocab_size=len(data['trg_vocab']),
            embed_dim=config.embed_dim,
            hidden_size=config.hidden_size,
            enc_layers=config.enc_layers,
            dec_layers=config.dec_layers,
            cell_type=config.cell_type,
            dropout=config.dropout
        ).to(device)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        best_val_loss = float('inf')
        
        for epoch in range(config.epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in tqdm(data['train_loader'], desc=f"Epoch {epoch+1}"):
                src = batch['source'].to(device)
                trg = batch['target'].to(device)
                
                optimizer.zero_grad()
                output = model(src, trg, teacher_forcing_ratio=config.teacher_forcing)
                
                # Reshape for loss calculation
                output = output[:, 1:].reshape(-1, output.size(-1))
                targets = batch['target_output'].to(device).reshape(-1)
                
                loss = criterion(output, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in data['dev_loader']:
                    src = batch['source'].to(device)
                    trg = batch['target'].to(device)
                    
                    output = model(src, trg, teacher_forcing_ratio=0)
                    output = output[:, 1:].reshape(-1, output.size(-1))
                    targets = batch['target_output'].to(device).reshape(-1)
                    
                    loss = criterion(output, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output, 1)
                    mask = targets != pad_idx
                    correct += ((predicted == targets) * mask).sum().item()
                    total += mask.sum().item()
            
            avg_train_loss = train_loss / len(data['train_loader'])
            avg_val_loss = val_loss / len(data['dev_loader'])
            val_acc = correct / total
            
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pth")





