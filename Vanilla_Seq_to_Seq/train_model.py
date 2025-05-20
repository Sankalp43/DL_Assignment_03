# dakshina_train.py

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from dataloader import prepare_dakshina_data
from vanilla_model import Seq2Seq

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------- Argument Parsing ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Seq2Seq Training for Dakshina Transliteration")

    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for RNN')
    parser.add_argument('--enc_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--cell_type', type=str, default='lstm', choices=['lstm', 'gru', 'rnn'], help='RNN cell type')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--wandb_project', type=str, default='DA6401_assignment3', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--lang_code', type=str, default='hi', help='Language code for Dakshina')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/dakshina/dakshina_dataset_v1.0/', help='Path to Dakshina dataset')
    return parser.parse_args()

# ---------------- Data Preparation ---------------- #
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

# ---------------- Training Function ---------------- #
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare_data(args)
    pad_idx = data['trg_vocab']['<pad>']

    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    model = Seq2Seq(
        src_vocab_size=len(data['src_vocab']),
        trg_vocab_size=len(data['trg_vocab']),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        cell_type=args.cell_type,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(data['train_loader'], desc=f"Epoch {epoch+1}"):
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=args.teacher_forcing)
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
                _, predicted = torch.max(output, 1)
                mask = targets != pad_idx
                correct += ((predicted == targets) * mask).sum().item()
                total += mask.sum().item()

        avg_train_loss = train_loss / len(data['train_loader'])
        avg_val_loss = val_loss / len(data['dev_loader'])
        val_acc = correct / total if total > 0 else 0

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
            logging.info(f"New best model saved with val_loss: {avg_val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)
