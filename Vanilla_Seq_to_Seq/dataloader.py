import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Load the splits
def load_dakshina_splits(lang_code='hi', base_dir='dakshina_dataset_v1.0'):
    lex_dir = f"{base_dir}\{lang_code}\lexicons\\"
    train_path = lex_dir + f"{lang_code}.translit.sampled.train.tsv"
    dev_path   = lex_dir + f"{lang_code}.translit.sampled.dev.tsv"
    test_path  = lex_dir + f"{lang_code}.translit.sampled.test.tsv"
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['native', 'latin', 'count'], na_filter = False)
    dev_df   = pd.read_csv(dev_path,
                           sep='\t', header=None, names=['native', 'latin', 'count'], na_filter = False)
    test_df  = pd.read_csv(test_path,  sep='\t', header=None, names=['native', 'latin', 'count'], na_filter = False)
    train_df.drop(["count"], axis = 1,inplace = True)
    dev_df.drop(["count"], axis = 1,inplace = True)
    test_df.drop(["count"], axis = 1,inplace = True)

    return train_df, dev_df, test_df

# 2. Preprocessing: add <start> and <end> tokens, tokenize as characters
def preprocess_df(df):
    def process(text):
        return ' '.join(['<start>'] + list(str(text).strip()) + ['<end>'])
    df['native_proc'] = df['native'].apply(process)
    df['latin_proc'] = df['latin'].apply(process)
    return df

# 3. Build vocabulary from training data only
def build_vocab(texts):
    special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
    chars = set()
    for text in texts:
        chars.update(text.split())
    chars = [c for c in sorted(chars) if c not in special_tokens]
    vocab = special_tokens + chars
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    return char2idx, idx2char

# 4. Convert text to padded sequences of indices
def texts_to_sequences(texts, vocab, max_len=None):
    seqs = []
    for text in texts:
        seq = [vocab.get(c, vocab['<unk>']) for c in text.split()]
        seqs.append(seq)
    if not max_len:
        max_len = max(len(seq) for seq in seqs)
    padded_seqs = [seq + [vocab['<pad>']] * (max_len - len(seq)) for seq in seqs]
    # print(padded_seqs , max_len)
    return np.array(padded_seqs), max_len

# 5. PyTorch Dataset
class TransliterationDataset(Dataset):
    def __init__(self, src_seqs, trg_seqs):
        self.src = torch.LongTensor(src_seqs)
        self.trg = torch.LongTensor(trg_seqs)
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return {
            'source': self.src[idx],
            'target': self.trg[idx],
            'target_input': self.trg[idx][:-1],  # Exclude <end>
            'target_output': self.trg[idx][1:]   # Exclude <start>
        }

# 6. Main function to prepare everything
def prepare_dakshina_data(base_dir,lang_code='hi', batch_size=64):
    # Load splits
    train_df, dev_df, test_df = load_dakshina_splits(lang_code,base_dir)
    train_df = preprocess_df(train_df)
    dev_df = preprocess_df(dev_df)
    test_df = preprocess_df(test_df)

    # Build vocabs from training only
    src_vocab, src_idx2char = build_vocab(train_df['latin_proc'])
    trg_vocab, trg_idx2char = build_vocab(train_df['native_proc'])

    # Find max lengths across all splits for consistent padding
    src_max_len = max(
        train_df['latin_proc'].apply(lambda x: len(x.split())).max(),
        dev_df['latin_proc'].apply(lambda x: len(x.split())).max(),
        test_df['latin_proc'].apply(lambda x: len(x.split())).max()
    )
    trg_max_len = max(
        train_df['native_proc'].apply(lambda x: len(x.split())).max(),
        dev_df['native_proc'].apply(lambda x: len(x.split())).max(),
        test_df['native_proc'].apply(lambda x: len(x.split())).max()
    )

    # Convert to sequences
    train_src, _ = texts_to_sequences(train_df['latin_proc'], src_vocab, src_max_len)
    train_trg, _ = texts_to_sequences(train_df['native_proc'], trg_vocab, trg_max_len)
    dev_src, _ = texts_to_sequences(dev_df['latin_proc'], src_vocab, src_max_len)
    dev_trg, _ = texts_to_sequences(dev_df['native_proc'], trg_vocab, trg_max_len)
    test_src, _ = texts_to_sequences(test_df['latin_proc'], src_vocab, src_max_len)
    test_trg, _ = texts_to_sequences(test_df['native_proc'], trg_vocab, trg_max_len)

    # Datasets and loaders
    train_dataset = TransliterationDataset(train_src, train_trg)
    dev_dataset = TransliterationDataset(dev_src, dev_trg)
    test_dataset = TransliterationDataset(test_src, test_trg)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {
        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'test_loader': test_loader,
        'src_vocab': src_vocab,
        'trg_vocab': trg_vocab,
        'src_idx2char': src_idx2char,
        'trg_idx2char': trg_idx2char,
        'src_max_len': src_max_len,
        'trg_max_len': trg_max_len,
    }


