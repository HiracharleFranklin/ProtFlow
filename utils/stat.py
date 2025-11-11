import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from encoders import ESM2EncoderModel, EncNormalizerStat
from config import create_config
from Bio import SeqIO

def load_fasta_file(file_path):
    sequences = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
    return sequences

def get_loader(batch_size):
    train_dataset = load_fasta_file(config.data.train_dataset_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )
    return train_loader

def compute_min_max(
        encoder,
        model_name, 
):
    min_ = None
    max_ = None
    num = 0
    batch_size = 64 #512

    train_loader = get_loader(batch_size=batch_size)

    T = tqdm(train_loader)
    print(len(T))

    for i, X in enumerate(T):
        print("Processing batch", i, "/", len(T))
        with torch.no_grad():
            output, _ = encoder.batch_encode(X)

        # output has shape (batch_size, seq_length, embedding_dim)
        # Normalize along the seq_length dimension
        min_batch = torch.min(output, dim=1)[0]  # Min across seq_length, shape (batch_size, embedding_dim)
        max_batch = torch.max(output, dim=1)[0]  # Max across seq_length, shape (batch_size, embedding_dim)

        # Update global min and max
        min_ = min_batch.min(dim=0)[0] if min_ is None else torch.min(min_, min_batch.min(dim=0)[0])
        max_ = max_batch.max(dim=0)[0] if max_ is None else torch.max(max_, max_batch.max(dim=0)[0])

        num += output.shape[0]  # Increment by batch size

        T.set_description(f"min: {[m.item() for m in min_]}, max: {[m.item() for m in max_]}")

    folder_path = f"./data/{config.data.dataset}/"
    os.makedirs(folder_path, exist_ok=True)
    torch.save(min_, f'{folder_path}/encodings-{model_name}-min.pt')
    torch.save(max_, f'{folder_path}/encodings-{model_name}-max.pt')

if __name__ == "__main__":
    config = create_config()
    model_name=config.model.hg_name_hash
    folder_path = f"./data/{config.data.dataset}/"

    enc_normalizer = EncNormalizerStat(
        enc_mean_path=f"{folder_path}/encodings-{model_name}-mean.pt",
        enc_std_path=f"{folder_path}/encodings-{model_name}-std.pt",
    ).cuda()
    
    encoder = ESM2EncoderModel(
        "/facebook/esm2_t12_35M_UR50D", 
        device="cuda:0", 
        decoder_path=None, 
        max_seq_len=50,
        enc_normalizer=enc_normalizer,
    )

    compute_min_max(
        config,
        encoder,
        model_name=config.model.hg_name_hash
    )