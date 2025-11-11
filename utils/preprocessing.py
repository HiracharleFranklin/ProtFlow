import random
from Bio import SeqIO

def load_fasta_file(file_path):
    sequences = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
    return sequences

def contains_special_letters(seq):
    special_letters = "UZOXB"
    for letter in special_letters:
        if letter in seq:
            return True
    return False

def trim_long_seqs(sequences, max_seq_length):
    processed_sequences = []
    for seq in sequences:
        # TODO: Implement preprocessing
        # Convert to upper case, check for invalid characters

        # Trim sequences longer than max_seq_length
        if len(seq) > max_seq_length:
            start = random.randint(0, len(seq) - max_seq_length)
            seq = seq[start:start + max_seq_length]

        if contains_special_letters(seq):
            continue

        processed_sequences.append(seq)
    
    return processed_sequences

def remove_short_seqs(sequences, min_seq_length):
    processed_sequences = []
    for seq in sequences:
        if len(seq) < min_seq_length:
            continue

        processed_sequences.append(seq)
    
    return processed_sequences
