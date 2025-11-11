import numpy as np
from collections import Counter
from utils.preprocessing import load_fasta_file

class LengthSampler:
    def __init__(self, path, max_len=254):
        data = load_fasta_file(path)
        self.dataset_len = np.clip([len(t) for t in data], a_min=1, a_max=max_len)
        freqs = Counter(self.dataset_len)
        self.distrib = []
        for i in range(max_len + 1):
            self.distrib.append(freqs.get(i, 0))
            
        self.distrib = np.array(self.distrib) / np.sum(self.distrib)
            
    def sample(self, num_samples):
        s = np.argmax(np.random.multinomial(1, self.distrib, size=(num_samples)), axis=1)
        return s