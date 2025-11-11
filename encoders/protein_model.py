from abc import ABC, abstractmethod

class ProteinModel(ABC):
    """
    ProteinModel is an abstract base class for protein language models.
    It enforces a consistent interface for encoding and decoding protein sequences.
    """

    @abstractmethod
    def encode(self, sequence):
        """
        Encodes a protein sequence into an embedding.
        
        Args:
            sequence (str): A protein sequence as a string.
        
        Returns:
            torch.Tensor: The embedding of the protein sequence.
        """
        pass

    @abstractmethod
    def decode(self, embedding):
        """
        Decodes an embedding back into a protein sequence.
        
        Args:
            embedding (torch.Tensor): The embedding of a protein sequence.
        
        Returns:
            str: The decoded protein sequence as a string.
        """
        pass
