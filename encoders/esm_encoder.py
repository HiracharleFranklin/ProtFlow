from encoders import ESM2


class ESM2EncoderModel(ESM2):
    def __init__(self, config, device, enc_normalizer, decoder_path, max_seq_len):
        super().__init__(config, device=device, decoder_path=decoder_path, max_seq_len=max_seq_len)
        self.enc_normalizer = enc_normalizer

    def batch_encode(
            self,
            sequences
    ):
        outputs, tokenized = super().batch_encode(sequences)
        if self.enc_normalizer is not None:
            outputs = self.enc_normalizer.normalize(outputs)
        return outputs, tokenized

    def batch_decode(
            self,
            embeddings,
            detokenized: bool = True,
            attention_mask=None,
    ):  
        if self.enc_normalizer is not None:
            embeddings = self.enc_normalizer.denormalize(embeddings)
        return super().batch_decode(embeddings, detokenized, attention_mask)
    
    def pred_tokens(self, embeddings):
        if self.enc_normalizer is not None:
            embeddings = self.enc_normalizer.denormalize(embeddings)
        return super().pred_tokens(embeddings)
