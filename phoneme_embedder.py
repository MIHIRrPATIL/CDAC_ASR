import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2PhonemeEmbedder(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 1. The Base Wav2Vec2 Audio Encoder
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 2. Audio projection to the embedding space
        # Use config.classifier_proj_size if exists, else default to 256
        self.proj_size = getattr(config, "classifier_proj_size", 256)
        self.audio_proj = nn.Linear(config.hidden_size, self.proj_size)
        
        # 3. The Learnable Phoneme Dictionary
        # Vocab size includes the CTC blank token (usually at index 0)
        self.vocab_size = config.vocab_size
        self.phoneme_embeddings = nn.Parameter(torch.randn(self.vocab_size, self.proj_size))
        
        # 4. Temperature parameter to scale cosine similarity (learnable)
        # Initialized to a value that gives a reasonable starting temperature (around 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Initialize weights
        self.post_init()

    def forward(self, input_values, attention_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Force plain Python bool — newer transformers configs or internal states can return Tensors here
        return_dict = True if return_dict is None else bool(return_dict)

        # Extract audio features (Shape: [Batch, Time, Hidden_Size])
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        
        # Project audio features to the embedding space
        audio_features = self.audio_proj(hidden_states)
        
        # L2 Normalize audio and phoneme embeddings for Cosine Similarity
        audio_features = audio_features / (audio_features.norm(dim=-1, keepdim=True) + 1e-8)
        phoneme_features = self.phoneme_embeddings / (self.phoneme_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Calculate Cosine Similarity Logits (Shape: [Batch, Time, Vocab_Size])
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(audio_features, phoneme_features.t())
        
        loss = None
        if labels is not None:
            # 1. Prepare targets for CTC
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(dim=-1)
            flattened_targets = labels[labels_mask]

            # 2. Calculate input lengths for CTC (downsampled by CNN layers)
            if attention_mask is not None:
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            else:
                input_lengths = torch.full((logits.shape[0],), logits.shape[1], device=logits.device, dtype=torch.long)

            # 3. PyTorch CTCLoss expects log_probs in [Time, Batch, Vocab_Size]
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            # 4. Calculate CTC Loss
            # Blank token is usually 0 in Wav2Vec2 processors
            loss_fn = nn.CTCLoss(blank=self.config.pad_token_id or 0, zero_infinity=True)
            loss = loss_fn(log_probs, flattened_targets, input_lengths, target_lengths)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
        
    def _get_feat_extract_output_lengths(self, input_lengths):
        """Helper to compute downsampled lengths through the CNN layers."""
        # This implementation matches the one in Wav2Vec2ForCTC
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
            
        return input_lengths
