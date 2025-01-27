from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, L1Loss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roformer.modeling_roformer import (
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerSinusoidalPositionalEmbedding,
)
import copy
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, sinusoidal_pos=None):
        # Self-attention with sinusoidal position embeddings
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos
        )
        attention_output = self_attention_outputs[0]

        # Apply intermediate and output layers
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        hidden_states = self.layer_norm(hidden_states + layer_output)
        return hidden_states
def transform_tokens_to_chunks(hidden_states, chunk_size):
    # Split sequence into chunks of the specified size
    num_chunks = (hidden_states.size(1) + chunk_size - 1) // chunk_size
    padded_length = num_chunks * chunk_size
    padding = torch.zeros(
        hidden_states.size(0),
        padded_length - hidden_states.size(1),
        hidden_states.size(2),
        device=hidden_states.device
    )
    hidden_states = torch.cat([hidden_states, padding], dim=1)
    return hidden_states.view(hidden_states.size(0), num_chunks, chunk_size, hidden_states.size(2))


def transform_chunks_to_tokens(chunks):
    # Merge chunks back into a single sequence
    batch_size, num_chunks, chunk_size, hidden_size = chunks.size()
    return chunks.view(batch_size, num_chunks * chunk_size, hidden_size)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

    def forward(self, hidden_states, attention_mask, chunk_size):
        # Add positional embeddings
        batch_size, seq_len, hidden_size = hidden_states.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states += position_embeddings

        # Split into chunks
        chunked_hidden_states = transform_tokens_to_chunks(hidden_states, chunk_size)
        chunked_attention_mask = transform_tokens_to_chunks(attention_mask.unsqueeze(-1), chunk_size).squeeze(-1)

        # Process each chunk independently through all layers
        for layer in self.layers:
            # Flatten batch and chunks for processing
            batch_size, num_chunks, chunk_size, hidden_size = chunked_hidden_states.size()
            chunked_hidden_states = chunked_hidden_states.view(-1, chunk_size, hidden_size)
            chunked_attention_mask = chunked_attention_mask.view(-1, 1, chunk_size)

            # Pass through the transformer layer
            chunked_hidden_states = layer(
                chunked_hidden_states, attention_mask=chunked_attention_mask
            )

            # Reshape back to (batch_size, num_chunks, chunk_size, hidden_size)
            chunked_hidden_states = chunked_hidden_states.view(batch_size, num_chunks, chunk_size, hidden_size)

        # Merge chunks back into a single sequence
        merged_hidden_states = transform_chunks_to_tokens(chunked_hidden_states)
        return merged_hidden_states

class ChunkBasedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = TransformerEncoder(config)
        self.aggregate = nn.Linear(config.hidden_size, config.hidden_size)
        self.loss_fn = CrossEntropyLoss()
        self.chunk_size = 228  # Number of tokens per chunk

    def forward(self, input_ids, attention_mask, bytes=None,iats=None, total_bursts=None, directions=None, swappedLabels=None,burstMetasToBeMasked=None,labels=None):
        batch_size, seq_len = input_ids.size()

        # Chunk the inputs
        chunked_input_ids = transform_tokens_to_chunks(input_ids, self.chunk_size)
        chunked_attention_mask = transform_tokens_to_chunks(attention_mask, self.chunk_size)

        # Process each chunk independently
        chunk_outputs = []
        for i in range(chunked_input_ids.size(1)):  # Iterate over chunks
            current_chunk = chunked_input_ids[:, i, :]
            current_mask = chunked_attention_mask[:, i, :]

            # Generate position IDs for the current chunk
            chunk_position_ids = torch.arange(self.chunk_size, dtype=torch.long, device=input_ids.device)
            chunk_position_ids = chunk_position_ids.unsqueeze(0).expand(current_chunk.size(0), -1)

            # Embedding and encoder
            chunk_embeddings = self.embedding(current_chunk)
            position_embeddings = self.encoder.position_embeddings(chunk_position_ids)
            chunk_embeddings += position_embeddings

            # Pass through encoder
            encoded_chunk = self.encoder(chunk_embeddings, current_mask, self.chunk_size)
            chunk_outputs.append(encoded_chunk)

        # Aggregate outputs
        chunk_outputs = torch.cat(chunk_outputs, dim=1)  # Concatenate along sequence dimension
        aggregated_output = chunk_outputs.mean(dim=1)  # Mean pooling
        logits = self.aggregate(aggregated_output)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )
