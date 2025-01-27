from typing import Optional, Tuple
from utils import get_logger
import torch.nn
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from transformers import PreTrainedModel
import torch.nn as nn
from transformers.activations import gelu
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.roformer.modeling_roformer import (
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerEmbeddings,
    RoFormerSinusoidalPositionalEmbedding,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaEmbeddings,
)
from transformers.utils import ModelOutput
import copy
from dataclasses import dataclass
import pdb

logger = get_logger(__name__)

# Helper Functions for Transformations
def transform_tokens2bursts(hidden_states, num_bursts, max_burst_length):
    seg_hidden_states = torch.reshape(
        hidden_states,
        (hidden_states.size(0), num_bursts, max_burst_length, hidden_states.size(-1)),
    )
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, max_burst_length, seg_hidden_states.size(-1)
    )
    return hidden_states_reshape

def transform_masks2bursts(hidden_states, num_bursts, max_burst_length):
    seg_hidden_states = torch.reshape(
        hidden_states, (hidden_states.size(0), 1, 1, num_bursts, max_burst_length)
    )
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, 1, 1, seg_hidden_states.size(-1)
    )
    return hidden_states_reshape

def transform_bursts2tokens(seg_hidden_states, num_bursts, max_burst_length):
    hidden_states = seg_hidden_states.contiguous().view(
        seg_hidden_states.size(0) // num_bursts,
        num_bursts,
        max_burst_length,
        seg_hidden_states.size(-1),
    )
    hidden_states = hidden_states.contiguous().view(
        hidden_states.size(0), num_bursts * max_burst_length, hidden_states.size(-1)
    )
    return hidden_states

# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roformer = config.roformer
        self.attention = RoFormerAttention(config) if self.roformer else RobertaAttention(config)
        self.intermediate = RoFormerIntermediate(config) if self.roformer else RobertaIntermediate(config)
        self.output = RoFormerOutput(config) if self.roformer else RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, seqNo=None):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos=seqNo if self.roformer else None,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs
class NetFoundEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roformer = config.roformer
        self.layer = nn.ModuleList(
            [
                NetFoundLayer(config)
                if not self.config.flat
                else NetFoundLayerFlat(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.burst_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        self.flow_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_bursts + 1, config.hidden_size // config.num_attention_heads
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_bursts=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_burst_attentions = () if output_attentions else None

        burst_seqs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=self.config.max_burst_length
        )
        past_key_values_length = 0
        burstSeqNo = self.burst_positions(burst_seqs.shape[:-1], past_key_values_length)[None, None, :, :]
        flow_seqs = transform_bursts2tokens(
            burst_seqs,
            num_bursts=num_bursts,
            max_burst_length=self.config.max_burst_length,
        )[:, :: self.config.max_burst_length]
        flowSeqNo = self.flow_positions(flow_seqs.shape[:-1], past_key_values_length)[None, None, :, :]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, burstSeqNo, flowSeqNo)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, num_bursts, output_attentions, burstSeqNo, flowSeqNo
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_burst_attentions = all_burst_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_burst_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithFlowAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            flow_attentions=all_burst_attentions,
        )
class NetFoundEmbeddingsWithMeta:
    def __init__(self, config):
        self.metaEmbeddingLayer1 = nn.Linear(config.metaFeatures, 1024)
        self.metaEmbeddingLayer2 = nn.Linear(1024, config.hidden_size)
        self.no_meta = config.no_meta
        self.protoEmbedding = nn.Embedding(65536, config.hidden_size)
        self.compressEmbeddings = nn.Linear(config.hidden_size * 3, config.hidden_size)

    def addMetaEmbeddings(
        self,
        embeddings,
        direction=None,
        iats=None,
        bytes=None,
        pkt_count=None,
        protocol=None,
    ):
        linearLayerDtype = self.metaEmbeddingLayer1.weight.dtype
        if not self.no_meta:
            metaEmbeddings = self.metaEmbeddingLayer2(
                self.metaEmbeddingLayer1(
                    torch.concat(
                        [
                            direction.unsqueeze(2).to(linearLayerDtype),
                            bytes.unsqueeze(2).to(linearLayerDtype) / 1000,
                            pkt_count.unsqueeze(2).to(linearLayerDtype),
                            iats.unsqueeze(2).to(linearLayerDtype),
                        ],
                        dim=-1,
                    )
                )
            )
            embeddings = torch.concat([embeddings, metaEmbeddings], dim=-1)
        else:
            embeddings = torch.concat(
                [embeddings, torch.zeros_like(embeddings)], dim=-1
            )

        protoEmbeddings = (
            self.protoEmbedding(protocol).unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        )

        return self.compressEmbeddings(torch.concat([embeddings, protoEmbeddings], dim=-1))


class NetFoundRobertaEmbeddings(RobertaEmbeddings, NetFoundEmbeddingsWithMeta):
    def __init__(self, config):
        RobertaEmbeddings.__init__(self, config)
        NetFoundEmbeddingsWithMeta.__init__(self, config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        direction=None,
        iats=None,
        bytes=None,
        pkt_count=None,
        protocol=None,
    ):
        position_ids = self.create_position_ids_from_input_ids(
            input_ids, self.padding_idx, self.position_ids
        )
        embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.addMetaEmbeddings(
            embeddings, direction, iats, bytes, pkt_count, protocol
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
        mask = input_ids.ne(padding_idx).int()
        position_ids = (
            position_ids.repeat(
                input_ids.shape[0], input_ids.shape[1] // position_ids.shape[1]
            )
            * mask
        )
        return position_ids


class NetFoundRoformerEmbeddings(RoFormerEmbeddings, NetFoundEmbeddingsWithMeta):
    def __init__(self, config):
        RoFormerEmbeddings.__init__(self, config)
        NetFoundEmbeddingsWithMeta.__init__(self, config)
        self.roformer = config.roformer

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        direction=None,
        iats=None,
        bytes=None,
        pkt_count=None,
        protocol=None,
    ):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.addMetaEmbeddings(
            embeddings, direction, iats, bytes, pkt_count, protocol
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
