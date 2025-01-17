from typing import Optional, Tuple

from utils import get_logger
#do some aggregate/summation/ in the worst case take only the first 12 packets and use that as the representation ? 
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
    RoFormerSinusoidalPositionalEmbedding
)
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaEmbeddings,
)
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers.utils import ModelOutput
import copy
import time
from dataclasses import dataclass
import pdb
logger = get_logger(__name__)

def transform_tokens2bursts(hidden_states, num_bursts, max_burst_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states,
        (hidden_states.size(0), num_bursts, max_burst_length, hidden_states.size(-1)),
    )
    # squash segments into sequence into a single axis (samples * segments, max_segment_length, hidden_size)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, max_burst_length, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_masks2bursts(hidden_states, num_bursts, max_burst_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states, (hidden_states.size(0), 1, 1, num_bursts, max_burst_length)
    )
    # squash segments into sequence into a single axis (samples * segments, 1, 1, max_segment_length)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, 1, 1, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_bursts2tokens(seg_hidden_states, num_bursts, max_burst_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(
        seg_hidden_states.size(0) // num_bursts,
        num_bursts,
        max_burst_length,
        seg_hidden_states.size(-1),
    )
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(
        hidden_states.size(0), num_bursts * max_burst_length, hidden_states.size(-1)
    )
    return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roformer = config.roformer
        self.max_burst_length=config.max_burst_length
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # self.attention = RoFormerAttention(config) if self.roformer else RobertaAttention(config)
        self.attention = LongformerSelfAttention(
           config=config,
           layer_id=0
        )

        self.attention_flow=RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = RoFormerIntermediate(config) if self.roformer else RobertaIntermediate(config)
        self.output = RoFormerOutput(config) if self.roformer else RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        seqNo = None
    ):
        # pdb.set_trace()
        # print(hidden_states.size())
        
        # print(is_index_masked.shape)
        if(attention_mask.shape[3]==1):
             self_attention_outputs = self.attention_flow(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )
        elif not self.roformer:
            attention_mask=attention_mask.view(1,attention_mask.shape[3])
            is_index_masked = attention_mask < 0
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                is_index_masked=is_index_masked
            )
        else:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                sinusoidal_pos = seqNo,
                output_attentions=output_attentions,
            )
        attention_output = self_attention_outputs[0]
        # print("finished attention")
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
            [NetFoundLayer(config) if not self.config.flat else NetFoundLayerFlat(config) for idx in range(config.num_hidden_layers)]
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
            else:
                all_self_attentions = None
                all_burst_attentions = None
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

    def _tie_weights(self):
        original_position_embeddings = None
        for module in self.layer:
            if hasattr(module, "position_embeddings"):
                assert hasattr(module.position_embeddings, "weight")
                if original_position_embeddings is None:
                    original_position_embeddings = module.position_embeddings
                if self.config.torchscript:
                    module.position_embeddings.weight = nn.Parameter(
                        original_position_embeddings.weight.clone()
                    )
                else:
                    module.position_embeddings.weight = (
                        original_position_embeddings.weight
                    )
        return

class NetFoundEmbeddingsWithMeta:
    def __init__(self, config):
        self.metaEmbeddingLayer1 = nn.Linear(config.metaFeatures, 1024)
        self.metaEmbeddingLayer2 = nn.Linear(1024, config.hidden_size)
        self.no_meta = config.no_meta
        self.protoEmbedding = nn.Embedding(65536, config.hidden_size)
        self.compressEmbeddings = nn.Linear(config.hidden_size*3, config.hidden_size)

    def addMetaEmbeddings(self,
                          embeddings,
                          direction=None,
                          iats=None,
                          bytes=None,
                          pkt_count=None,
                          protocol=None):
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
            embeddings = torch.concat([embeddings, metaEmbeddings], dim = -1)
        else:
            embeddings = torch.concat([embeddings, torch.zeroes(embeddings.shape)], dim = -1)
        protoEmbeddings = (
            self.protoEmbedding(protocol).unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        )

        return self.compressEmbeddings(torch.concat([embeddings, protoEmbeddings], dim = -1))

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
            if embeddings.shape[1]!=9500:
                pdb.set_trace()
            embeddings += position_embeddings
        # embeddings = self.addMetaEmbeddings(embeddings, direction, iats, bytes, pkt_count, protocol)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
        # pdb.set_trace()
        # mask = input_ids.ne(padding_idx).int()
        # position_ids = (
        #     position_ids.repeat(
        #         input_ids.shape[0], input_ids.shape[1] // position_ids.shape[1]
        #     )
        #     * mask
        # )
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
        embeddings = self.addMetaEmbeddings(embeddings, direction, iats, bytes, pkt_count, protocol)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class NetFoundLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_burst_length = config.max_burst_length
        self.max_bursts = config.max_bursts
        self.hidden_size = config.hidden_size
        self.burst_encoder = TransformerLayer(config)
        self.flow_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(
            config.max_bursts + 1, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.roformer = config.roformer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_bursts=None,
        output_attentions=False,
        burstSeqNo = None,
        flowSeqNo = None
    ):
        # transform sequences to bursts
        burst_inputs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=self.max_burst_length
        )
        burst_masks = transform_masks2bursts(
            attention_mask,
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )
        # print("doing burst encoder")
        burst_outputs = self.burst_encoder(
            burst_inputs, burst_masks, output_attentions=output_attentions, seqNo = burstSeqNo
        )
        # print("I'm here")
        # flatten bursts back to tokens
        outputs = transform_bursts2tokens(
            burst_outputs[0],
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )

        burst_global_tokens = outputs[:, :: self.max_burst_length].clone()
        burst_attention_mask = attention_mask[:, :, :, :: self.max_burst_length].clone()

        burst_positions = torch.arange(1, num_bursts + 1).repeat(outputs.size(0), 1)\
                              .to(outputs.device) * (burst_attention_mask.reshape(-1, num_bursts) >= -1).int().to(outputs.device)
        outputs[:, :: self.max_burst_length] += self.position_embeddings(burst_positions)
        # print("doing flow encoder !")
        flow_outputs = self.flow_encoder(
            burst_global_tokens,
            burst_attention_mask,
            output_attentions=output_attentions,
            seqNo = flowSeqNo
        )

        # replace burst representative tokens
        outputs[:, :: self.max_burst_length] = flow_outputs[0]
        # print("**********************************************")
        return outputs, burst_outputs, flow_outputs


class NetFoundLayerFlat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_burst_length = config.max_burst_length
        self.max_bursts = config.max_bursts
        self.hidden_size = config.hidden_size
        self.burst_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(
            config.max_bursts + 1, config.hidden_size, padding_idx=config.pad_token_id
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_bursts=None,
        output_attentions=False,
        burstSeqNo = None,
        flowSeqNo = None
    ):
        burst_inputs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=self.max_burst_length
        )
        burst_masks = transform_masks2bursts(
            attention_mask,
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )
        burst_outputs = self.burst_encoder(
            burst_inputs, burst_masks, output_attentions=output_attentions, seqNo = burstSeqNo
        )
        outputs = transform_bursts2tokens(
            burst_outputs[0],
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )
        return outputs, burst_outputs



@dataclass
class BaseModelOutputWithFlowAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    flow_attentions: Optional[Tuple[torch.FloatTensor]] = None


class NetFoundPretrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, NetFoundEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [
                k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore
            ]
            self._keys_to_ignore_on_load_missing = [
                k
                for k in self._keys_to_ignore_on_load_missing
                if k not in del_keys_to_ignore
            ]

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)


class NetFoundBase(NetFoundPretrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.roformer:
            self.embeddings = NetFoundRoformerEmbeddings(config)
        else:
            self.embeddings = NetFoundRobertaEmbeddings(config)
        self.seg_embeddings = torch.nn.Embedding(
            num_embeddings=3, embedding_dim=config.hidden_size
        )
        self.encoder = NetFoundEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iats=None,
        bytes=None,
        pkt_count=None,
        protocol=None,
    ):

        embeddings = self.embeddings(
            input_ids, position_ids, direction, iats, bytes, pkt_count, protocol
        )
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        num_bursts = input_ids.shape[-1] // self.config.max_burst_length
        encoder_outputs = self.encoder(
            embeddings,
            extended_attention_mask,
            num_bursts,
            output_attentions,
            output_hidden_states,
        )
        final_output = encoder_outputs[0]

        if not return_dict:
            return (final_output) + encoder_outputs[1:]
        # print("*********************************************************************************************************************************")
        return BaseModelOutputWithFlowAttentions(
            last_hidden_state=final_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            flow_attentions=encoder_outputs.flow_attentions,
        )

    """
    BaseModel:
    embedding:
        tokens RobertaAttention: vocab->768
        meta:  4->768
    encoder:
    burst : (seqLength+1) X 768
    concat: (seqLength+1)*num_sen X 768

    flow: num_sen X 768
    replace the reps

    """


class LMHead(nn.Module):
    def __init__(self, config):
        config = copy.deepcopy(config)
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class NetFoundLanguageModelling(NetFoundPretrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.base_transformer = NetFoundBase(config)
        self.lm_head = LMHead(config)
        self.no_mlm = config.no_mlm
        self.no_swapped_bursts = config.no_swapped_bursts
        self.attentivePooling = AttentivePooling(config)
        self.max_burst_length = config.max_burst_length
        self.portClassifierHiddenLayer = nn.Linear(config.hidden_size, 65536)
        self.swappedClassifierHiddenLayer = nn.Linear(config.hidden_size, 2)
        self.linearMetadataPred = nn.Linear(config.hidden_size, 3)
        self.dirPred = nn.Linear(config.hidden_size, 2)
        # self.aggregate = MeanPooling()
        self.chunk_size=950
        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(
            input_embeddings, "num_embeddings"
        ):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.base_transformer.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.base_transformer.embeddings.word_embeddings = value

    def maskMeta(self, bursts_to_mask, metaFeature):
        for i in range(bursts_to_mask.shape[0]):
            for j in range(bursts_to_mask.shape[1]):
                if bursts_to_mask[i][j]:
                    metaFeature[i][j*self.max_burst_length:(j+1)*self.max_burst_length] = 0
        return metaFeature


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        directions=None,
        iats=None,
        bytes=None,
        pkt_count=None,
        ports=None,
        swappedLabels=None,
        burstMetasToBeMasked = None,
        protocol=None,
        total_bursts=None
    ):
        # TODO(maybe-hello-world): check the whole function to remove non public parts
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # pdb.set_trace()
        
        #creating ground truths tensors before masking
        # direction_orig = direction.clone().to(torch.long)
        # iat_orig = iats.clone()/1000 #adjusting as values are higher.
        # bytes_orig = bytes.clone()/1000 #adjusting as values are higher.
        # pktCount_orig = pkt_count.clone()

        # direction = self.maskMeta(burstMetasToBeMasked, direction)
        # iats = self.maskMeta(burstMetasToBeMasked, iats)
        # bytes = self.maskMeta(burstMetasToBeMasked, bytes)
        # pktCount = self.maskMeta(burstMetasToBeMasked, pkt_count)
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        all_representations=[]
        # Loop through each data point in the batch
        # pdb.set_trace()
        for i in range(batch_size):
            # Extract the entire sequence for the current data point
            full_input_ids = input_ids[i, :seq_len].unsqueeze(0)
            full_attention_mask = attention_mask[i, :seq_len].unsqueeze(0)

            # Handle padding if necessary
            padding_length = self.chunk_size - full_input_ids.shape[1]
            if padding_length > 0:
                # Pad full_input_ids with `padding_idx` (commonly 0)
                full_input_ids = nn.functional.pad(full_input_ids, (0, padding_length), value=0)

                # Pad full_attention_mask with 0 (no attention to padding tokens)
                full_attention_mask = nn.functional.pad(full_attention_mask, (0, padding_length), value=0)

            # Pass the entire sequence through the transformer
            outputs = self.base_transformer(
                full_input_ids,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # Extract the last hidden state
            combined_representation = outputs[0]  # Last hidden state

            # Append the representation to the list
            all_representations.append(combined_representation)

            # Clear variables and cache to save memory
            del full_input_ids, full_attention_mask, outputs, combined_representation
            torch.cuda.empty_cache()

        # Stack all representations into a single tensor
        final_representation = torch.stack(all_representations, dim=0)

        # Clear the all_representations list to save memory
        del all_representations

        # Adjust the final representation to match the shape of the labels
        final_representation = final_representation[:, :, :labels.shape[1], :]
        torch.cuda.empty_cache()

        ''' WORKING CODE
        for i in range(batch_size):
            data_point_representations = []

            # Loop through each chunk for the current data point
            for chunk_idx in range(int((seq_len)//self.chunk_size)+1):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)

                # Extract chunk
                chunk_input_ids = input_ids[i, start_idx:end_idx].unsqueeze(0)
                chunk_attention_mask = (
                    attention_mask[i, start_idx:end_idx].unsqueeze(0)
                )
                # pdb.set_trace()
                # Pass through base transformer
                padding_length = self.chunk_size - chunk_input_ids.shape[1]
                if padding_length > 0:
                    # Pad chunk_input_ids with `padding_idx` (commonly 0)
                    chunk_input_ids = nn.functional.pad(chunk_input_ids, (0, padding_length), value=0)

                    # Pad chunk_attention_mask with 0 (no attention to padding tokens)
                    chunk_attention_mask = nn.functional.pad(chunk_attention_mask, (0, padding_length), value=0)

                    # Now, both tensors have shape: [batch_size, max_length]
                    # print(f"Padded chunk_input_ids shape: {{chunk_attention_mask.shape}}")
                # pdb.set_trace()
                outputs = self.base_transformer(
                    chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                data_point_representations.append(outputs[0])  # Last hidden state
                del chunk_input_ids, chunk_attention_mask, outputs
                # torch.cuda.empty_cache()
            # pdb.set_trace()
            # concatenated_rep=torch.stack(data_point_representations, dim=2)
            # aggregated_representation = torch.mean(torch.stack(data_point_representations, dim=0), dim=0)
            # all_representations.append(aggregated_representation)
            combined_representation = torch.cat(data_point_representations, dim=1)
            del data_point_representations
            # pdb.set_trace()
            all_representations.append(combined_representation)
            del combined_representation
        # pdb.set_trace()
        final_representation = torch.stack(all_representations, dim=0)
        del all_representations
        torch.cuda.empty_cache()
        '''
        final_representation = final_representation[:,:, :labels.shape[1], :]
        # final_representation = torch.stack(all_representations, dim=0)
        
        # pdb.set_trace()
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # e.g., "20241212-153045"

        # Create a unique file name with the timestamp
        file_name = f'/global/homes/h/haarika/pscratch/network-data-representation/model_output2/window_representations/final_representation_{timestamp}.pt'

        # Save the tensor
        torch.save(final_representation.cpu(), file_name)
        prediction_scores = self.lm_head(final_representation)   
        # pdb.set_trace()
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_fct2 = L1Loss()
            if not self.no_mlm:
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                totalLoss = masked_lm_loss
            # if not self.no_swapped_bursts:
            #     swappedClassificationLoss = loss_fct(swappedLogits, swappedLabels)
            # # logger.debug(f"mlm loss: {masked_lm_loss} , swapped_loss: {swappedClassificationLoss}")
            # if self.no_mlm and self.no_swapped_bursts:
            #     raise RuntimeError("no_mlm and no_swapped_bursts are True. Dont have a valid loss function")
            # else:
            #     if self.no_mlm and not self.no_swapped_bursts:
            #         totalLoss = swappedClassificationLoss
            #     else:
            #         totalLoss = masked_lm_loss
            #         if not self.no_swapped_bursts:
            #             totalLoss += swappedClassificationLoss
            # metaLoss = loss_fct2(metaPreds, metaLabels.to(metaPreds.dtype))
            # dirLoss = loss_fct(directn_logits, direction_orig.flatten())
            # totalLoss += metaLoss
            # if not torch.isnan(dirLoss):
            #     totalLoss += dirLoss
        if not return_dict:
            logger.debug(return_dict)
        # ls,ops = MaskedLMOutput(
        #     loss=totalLoss,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        # pdb.set_trace()
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((totalLoss,) + output) if masked_lm_loss is not None else output

        # return MaskedLMOutput(
        #     loss=totalLoss,
        #     logits=(prediction_scores, protocol),
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        del final_representation
        torch.cuda.empty_cache()
        # pdb.set_trace()
        return MaskedLMOutput(
                loss=totalLoss,
                logits=prediction_scores,
                # logits=(prediction_scores, protocol),
                hidden_states=None,
                attentions=None,
            )


def poolingByConcat(sequence_output, max_burst_length, hidden_size, max_bursts):
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    pads = torch.zeros(
        burstReps.shape[0],
        hidden_size * (max_bursts - burstReps.shape[1]),
        dtype=burstReps.dtype,
    ).to(burstReps.device)
    return torch.concat(
        [torch.reshape(burstReps, (burstReps.shape[0], -1)), pads], dim=-1
    ).to(burstReps.device)


def poolingByMean(sequence_output, attention_mask, max_burst_length):
    burst_attention = attention_mask[:, ::max_burst_length].detach().clone()
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    burst_attention = burst_attention / torch.sum(burst_attention, dim=-1).unsqueeze(
        0
    ).transpose(0, 1)
    orig_shape = burstReps.shape
    burstReps = burst_attention.reshape(
        burst_attention.shape[0] * burst_attention.shape[1], -1
    ) * burstReps.reshape((burstReps.shape[0] * burstReps.shape[1], -1))
    return burstReps.reshape(orig_shape).sum(dim=1)


def poolingByAttention(attentivePooling, sequence_output, max_burst_length):
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    return attentivePooling(burstReps)


class AttentivePooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_dropout = config.hidden_dropout_prob
        self.lin_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)


