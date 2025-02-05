from transformers.utils import logging
from transformers import PretrainedConfig

logger = logging.get_logger(__name__)


class NetfoundConfig(PretrainedConfig):
    model_type = "NetFound"

    def __init__(
        self,
        vocab_size=65539,
        hidden_size=768,
        max_bursts=12,
        max_burst_length=1000,
        model_max_length=1296 + 12,
        num_hidden_layers=6, #changed to 6 from 12
        num_attention_heads=6, #changed to 6 from 12
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1000,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        encoder_layout=None,
        use_cache=True,
        classifier_dropout=None,
        metaFeatures=4,
        roformer=False,
        longformer=True,
        no_meta = False,
        flat = False,
        no_mlm=False,
        no_swapped_bursts = True,
        rep_output_path = None,
        subflow_bursts = 3,
        attention_window=[1000],
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size
        self.max_bursts = max_bursts
        self.max_burst_length = max_burst_length
        self.model_max_length = model_max_length
        self.encoder_layout = encoder_layout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.metaFeatures = metaFeatures
        self.p = 0
        self.pretraining = True
        self.roformer = roformer
        self.no_meta = no_meta
        self.flat = flat
        self.limit_bursts = False
        self.rotary_value = False
        self.subflow_len=-1
        self.no_mlm = no_mlm
        self.no_swapped_bursts = no_swapped_bursts
        self.rep_output_path = rep_output_path
        self.subflow_bursts = subflow_bursts
        self.attention_window = attention_window

