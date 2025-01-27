import os
import random
from typing import List, Union, Optional, Tuple
import itertools
import numpy as np
import pyarrow as pa  # Import pyarrow for large_list support
from transformers import PreTrainedTokenizer, BatchEncoding
from datasets.formatting.formatting import LazyBatch
import pdb

PROTOS_TO_LEN = {6: 18, 1: 13, 17: 12}

class NetFoundTokenizer(PreTrainedTokenizer):
    CLS_TOKEN = 65537
    PAD_TOKEN = 0
    mask_token = 65538
    vocab_size = 65539
    ATTN_PRESENCE_TOKEN = 1
    ATTN_ABSENCE_TOKEN = 0

    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.max_bursts = 200000
        self.max_burst_length = 200000
        self.p = config.p
        self.pretraining = config.pretraining
        self.name_or_path = config.name_or_path
        self.limit_bursts = config.limit_bursts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, max_bursts={self.max_bursts}, "
            f"max_burst_length={self.max_burst_length}, p={self.p})"
        )

    @property
    def all_special_ids(self) -> List[int]:
        return [self.CLS_TOKEN, self.PAD_TOKEN]

    def save_pretrained(self, save_directory: Union[str, os.PathLike], legacy_format: Optional[bool] = None,
                        filename_prefix: Optional[str] = None, push_to_hub: bool = False, **kwargs) -> Tuple[str]:
        return

    def __len__(self):
        return self.vocab_size

    def pad_bursts(self, flow: list[list[int]], max_burst_length: int, pad_token: Optional[int] = None) -> list[list[int]]:
        pad_token = pad_token if pad_token is not None else self.PAD_TOKEN
        padded_bursts = [
            burst[:max_burst_length] + [pad_token] * max(0, max_burst_length - len(burst))
            for burst in flow
        ]
        return padded_bursts

    def pad_flow(self, flow: list[list[int]], max_bursts: int, token: int = None) -> list[int]:
        token = token if token is not None else self.PAD_TOKEN
        pad_bursts = max(0, max_bursts - len(flow))
        flow = list(itertools.chain.from_iterable(flow[:max_bursts]))  # Flatten
        flow.extend([token] * len(flow[0]) * pad_bursts)
        return flow

    @staticmethod
    def prepend_to_list(flow: list[list[int]], token: Optional[int]) -> list[list[int]]:
        return [[token] + burst if token is not None else burst for burst in flow]

    @staticmethod
    def convert_to_tokens(flow: list[list[int]], add_one: bool = False) -> list[list[int]]:
        return [[tok + add_one for tok in burst] for burst in flow] if add_one else flow

    @staticmethod
    def convert_to_attn(bursts):
        return [[1] * len(burst) for burst in bursts]

    def __call__(self, dataset):
        return self.tokenize(dataset)

    def trunc_flow(self, ls, idxs):
        return [".".join(ls[i].split(".")[:idxs[i]]) + "." for i in range(len(ls))]

    @staticmethod
    def _expand_bursts(flows: list[list[int]], burst_sizes: list[list[int]]) -> list[list[list[int]]]:
        return [[[value] * burst_sizes[idx][i] for i, value in enumerate(flow)] for idx, flow in enumerate(flows)]

    def tokenize(self, text, **kwargs):
        pdb.set_trace()
        dataset: LazyBatch = text
        direction = self.tokenize_fields([[1 if direction else -1 for direction in flow] for flow in dataset["directions"]])
        pkt_bytes = self.tokenize_fields(dataset["bytes"])
        iats = self.tokenize_fields(dataset["iats"])
        input_ids, attention_mask = self.tokenize_fields_with_attn(
            dataset["burst_tokens"], prepend_token=self.CLS_TOKEN, add_one=True
        )

        all_chunked_input_ids, all_chunked_metadata = [], []
        pdb.set_trace()
        for i in range(len(input_ids)):
            chunks, metadata = chunk_with_sliding_window(
                input_ids[i], dataset["rts"][i], attention_mask[i], direction[i], pkt_bytes[i], iats[i],
                dataset["protocol"]
            )
            all_chunked_input_ids.extend(chunks)
            all_chunked_metadata.extend(metadata)
        pdb.set_trace()
        batchDict = {
        "burst_tokens": [meta["protocol"] for meta in all_chunked_metadata],
        "input_ids": [pa.array(chunk, type=pa.large_list(pa.int32())) for chunk in all_chunked_input_ids],
        "attention_mask": [meta["attention_mask"] for meta in all_chunked_metadata],
        "directions": [meta["direction"] for meta in all_chunked_metadata],
        "directions_tok": [meta["direction"] for meta in all_chunked_metadata],
        "bytes": [meta["bytes"] for meta in all_chunked_metadata],
        "iats": [meta["iats"] for meta in all_chunked_metadata],
        "rts": [meta["rts"] for meta in all_chunked_metadata],
        "protocol": [meta["protocol"] for meta in all_chunked_metadata],
        "total_bursts": [meta["total_bursts"] for meta in all_chunked_metadata]
        }
        # batchDict = {
        #     "burst_tokens": [meta["protocol"] for meta in all_chunked_metadata],
        #     "input_ids": [pa.array(chunk, type=pa.large_list(pa.int32())) for chunk in all_chunked_input_ids],
        #     "attention_mask": [meta["attention_mask"] for meta in all_chunked_metadata],
        #     "directions": [meta["direction"] for meta in all_chunked_metadata],
        #     "directions_tok": [meta["direction"] for meta in all_chunked_metadata],
        #     "bytes": [meta["bytes"] for meta in all_chunked_metadata],
        #     "iats": [meta["iats"] for meta in all_chunked_metadata],
        #     "rts": [meta["rts"] for meta in all_chunked_metadata],
        #     "protocol": [meta["protocol"] for meta in all_chunked_metadata],
        #     "total_bursts": [meta["total_bursts"] for meta in all_chunked_metadata]
        # }
        # pdb.set_trace()
        return BatchEncoding(batchDict)

    def tokenize_fields(self, dataset: list[list[list[int]]], prepend_token: int = None, add_one: bool = False) -> list[list[list[int]]]:
        tokenized_data = [
            self.prepend_to_list(self.convert_to_tokens(flow, add_one), prepend_token)
            for flow in dataset
        ]
        return tokenized_data

    def tokenize_fields_with_attn(self, dataset: list[list[list[int]]], prepend_token: int = None, add_one: bool = False) -> Tuple[list[list[list[int]]], list[list[list[int]]]]:
        tokenized_data = self.tokenize_fields(dataset, prepend_token, add_one)
        attn = [self.prepend_to_list(self.convert_to_attn(flow), self.ATTN_PRESENCE_TOKEN) for flow in dataset]
        return tokenized_data, attn


from typing import List, Tuple, Dict

def chunk_with_sliding_window(
    input_ids: List[List[int]], timestamps: List[int], attention_mask: List[List[int]],
    direction: List[int], pkt_bytes: List[int], iats: List[int], 
    protocol: int, window_size_ms: int = 100, step_size_ms: int = 10, min_packets: int = 50
) -> Tuple[List[List[List[int]]], List[Dict]]:
    # Convert timestamps from nanoseconds to milliseconds
    timestamps_ms = [ts / 1e6 for ts in timestamps]

    # Initialize lists to store chunks and their metadata
    chunked_input_ids, chunked_metadata = [], []

    # Initialize the sliding window
    start_idx, num_packets = 0, len(timestamps_ms)
    pdb.set_trace()
    while start_idx < num_packets:
        # Set the window boundaries
        window_start_time = timestamps_ms[start_idx]
        window_end_time = window_start_time + window_size_ms
        end_idx = start_idx

        # Find the end index of the current window
        while end_idx < num_packets and timestamps_ms[end_idx] <= window_end_time:
            end_idx += 1

        # Packets in the current window
        window_packets = input_ids[start_idx:end_idx]
        attn_mask = attention_mask[start_idx:end_idx]
        window_size = len(window_packets)

        # Determine the number of chunks (ceil of window_size / min_packets)
        num_chunks = (window_size + min_packets - 1) // min_packets

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * min_packets
            chunk_end = min((chunk_idx + 1) * min_packets, window_size)

            # Extract the chunk and its corresponding data
            chunk = window_packets[chunk_start:chunk_end]
            chunk_attn_mask = attn_mask[chunk_start:chunk_end]

            # Pad the chunk if it is smaller than `min_packets`
            pad_len = max(0, min_packets - len(chunk))
            if pad_len > 0:
                chunk.extend([[NetFoundTokenizer.PAD_TOKEN] * len(input_ids[0])] * pad_len)
                chunk_attn_mask.extend([[0] * len(attn_mask[0])] * pad_len)

            # Append the current chunk and its metadata
            chunked_input_ids.append(chunk)
            pdb.set_trace()
            chunked_metadata.append({
                "timestamps": timestamps[start_idx:end_idx][chunk_start:chunk_end] + [0] * pad_len,
                "direction": direction[start_idx:end_idx][chunk_start:chunk_end] + [0] * pad_len,
                "bytes": pkt_bytes[start_idx:end_idx][chunk_start:chunk_end] + [0] * pad_len,
                "iats": iats[start_idx:end_idx][chunk_start:chunk_end] + [0] * pad_len,
                "rts": timestamps[start_idx:end_idx][chunk_start:chunk_end] + [0] * pad_len,
                "protocol": [protocol] * min_packets,
                "attention_mask": chunk_attn_mask,
                "total_bursts": len(chunk)
            })

        # Move the start index by `step_size_ms` to ensure overlap
        next_start_time = window_start_time + step_size_ms

        # Update the start index to the point that meets or exceeds the next start time
        while start_idx < num_packets and timestamps_ms[start_idx] < next_start_time:
            start_idx += 1
    pdb.set_trace()
    return chunked_input_ids, chunked_metadata
