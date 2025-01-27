import os
import random
from typing import List, Union, Optional, Tuple
import itertools

import numpy as np
from transformers import PreTrainedTokenizer, BatchEncoding
from datasets.formatting.formatting import LazyBatch
import pdb 
PROTOS_TO_LEN = {6: 18, 1: 13, 17: 12}  # TODO(maybe-hello-world): refactor

#remove padding from flows/bursts, just add the CLS tokens to the dataset, padding token is not needed 
# ['flow_duration', 'burst_tokens', 'directions', 'bytes', 'iats', 'rts', 'protocol', 'labels', 'total_bursts']


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
            f" vocab_size={self.vocab_size}, max_bursts={self.max_bursts}, max_burst_length={self.max_burst_length}, p={self.p})"
        )

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return [self.CLS_TOKEN, self.PAD_TOKEN]

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
    ) -> Tuple[str]:
        return

    def __len__(self):
        return self.vocab_size

    def pad_bursts(
            self,
            flow: list[list[int]],
            max_burst_length: int,
            pad_token: Optional[int] = None
    ) -> np.ndarray:
        """
        Truncate each burst to `max_burst_length` and pad with token if necessary.
        """
        if pad_token is None:
            pad_token = self.PAD_TOKEN
        return np.array([
            burst[:max_burst_length] + [pad_token] * max((max_burst_length - len(burst)), 0)
            for burst in flow
        ])

    def pad_flow(self, flow, max_bursts: int, token: int = None):
        """
        Truncate the flow to `max_bursts` and pad with token if necessary.
        """
        if token is None:
            token = self.PAD_TOKEN

        pad_bursts = max(max_bursts - len(flow), 0)
        pads = [token] * len(flow[0]) * pad_bursts

        flow = list(itertools.chain.from_iterable(flow[:max_bursts]))  # flatten
        flow += pads
        return flow

    @staticmethod
    def prepend_to_list(flow: list[list[int]], token: Optional[int]) -> list[list[int]]:
        # Sometimes we prepend CLS_TOKEN or similar
        if token is not None:
            return [[token] + burst for burst in flow]
        else:
            return [ burst for burst in flow]

    @staticmethod
    def convert_to_tokens(flow: list[list[int]], add_one: bool = False) -> list[list[int]]:
        if not add_one:
            return flow  # noop
        return [[tok + add_one for tok in burst] for burst in flow]

    @staticmethod
    def convert_to_attn(bursts):
        return [[1] * len(burst) for burst in bursts]

    def __call__(self, dataset):
        return self.tokenize(dataset)

    def trunc_flow(self, ls, idxs):
        return [
            ".".join(ls[i].split(".")[:idxs[i]]) + "."
            for i in range(len(ls))
        ]

    @staticmethod
    def _expand_bursts(flows: list[list[int]], burst_sizes: list[list[int]]) -> list[list[list[int]]]:
        """
        To save space, some repetitive info is stored as a single value for the entire burst.
        This function expands the burst sizes to match the actual burst lengths.
        """
        return [
            [
                [value] * burst_sizes[idx][i]
                for i, value in enumerate(flow)
            ]
            for idx, flow in enumerate(flows)
        ]

    def tokenize(self, text, **kwargs):
        try:
            dataset: LazyBatch = text
            # dataset_burst_sizes = [[len(burst) for burst in flow] for flow in dataset["burst_tokens"]]

            # if not self.pretraining and "labels" in dataset:
            #     labels = np.array(dataset["labels"], dtype=int)
            #     if self.p > 0:
            #         num_noise_samples = int(self.p * len(labels))
            #         indices = random.sample(range(0, len(labels) - 1), num_noise_samples)
            #         noisy_labels = np.random.random_integers(
            #             0, 10, size=(num_noise_samples,)  # TODO(maybe-hello-world): refactor 0, 10 to min, max values of labels
            #         )
            #         labels[indices] = noisy_labels
            #     labels = labels.tolist()
            # restore directions: true/false -> 1/-1
            direction = [[1 if direction else -1 for direction in flow] for flow in dataset["directions"]]

            direction = self.tokenize_fields(direction)
            # direction = self.tokenize_fields(self._expand_bursts(direction, dataset_burst_sizes))

            pkt_bytes = self.tokenize_fields(dataset["bytes"])
            iats = self.tokenize_fields(dataset["iats"])
            input_ids, attention_mask = self.tokenize_fields_with_attn(
                dataset["burst_tokens"], prepend_token=self.CLS_TOKEN, add_one=True
            )
            # pkt_count = self.tokenize_fields(self._expand_bursts(dataset["counts"], dataset_burst_sizes))
            # iats = self.tokenize_fields(self._expand_bursts(dataset["iats"], dataset_burst_sizes))
            # input_ids, attention_mask = self.tokenize_fields_with_attn(
            #     dataset["burst_tokens"], prepend_token=self.CLS_TOKEN, add_one=True
            # )
            # total_bursts = [len(flow) for flow in dataset["burst_tokens"]]

            # batchDict = {
            #     "input_ids": input_ids,
            #     "attention_mask": attention_mask,
            #     "direction": direction,
            #     "bytes": pkt_bytes,
            #     "pkt_count": pkt_count,
            #     "iats": iats,
            #     "total_bursts": total_bursts,
            #     "flow_duration": dataset["flow_duration"],
            #     "protocol": dataset["protocol"],
            # }

            # ['flow_duration', 'burst_tokens', 'directions', 'bytes', 'iats', 'rts', 'protocol', 'labels', 'total_bursts']
            # pdb.set_trace()
            all_chunked_input_ids = []
            all_chunked_metadata = []
            for i in range(len(input_ids)):
                chunks, metadata = chunk_with_sliding_window(
                    input_ids[i], dataset["rts"][i],attention_mask[i],direction[i],pkt_bytes[i],iats[i],dataset["flow_duration"][i],dataset["protocol"]
                )
            all_chunked_input_ids.extend(chunks)
            all_chunked_metadata.extend(metadata)
            # pdb.set_trace()
            # Prepare the BatchEncoding with the chunked data
            batchDict = {
                "burst_tokens": [meta["protocol"] for meta in all_chunked_metadata],
                "input_ids": all_chunked_input_ids,
                "attention_mask":[meta["attention_mask"] for meta in all_chunked_metadata],  # Update this as needed based on chunking logic
                "directions": [meta["direction"] for meta in all_chunked_metadata],
                "bytes": [meta["bytes"] for meta in all_chunked_metadata],
                "iats": [meta["iats"] for meta in all_chunked_metadata],
                "rts": [meta["rts"] for meta in all_chunked_metadata],
                "flow_duration": [meta["flow_duration"] for meta in all_chunked_metadata],
                "protocol": [meta["protocol"] for meta in all_chunked_metadata],
                "total_bursts":[meta["total_bursts"] for meta in all_chunked_metadata]
            }
            # pdb.set_trace()
            # batchDict = {
            # "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "direction": direction, #done 
            # "bytes": pkt_bytes, #done
            # "iats": iats, #done 
            # "rts": dataset["rts"], #done 
            # "flow_duration": dataset["flow_duration"], #done 
            # "protocol": dataset["protocol"], #done 
            # }
            # if not self.pretraining and "labels" in dataset:
            #     batchDict.update({"labels": labels})
            # flattened_input_ids = [list(itertools.chain.from_iterable(item)) for item in input_ids]
            return BatchEncoding(batchDict)
        except Exception as e:
            print(f"Tokenizer error: {e}")

    # def tokenize_fields(
    #         self,
    #         dataset: list[list[list[int]]],
    #         prepend_token: int = None,
    #         add_one: bool = False
    # ) -> list[list[list[int]]]:
    #     tokenized_data = [
    #         self.pad_flow(
    #             self.pad_bursts(
    #                 self.prepend_to_list(self.convert_to_tokens(flow, add_one), prepend_token),
    #                 self.max_burst_length,
    #             ),
    #             self.max_bursts,
    #         )
    #         for flow in dataset
    #     ]

        # return tokenized_data
    
    def tokenize_fields(
        self,
        dataset: list[list[list[int]]],
        prepend_token: int = None,
        add_one: bool = False
        ) -> list[list[list[int]]]:
        tokenized_data = [
            self.prepend_to_list(self.convert_to_tokens(flow, add_one), prepend_token)
            for flow in dataset
        ]
        return tokenized_data

    # def tokenize_fields_with_attn(
    #         self,
    #         dataset: list[list[list[int]]],
    #         prepend_token: int = None,
    #         add_one: bool = False
    # ) -> Tuple[list[list[list[int]]], list[list[list[int]]]]:
    #     tokenized_data = self.tokenize_fields(dataset, prepend_token, add_one)
    #     attn = [
    #         self.pad_flow(
    #             self.pad_bursts(
    #                 self.prepend_to_list(self.convert_to_attn(flow), self.ATTN_PRESENCE_TOKEN),
    #                 max_burst_length=self.max_burst_length,
    #                 pad_token=self.ATTN_ABSENCE_TOKEN
    #             ),
    #             max_bursts=self.max_bursts,
    #             token=self.ATTN_ABSENCE_TOKEN
    #         )
    #         for flow in dataset
    #     ]
    #     return tokenized_data, attn

    def tokenize_fields_with_attn(
        self,
        dataset: list[list[list[int]]],
        prepend_token: int = None,
        add_one: bool = False
) -> Tuple[list[list[list[int]]], list[list[list[int]]]]:
        # Tokenize data without padding
        tokenized_data = self.tokenize_fields(dataset, prepend_token, add_one)
        
        # Generate attention masks without padding
        attn = [
            self.prepend_to_list(self.convert_to_attn(flow), self.ATTN_PRESENCE_TOKEN)
            for flow in dataset
        ]
        
        return tokenized_data, attn

def chunk_with_sliding_window(input_ids: list[list[int]], timestamps: list[list[int]],attention_mask:list, direction:list[int],pkt_bytes:list[int],iats:list[int],flow_duration:int,protocol:int,window_size_ms: int = 100, step_size_ms: int = 10, min_packets: int = 12) -> Tuple[list[list[int]], list[dict]]:
    # """
    # Chunk input_ids using a sliding window based on timestamps.

    # Parameters:
    # - input_ids (list[list[int]]): The list of tokenized packets for a flow.
    # - timestamps (list[int]): Corresponding timestamps for each packet in nanoseconds.
    # - window_size_ms (int): Size of the window in milliseconds (default: 100ms).
    # - step_size_ms (int): Step size for the sliding window in milliseconds (default: 10ms).
    # - min_packets (int): Minimum number of packets in a window. Pad if fewer.

    # Returns:
    # - Tuple containing the chunked input_ids and a list of corresponding metadata dictionaries.
    # """
    # Convert timestamps from nanoseconds to milliseconds
    timestamps_ms = [ts / 1e6 for ts in timestamps]

    chunked_input_ids = []
    chunked_metadata = []  # Holds metadata like direction, bytes, iats, etc.

    start_idx = 0
    end_idx = 0
    num_packets = len(timestamps_ms)

    while start_idx < num_packets:
        window_start_time = timestamps_ms[start_idx]
        window_end_time = window_start_time + window_size_ms

        # Find the end index for the current window
        while end_idx < num_packets and timestamps_ms[end_idx] <= window_end_time:
            end_idx += 1

        # Create the chunk
        chunk = input_ids[start_idx:end_idx]
        if len(chunk) < min_packets:
            # Pad the chunk with PAD_TOKEN
            chunk += [[NetFoundTokenizer.PAD_TOKEN] * len(input_ids[0])] * (min_packets - len(chunk))
            attention_mask += [[0] * len(attention_mask[0])]  * (min_packets - len(chunk))
        # Add the chunk and its corresponding metadata
        chunked_input_ids.append(chunk)
        total_bursts=len(chunk)
        chunked_metadata.append({
            "timestamps": timestamps[start_idx:end_idx] + [0] * (min_packets - len(chunk)),  # Pad timestamps if needed
            # Include additional metadata (e.g., direction, bytes, iats) as necessary
            "direction": direction[start_idx:end_idx] + [[0]] * (min_packets - len(chunk)),
            "bytes": pkt_bytes[start_idx:end_idx] + [[0]] * (min_packets - len(chunk)),
            "iats": iats[start_idx:end_idx] + [[0]] * (min_packets - len(chunk)),
            "rts": timestamps[start_idx:end_idx] + [0] * (min_packets - len(chunk)),
            "flow_duration": [flow_duration] * max(len(chunk), min_packets),  # Repeat and pad as needed
            "protocol": [protocol] * max(len(chunk), min_packets),
            "attention_mask": attention_mask,
            "total_bursts":total_bursts
            })

        # Move the window by the step size
        start_idx += min_packets  # Packet-based step
        window_start_time += step_size_ms

    return chunked_input_ids, chunked_metadata
