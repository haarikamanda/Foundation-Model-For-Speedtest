import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from typing import Any, Dict, List, Optional, Union
import numpy as np
import random
from transformers.utils import requires_backends, is_torch_device
from utils import get_logger
import pdb
logger = get_logger(name=__name__)


class DataCollatorWithMeta(DataCollatorForLanguageModeling):
    def __init__(self, values_clip: Optional[int] = None, swap_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_clip = values_clip
        self.swap_rate = swap_rate

    def pad_and_convert_for_key(self,batch, key, pad_token=0):
        """
        Pads the values for a specific key in the batch to the maximum length and converts to a tensor.
        
        Args:
            batch (dict): The batch dictionary.
            key (str): The key to process.
            pad_token (int): The token to use for padding. Default is 0.
        
        Returns:
            torch.Tensor: Padded tensor for the specified key.
        """
        if key not in batch:
            raise KeyError(f"The key '{key}' is not present in the batch.")

        # Determine the max length for the specific key
        max_length = max(len(item) for item in batch[key])

        # Pad the sequences for the specific key
        padded_key = [
            item + [pad_token] * (max_length - len(item)) for item in batch[key]
        ]

        # Convert to PyTorch tensor
        tensor_key = torch.tensor(padded_key)

    

        return tensor_key
    def torch_call(
            self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {}
        burstsInEachFlow = [example["total_bursts"] for example in examples]
        maxBursts = max(burstsInEachFlow)
        # pdb.set_trace()
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            for key in inputs.keys():
                if key == "labels"  or key == "replacedAfter":
                    continue
                if key not in batch:
                    if key != "replacedAfter":
                        batch[key] = []
                if key == "ports":
                    batch[key].append(inputs[key] + 1)
                elif key in ("protocol", "flow_duration","total_bursts"):
                    batch[key].append(inputs[key])
                else:
                    batch[key].append(
                        inputs[key][: maxBursts * self.tokenizer.max_burst_length]
                    )
        for key in batch.keys():
            if key=="total_bursts":
                batch[key] = torch.Tensor(batch[key])
                continue
            if key=="protocol":
                continue
            if key=="input_ids" or key == "attention_mask":
                # batch[key] = [[item for sublist in x for item in sublist] if isinstance(x[0], list) else x for x in batch[key]]
                batch[key] = [[item for sublist in x for item in sublist] for x in batch[key]]

            batch[key] = self.pad_and_convert_for_key(batch, key, pad_token=0)
           
            # if (
            #         key == "input_ids"
            #         or key == "attention_masks"
            #         or key == "ports"
            #         or key == "protocol"
            # ):
            #     pdb.set_trace()
            #     batch[key] = torch.Tensor(batch[key]).to(torch.long)

        if self.mlm:
            burstsInEachFlow_n=[x*19 for x in burstsInEachFlow]
            batch["input_ids"], batch["labels"], batch["swappedLabels"], batch[
                "burstMetasToBeMasked"] = self.torch_mask_tokens(
                batch["input_ids"], burstsInEachFlow, self.tokenizer.max_burst_length, self.swap_rate,
                batch["protocol"], special_tokens_mask=None
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
       
        del batch['protocol']
        batch['total_bursts']=batch['total_bursts']*19
        # pdb.set_trace()
        return BatchEncoding(batch)

    def swap_bursts_adjust_prob_matrix(self, input_ids, burstsInEachFlow, max_burst_length, swap_rate):
        labels = torch.from_numpy(np.array(np.random.rand(len(burstsInEachFlow)) < swap_rate, dtype=int))
        swappedIds = []
        for i in range(input_ids.shape[0]):
            if labels[i] == 1:
                burstToRep = random.randint(0, burstsInEachFlow[i] - 1)
                flowChoice = random.randint(0, input_ids.shape[0] - 1)
                if flowChoice == i:
                    flowChoice = (flowChoice + 1) % input_ids.shape[0]
                burstChoice = random.randint(0, burstsInEachFlow[flowChoice] - 1)
                swappedIds.append([i, burstToRep])
                input_ids[i][burstToRep * max_burst_length:(burstToRep + 1) * max_burst_length] = input_ids[flowChoice][burstChoice * max_burst_length:(burstChoice + 1) * max_burst_length]
        return input_ids, swappedIds, labels

    def maskMetaData(self, input_ids, burstsInEachFlow, swapped_bursts):
        maskedMetaBursts = np.full((input_ids.shape[0], max(burstsInEachFlow)), 0.3)
        for ids in swapped_bursts:
            maskedMetaBursts[ids[0]][ids[1]] = 0
        candidateFlows = np.array(
            [np.array(np.array(burstsInEachFlow) > 3, dtype=int)]).transpose()  # converting to nX1 matrix
        return torch.bernoulli(torch.from_numpy(candidateFlows * maskedMetaBursts)).bool()

    def torch_mask_tokens(self, input_ids, burstsInEachFlow, max_burst_length, swap_rate, protos, **kwargs):
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # pdb.set_trace()

        ###NEW PART 
        fields_to_exclude=[0,1,2,4,12,13,14,15,16,17,18]
        exclude_mask = torch.zeros_like(probability_matrix, dtype=torch.bool)
        for field in fields_to_exclude:
            exclude_mask[:, field::19] = True  # Assuming each 19 tokens correspond to a packet
        probability_matrix.masked_fill_(exclude_mask, value=0.0)
        ###NEW PART 
        # pdb.set_trace()
        new_ip_ids, swappedIds, swappedLabels = self.swap_bursts_adjust_prob_matrix(input_ids, burstsInEachFlow,
                                                                                    max_burst_length, swap_rate)
        maskMetaData = self.maskMetaData(input_ids, burstsInEachFlow, swappedIds)
        for ids in swappedIds:
            probability_matrix[ids[0]][ids[1] * max_burst_length:(ids[1]) * max_burst_length] = 0
        input_ids = new_ip_ids

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels, swappedLabels, maskMetaData


class DataCollatorForFlowClassification:
    label_names: Dict

    def __init__(self, max_burst_length):
        self.max_burst_length = max_burst_length

    def __call__(self, examples):
        first = examples[0]
        maxBursts = max([int(example["total_bursts"]) for example in examples])
        for i in range(len(examples)):
            if "stats" in examples[i]:
                examples[i]["stats"] = [
                    float(t) for t in examples[i]["stats"].split(" ")
                ]
        batch = {}
        if "labels" in first and first["labels"] is not None:
            label = (
                first["labels"].item()
                if isinstance(first["labels"], torch.Tensor)
                else first["labels"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["labels"] for f in examples], dtype=dtype)
        if "protocol" in first and first["protocol"] is not None:
            label = (
                first["protocol"].item()
                if isinstance(first["protocol"], torch.Tensor)
                else first["protocol"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["protocol"] = torch.tensor([f["protocol"] for f in examples], dtype=dtype)
        if "flow_duration" in first and first["flow_duration"] is not None:
            label = (
                first["flow_duration"].item()
                if isinstance(first["flow_duration"], torch.Tensor)
                else first["flow_duration"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["flow_duration"] = torch.tensor([f["flow_duration"] for f in examples], dtype=dtype)
        for k, v in first.items():
            if (
                    k not in ("labels", "label_ids", "total_bursts", "protocol", "flow_duration")
                    and v is not None
                    and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
                else:
                    batch[k] = torch.tensor(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
        return batch


# TODO(maybe-hello-world): not public
class MyBatchEncoding(BatchEncoding):
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            tmp = {}
            for k, v in self.data.items():
                if type(v) == list:
                    # If the values is a list of tensors, keeping them as is. The errors should be handled by the user.
                    logger.warning(f"Attempting to cast {k} to {str(device)}, this is a list, so keeping them as is")
                    tmp[k] = v
                else:
                    tmp[k] = v.to(device=device)
            self.data = tmp
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self


# TODO(maybe-hello-world): not public
class DataCollatorForSublflows(DataCollatorForLanguageModeling):
    def __init__(self, values_clip: Optional[int] = None, swap_rate=0.5, subflow_bursts=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_clip = values_clip
        self.swap_rate = swap_rate
        self.keys_to_clip = {
            "iats",
            "bytes",
        }
        self.subflow_bursts = subflow_bursts

    @staticmethod
    def _clip_values(values: list[Union[int, float]], clip: Union[int, float]) -> list:
        if clip is not None:
            values = np.clip(values, -clip, clip)
        return values

    def torch_call(
            self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {}
        burstsInEachFlow = [int(example["total_bursts"]) for example in examples]
        maxBursts = max(burstsInEachFlow)
        batch["labels"] = []
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            batch["labels"].append([sum(np.array(inputs["pkt_count"][burst * self.tokenizer.max_burst_length: (
                                                                                                                      burst + self.subflow_bursts) * self.tokenizer.max_burst_length: self.tokenizer.max_burst_length],
                                                 dtype=int)) for burst in
                                    range(self.subflow_bursts, maxBursts - self.subflow_bursts)])
            for key in inputs.keys():
                if key == "labels" or key == "total_bursts":
                    continue
                if key not in batch:
                    if key != "replacedAfter":
                        batch[key] = []
                if key == "ports":
                    batch[key].append(inputs[key] + 1)
                elif key == "protocol":
                    batch[key].append(inputs[key])
                elif key in self.keys_to_clip and self.values_clip is not None:
                    batch[key].append(
                        self._clip_values(
                            inputs[key][: maxBursts * self.tokenizer.max_burst_length],
                            self.values_clip,
                        )
                    )
                else:
                    batch[key].append(
                        inputs[key][: maxBursts * self.tokenizer.max_burst_length]
                    )
        for key in batch.keys():
            if (key == "fileName"):
                continue
            batch[key] = torch.Tensor(np.array(batch[key]))
            if (
                    key == "input_ids"
                    or key == "attention_masks"
                    or key == "ports"
                    or key == "protocol"
            ):
                batch[key] = torch.Tensor(batch[key]).to(torch.long)
        return MyBatchEncoding(batch)