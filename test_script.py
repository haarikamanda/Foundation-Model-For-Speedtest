import torch
from NetFoundDataCollator import DataCollatorWithSlidingWindowAndMasking

# Sample data mimicking the expected input structure
sample_examples = [
    {
        "burst_tokens": [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ],
        "rts": [0, 50, 120]  # Timestamps in microseconds
    },
    {
        "burst_tokens": [
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35],
        ],
        "rts": [0, 110, 250]  # Timestamps in microseconds
    }
]

# Instantiate the collator with your desired settings
collator = DataCollatorWithSlidingWindowAndMasking(
    window_ms=100, 
    step_ms=10, 
    packet_step=2, 
    min_packets=4, 
    padding_token=0, 
    mlm=True  # Assuming you want masking enabled
)

# Call the collator on the sample data
batch = collator.torch_call(sample_examples)

# Print the outputs to inspect the results
print("Input IDs:")
print(batch["input_ids"])
print("Attention Masks:")
print(batch["attention_masks"])
print("Labels:")
print(batch["labels"])
