# inference.py

from unsloth import FastLanguageModel
import torch

# Load your trained LoRA weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",    # folder containing trained weights
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Same Alpaca-style prompt used during training
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

while True:
    instruction = input("\nEnter your instruction (or 'exit'): ")

    if instruction.lower() == "exit":
        break

    input_text = ""       # OR ask the user for this too
    output_text = ""      # leave blank for generation

    prompt = alpaca_prompt.format(instruction, input_text, output_text)

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)

    print("\nModel Response:\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

