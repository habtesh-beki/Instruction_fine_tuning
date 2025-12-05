import json, random

with open("downloaded_data.json", "r") as file:
    data = json.load(file)   # this is a LIST of dicts

Test_Rate = 0.04
Validation_Rate = 0.04  

random.shuffle(data)

n = len(data)
n_test = int(n * Test_Rate)
n_val = int(n * Validation_Rate)
n_train = n - n_test - n_val

Training_Data   = data[:n_train]
Validation_Data = data[n_train : n_train + n_val]
Test_Data       = data[n_train + n_val :]

def convert_to_alpaca_format(data):
    instruction = data["instruction"]
    input_text  = data.get("input", "")
    output      = data["output"]

    # if input is empty, skip the section
    if input_text.strip():
        input_section = f"\n\n### Input:\n{input_text}"
    else:
        input_section = ""

    formatted = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}"
        f"{input_section}\n\n"
        f"### Response:\n{output}"
    )

    return formatted

train_formatted = [convert_to_alpaca_format(item) for item in Training_Data]
val_formatted   = [convert_to_alpaca_format(item) for item in Validation_Data]
test_formatted  = [convert_to_alpaca_format(item) for item in Test_Data]


def save_jsonl(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(json.dumps({"text": x}, ensure_ascii=False) + "\n")

save_jsonl("train.jsonl", train_formatted)
save_jsonl("val.jsonl", val_formatted)
save_jsonl("test.jsonl", test_formatted)
