from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import numpy as np
# Define model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load and preprocess dataset
train_file = "trainingdataset2.txt"

# Read the dataset and split it into formal and informal sentences
file1 = open(train_file, 'r')
lines = file1.readlines()

formalLines = []
casualLines = []
# Just some math to ignore the blank spaces inbetween pairs
if len(lines) >= 2:
    for i in range(0, len(lines), 2):
        formalLines.append(lines[i-1].strip())
        casualLines.append(lines[i].strip())
else:
    print("Not enough lines in the dataset.")

# Tokenize the formal sentences
formal_inputs = tokenizer(formalLines, return_tensors="pt", padding=True, truncation=True)

# Tokenize the casual sentences
casual_outputs = tokenizer(casualLines, return_tensors="pt", padding=True, truncation=True)

inputs = formal_inputs
outputs = casual_outputs

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=256,
    overwrite_cache=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3, #3 gave me the best output so far
    per_device_train_batch_size=8, #was 8, 0.0737 score, 10 goes back down to basically 0
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)


# Train the model
trainer.train()
print("done training")

# Save the model
model.save_pretrained("./informal_model")
tokenizer.save_pretrained("./informal_model")
