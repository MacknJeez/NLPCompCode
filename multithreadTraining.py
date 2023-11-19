import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import threading

# Define a function to load and preprocess dataset in a separate thread
def load_and_preprocess_dataset():
    global train_dataset
    global data_collator
    global train_file

    train_file = "trainingdataset.txt"
    with open(train_file, 'r') as file1:

        lines = file1.readlines()
        indexFormal = [i for i in range(0,407,3)]
        indexCasual = [j for j in range(1,408,3)]
        arrForInd = np.array(lines)
        arrCasInd = np.array(lines)

        formal_sentences = arrForInd[indexFormal]
        informal_sentences = arrCasInd[indexCasual]

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=512  #adjust
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )


# Define model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)

# Create a separate thread to load and preprocess the dataset
data_thread = threading.Thread(target=load_and_preprocess_dataset)
data_thread.start()
data_thread.join()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
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
print("DONE TRAINING DONE TRAINING DONE TRAINING DONE TRAINING DONE TRAINING ")

# Save the model
model.save_pretrained("./informal_model")
tokenizer.save_pretrained("./informal_model")

print("DONE SAVING DONE SAVING DONE SAVING DONE SAVING DONE SAVING DONE SAVING ")

print("")