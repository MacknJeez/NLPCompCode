from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./informal_model")
tokenizer = GPT2Tokenizer.from_pretrained("./informal_model")

# Define a formal sentence
formal_sentence = "Title: The Role of Renewable Energy in a Sustainable Future Renewable energy sources, such as wind and solar power, are pivotal for achieving a sustainable future. Transitioning to clean energy reduces carbon emissions, mitigates climate change, and promotes environmental stewardship."


# Tokenize and generate informal text
input_ids = tokenizer.encode(formal_sentence, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # Create an attention mask
output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens = 150, num_return_sequences=1, do_sample=True)

# Decode the output to get informal text
informal_text = tokenizer.decode(output[0], skip_special_tokens=True)
file2 = open("outputfile.txt", "w+")

file2.write(str(informal_text))
file2.close()

ref = formal_sentence.split()
test = informal_text.split()
print('BLEU score -> {}'.format(sentence_bleu(ref, test)))
      


print("done generating")