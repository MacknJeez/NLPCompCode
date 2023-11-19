from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
import torch

# Load the fine-tuned model and tokenizer
model_name = "./informal_model"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Set the model to evaluation mode
model.eval()

# Define a function to generate informal response
def generate_informal(formal_sentence):
    input_ids = tokenizer.encode(formal_sentence, return_tensors="pt")

    # Generate informal response
    with torch.no_grad():   
        outputs = model.generate(input_ids, 
                                attention_mask=input_ids.new_ones(input_ids.shape), 
                                max_length=70, #change according to size of input sentence
                                num_return_sequences=1, #was 1
                                no_repeat_ngram_size=3, #was 2
                                top_k=50, #was 50
                                #top_p=0.95, #was 0.95
                                #temperature=0.2, #was 0.7
                                #do_sample=True,
                                pad_token_id=tokenizer.eos_token_id)
        
        # testing greedy method, commented this out to change type of text gen
        #greedy_output = model.generate(input_ids, max_new_tokens = 10)
    informal_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return informal_response


# Test the model with multiple formal sentences and corresponding reference informal responses

formal_sentences = ["In our annual financial report, we observe a consistent increase in quarterly revenue, with a notable 12% growth in the last quarter."]
#formal_sentences = ["The study of climate change requires a multi-disciplinary approach, encompassing meteorology, environmental science, and data analysis."]

#formal_sentences = ["The parties to this agreement hereby consent to the exclusive jurisdiction of the courts in the state of Dubai for any disputes arising from this contract."]



for i, formal_sentence in enumerate(formal_sentences):
    informal_response = generate_informal(formal_sentence)
    #print(f"Informal: \n {informal_response}")

file2 = open("outputfile.txt", "w+")
file2.write(str(informal_response))
#file2.write("\n BLEU score:", calculate_bleu(formal_sentences,informal_response))
file2.close()


'''for i, formal_sentence in enumerate(formal_sentences):
    print(f"Formal: {formal_sentence}")
    
    #Generate informal response
    informal_response = generate_informal(formal_sentence)
    print(f"Informal: {informal_response}")
    
    #Evaluate BLEU score
    bleu_score = calculate_bleu(reference_responses[i], informal_response)
    print(f"BLEU Score: {bleu_score:.2f}\n")'''
