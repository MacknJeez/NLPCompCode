import nltk 

from nltk.translate.bleu_score import sentence_bleu

candidate = "In our annual financial report, we observe a consistent increase in quarterly revenue, with a notable 12% growth in the last quarter. We believe that our growth strategy is well suited to meet the needs of our customers and provide a competitive edge in the digital"
reference = "In our annual financial report, we observe a consistent increase in quarterly revenue, with a notable 12% growth in the last quarter."
#["The study of climate change requires a multi-disciplinary approach, encompassing meteorology, environmental science, and data analysis."]
#["In our annual financial report, we observe a consistent increase in quarterly revenue, with a notable 12 growth in the last quarter."]
#["The parties to this agreement hereby consent to the exclusive jurisdiction of the courts in the state of Dubai for any disputes arising from this contract."]


candidate_tokenized = nltk.word_tokenize(candidate.lower())
reference_tokenized = nltk.word_tokenize(reference.lower())
bleu_score = sentence_bleu([reference_tokenized], candidate_tokenized)

print(f"The BLEU score is: {bleu_score}")























#for block size 256, 2 epochs
#climatechange 1.7987047645554095e-78
#legaldoc 1.3947352471139056e-78
#finance 1.1935742466208903e-231


#3 epochs, 8 perdevicebatchsize, 256 block size