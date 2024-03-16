# import nltk
# from nltk.corpus import wordnet  # Library for synonyms and word relationships
# from nltk.tokenize import word_tokenize  # Tokenize sentences into words
# nltk.download('wordnet')
# nltk.download('punkt')

# def paraphrase_to_formal(sentence):
#   """
#   This function paraphrases a sentence to a more formal tone.

#   Args:
#       sentence: The input sentence as a string.

#   Returns:
#       A paraphrased sentence in formal tone, or the original sentence if no improvement is found.
#   """
#   formal_words = []
#   tokens = word_tokenize(sentence.lower())
#   for token in tokens:
#     # Find synonyms for informal words using WordNet
#     synonyms = wordnet.synsets(token)
#     for synset in synonyms:
#       # Check for synonyms with higher formality score (1.0 formal, 0.0 informal)
#       if synset.definition().split(";")[0].split(":")[0].strip() == "formality":
#         formal_score = float(synset.definition().split(";")[0].split(":")[1].strip())
#         if formal_score > 0.7:  # Consider synonyms with a formality score above 0.7
#           formal_words.append(synset.lemmas()[0].name())
#           break  # Move to next word after finding a formal synonym
#     else:
#       formal_words.append(token)
#   return " ".join(formal_words).capitalize()  # Join words and capitalize the first letter


# # Example usage
# sentence = "not good yes"
# formal_sentence = paraphrase_to_formal(sentence)
# print(f"Original: {sentence}")
# print(f"Formal paraphrase: {formal_sentence}")





























# from nltk.corpus import wordnet

# def paraphrase(sentence):
#   """
#   This function attempts to paraphrase a sentence using synonyms and rephrasing.

#   Args:
#       sentence: The sentence to paraphrase (string)

#   Returns:
#       A paraphrased sentence (string) or the original sentence if no paraphrase found.
#   """
#   # Tokenize the sentence
#   words = sentence.lower().split()
#   synonyms = []
#   # Find synonyms for each word
#   for word in words:
#     synsets = wordnet.synsets(word)
#     if synsets:
#       # Get the first synonym
#       first_synonym = synsets[0].lemmas()[0].name()
#       # Avoid replacing the original word with itself
#       if first_synonym != word:
#         synonyms.append(first_synonym)
#       else:
#         synonyms.append(word)
#     else:
#       synonyms.append(word)

#   # Try rephrasing with synonyms
#   paraphrased_sentence = " ".join(synonyms)

#   # Check if the paraphrase is actually different
#   if paraphrased_sentence != sentence:
#     return paraphrased_sentence
#   else:
#     return sentence

# # Example usage
# user_input = input("Enter a sentence: ")
# paraphrased = paraphrase(user_input)
# print("Original sentence:", user_input)
# print("Paraphrased sentence:", paraphrased)








from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import language_tool_python

# Load the pre-trained Pegasus paraphrasing model and tokenizer
model_name = 'tuner007/pegasus_paraphrase'
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)


# Initialize the grammar correction tool
tool = language_tool_python.LanguageTool('en-UK')

def paraphrase_and_correct(sentence, num_return_sequences=1):
    # Tokenize the text
    tokens = tokenizer(sentence, truncation=True, padding='longest', return_tensors='pt')
    
    # Generate paraphrased output
    outputs = model.generate(**tokens, num_beams=5, num_return_sequences=num_return_sequences)
    
    # Decode the generated paraphrases
    paraphrased_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Correct the paraphrased sentences
    corrected_sentences = [tool.correct(paraphrased_sentence) for paraphrased_sentence in paraphrased_sentences]
    
    return corrected_sentences

# Example usage
input_sentence = "borther is doing well."
corrected_paraphrases = paraphrase_and_correct(input_sentence)
print(corrected_paraphrases)





























# import torch

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
# model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')

# sentence = "This is something which i cannot understand at all"

# text =  "paraphrase: " + sentence + " </s>"

# encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
# input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


# outputs = model.generate(
#     input_ids=input_ids, attention_mask=attention_masks,
#     max_length=256,
#     do_sample=True,
#     top_k=120,
#     top_p=0.95,
#     early_stopping=True,
#     num_return_sequences=5
# )

# for output in outputs:
#     line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#     print(line)
