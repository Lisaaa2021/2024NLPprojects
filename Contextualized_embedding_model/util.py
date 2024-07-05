import pandas as pd
import nltk
nltk.download('punkt')
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def output_two_corpus(variable_file_path = 'Variable_under_divided_government_or_NOT.csv'):
    """
    This function takes a variable_file_path parameter, which contains variable(s) that will be used to group the DEC corpus.
    The file 'st_df.pk1' contains a pre-processed dataframe of the DEC corpus.
    Output: two divided corpus lists with each containing instances at the sentence level for the next step.
    """
     
    df_divi = pd.read_csv(variable_file_path).rename(columns = {'Document ID':'file'})
    #st_df pickle file contains an organized dataframe of DEC corpus
    df_text = pd.read_pickle("st_df.pkl")[['file','text']]
    df_join = pd.merge(df_divi, df_text, on='file', how='inner')

    # applying dividing criteria
    txt_1 = "\n".join(df_join[df_join['DivGov_MajorityProject']==1]['text'])
    txt_2 = '\n'.join(df_join[df_join['DivGov_MajorityProject']==0]['text'])
    #df_divi['DivGov_MajorityProject']
    #(1) under divided government
    #(2) under a democratic president.
    
    corpus_1 = nltk.sent_tokenize(txt_1)
    corpus_2 = nltk.sent_tokenize(txt_2)
    
    return corpus_1, corpus_2



# file_path = "output_1.txt"

# # Open the file in write mode and save the string
# with open(file_path, 'w') as file:
#     file.write(txt_1)
# print("String saved to", file_path)

# file_path = "output_2.txt"
# # Open the file in write mode and save the string
# with open(file_path, 'w') as file:
#     file.write(txt_2)
# print("String saved to", file_path)


def get_keyword_corpus(corpus_1, corpus_2, keyword):
    """ 
    Take two corpus and the keyword as input
    Output two corpus list with sentences containing the keyword
    """
    keyword_corpus_1 = []
    keyword_corpus_2 = []
    for line in corpus_1:
      if keyword in line:
        # Do something with the line, for example, print it
        keyword_corpus_1.append(line.strip())

    for line in corpus_2:
      if keyword in line:
        # Do something with the line, for example, print it
        keyword_corpus_2.append(line.strip())
    return keyword_corpus_1, keyword_corpus_2
    
    

def bert_text_preparation(text):
  '''Preprocesses text input in a way that BERT can interpret'''
  marked_text = '[CLS] ' + text + ' [SEP]'
  tokenized_text = tokenizer.tokenize(marked_text)
  #print('tokenized_text: ', tokenized_text)

  # id of the tokens
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  #print('indexed_tokens: ', indexed_tokens)

  # ?
  segments_ids = [1]*len(indexed_tokens)
  #print('segments_ids: ', segments_ids)

  #convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  #print('tokens_tensor', tokens_tensor)

  segments_tensor = torch.tensor([segments_ids])
  #print('segments_tensor', segments_tensor)


  return tokenized_text, tokens_tensor, segments_tensor



def get_bert_embeddings(tokens_tensor, segments_tensor):
  """Obtains BERT embeddings for tokens"""
  # gradient calculation id disabled
  with torch.no_grad():
    # obtain hidden states
    outputs = model(tokens_tensor, segments_tensor)
    #print(dir(outputs)) #transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    hidden_states = outputs[2]
    #Base class for model’s outputs that also contains a pooling of the last hidden states.
  # concatenate the tensors for all layers
  # use 'stack' to create new dimension in tensor
  token_embeddings = torch.stack(hidden_states, dim = 0)
  # remove dimension 1, the 'batches'
  token_embeddings = torch.squeeze(token_embeddings, dim = 1)
  # swap dimensions 0 and 1 so we can loop over tokens
  token_embeddings_swap = token_embeddings.permute(1,0,2)
  # innitilized list to store embeddings
  token_vecs_sum = []

  # 'token_embeddings' is a [Y * 12 * 768 tensor]
  # where Y is the number of tokens in the sentence

  # loop over tokens in sentence
  for token in token_embeddings_swap:
    sum_vec = torch.sum(token[-4:], dim = 0)
    token_vecs_sum.append(sum_vec)

  return token_vecs_sum


#get_bert_embeddings(tokens_tensor,segments_tensor,model)

def get_contextulized_embedding(corpus, keyword):

  keyword_embeddings = []

  for i,sentence in enumerate(corpus):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence)
    
    if len(segments_tensors[0]) < 513: # skip long sentences
      list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors)
      
      keyword_token_split = tokenizer.tokenize(keyword) # get a list of tokens that the keyword is split into
      keyword_split_len = len(keyword_token_split)

      for ind, token in enumerate(tokenized_text[1:-1]):
        if token == keyword_token_split[0]:
          end_ind = ind + keyword_split_len
          if tokenized_text[1:-1][ind:end_ind] == keyword_token_split: # two if statements to locate the keyword position index 
            stack = list_token_embeddings[1:-1][ind: end_ind] # get a list of embeddings of x tokens
            embedding = sum(stack) / keyword_split_len # compute the average embeddings for that instance
            keyword_embeddings.append(embedding) #append the instance to the final list


  return keyword_embeddings