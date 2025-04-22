#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

from flask import Flask, request
from flask_cors import CORS, cross_origin


import numpy as np
import pandas as pd


from transformers import AutoTokenizer, AutoModel
import torch



from langchain.text_splitter import  TokenTextSplitter


from openai import OpenAI

##########################################################
## GEMINI
##########################################################

GEMINI_KEY = 'YOUR-GEMINI-TOKEN-HERE'

client = OpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    n=1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            #### TYPE YOUR PROMPT HERE ####
            "content": "What is the capital city of Thailand" 
        }
    ]
)

print(response.choices[0].message.content)


##########################################################
## VECTOR MODEL
##########################################################

#model_name = "BAAI/bge-m3"
model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


##########################################################
## This is formatted as code
##########################################################

def get_llm_response(user_content, sys_content='You are a helpful assistant'):

  client = OpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
  )

  response = client.chat.completions.create(
      model="gemini-2.0-flash",
      n=1,
      messages=[
          {"role": "system", "content": sys_content},
          {
              "role": "user",
              "content": user_content 
          }
      ]
  )

  return response.choices[0].message.content


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)




def get_prompt(query, context):
  template = '''
    You are an AI model assistant.
    You are an expert at answering questions about requirment gathering.
    Answer the question based ONLY on the following context.

    {context}

    Original question: {question}"""
  '''
  return template.format(question=query, context=context)


##########################################################
## LOAD DATA
##########################################################

data='''
Requirement Statement for Online Barber System:
Appointment Booking and Management: The system must include a queue booking function that allows users to select their preferred date and time for barber services. It must display available time slots in real-time to ensure convenient and efficient booking. The system must support automatic appointment cancellation or rescheduling, with notifications sent to both the barber and the user whenever changes occur.
Barber Search and Selection: Users must be able to search for nearby barbers and barber shops using GPS-based location services. They should also be able to select barbers based on their specialties and customer review ratings. The system must display detailed barber profiles, including portfolio images, work history, and services offered, to help users make well-informed decisions.
Payment and Transaction Management: The system must support various online payment methods, such as credit/debit cards, bank transfers, or payment applications, for user convenience. It should store payment history and provide users with receipts after successful transactions. Additionally, the system must include features to help barbers manage their income and review their transactions transparently.
'''


text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=5)

chunks = text_splitter.split_text(data.replace('\n', ' '))



##########################################################
## Add to Vector DB
##########################################################


vecdb = pd.DataFrame({'text':chunks})
vecdb['vec'] = vecdb['text'].apply(lambda x: get_embedding(x))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def similarity_search(query):
    qvec = get_embedding(query)
    vecdb['sim'] = vecdb['vec'].apply(lambda x: cosine_similarity(x, qvec) )
    return vecdb.sort_values('sim', ascending=False).iloc[:5]['text'].tolist()
    

##########################################################
## RAG Function
##########################################################


def ask_my_rag(query):
  res_doc = similarity_search(query)
  res_text = '; '.join(res_doc)
  return get_llm_response(get_prompt(query, res_text))


ask_my_rag("what are users of the systems")


##########################################################
## RAG Function
##########################################################


def to_chat(user_content):

  client = OpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
  )

  response = client.chat.completions.create(
      model="gemini-2.0-flash",
      n=1,
      messages=[
          {"role": "system", "content": 'You are a helpful assistant'},
          {
              "role": "user",
              "content": user_content 
          }
      ]
  )

  return response

#############################################
#############################################
######             API                #######
#############################################
#############################################


app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/v1/chat/completions', methods=["POST"])
@cross_origin()
def chat_completions():
    last_query = request.json['messages'][-1]['content']
    print(last_query)
    prompt = ask_my_rag(last_query)
    print(prompt)
    res = to_chat(prompt)
    return res.json()




if __name__ == '__main__':
    app.run()



