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
Smart City System
Citizen Role: In the Smart City system, the citizen plays a central role as the primary user and beneficiary of city services. Citizens must be able to register and log into the system securely, maintaining control over their personal profiles and preferences. The platform should allow them to access a wide range of public services such as requesting waste collection, reporting damaged infrastructure, or alerting authorities about local disturbances. A tracking mechanism must be in place so citizens can follow the progress of their service requests and incident reports in real time. Additionally, citizens should be empowered to provide feedback on city services, participate in surveys, and contribute opinions on city planning or initiatives. The system should also act as an information hub, giving citizens access to announcements, emergency alerts, public events, and traffic updates to keep them informed and engaged.
Officer Role: Officers serve as the operational backbone of the Smart City system, responsible for handling and responding to citizen needs. They must be able to log into the system using role-based access credentials that define their scope of authority and responsibilities. Once logged in, officers should have access to a dashboard that lists tasks, complaints, and service requests assigned to them. They must be able to update the status of these requests, add resolution notes, and communicate with the reporting citizen if necessary. Officers should also be equipped with tools to send timely notifications—such as scheduled maintenance or safety alerts—to relevant communities. Moreover, the system should provide officers with analytical insights, helping them monitor department efficiency, identify recurring issues, and optimize service delivery.
Mayor Role: The mayor holds an administrative and strategic oversight role within the Smart City system. The platform should provide the mayor with access to comprehensive, city-wide dashboards that present real-time analytics on service performance, citizen satisfaction, and operational efficiency. From this high-level view, the mayor can identify patterns, monitor progress on public initiatives, and make informed policy decisions. The mayor must also be able to approve or review plans submitted by officers, initiate new programs, and allocate resources based on system-generated insights. Additionally, the mayor plays a vital role in public communication—making important announcements, launching awareness campaigns, and addressing select citizen feedback to foster transparency and trust within the community.
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



