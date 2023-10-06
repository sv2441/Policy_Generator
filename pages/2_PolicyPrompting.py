import streamlit as st
import docx2txt
import pandas as pd
import base64
import csv
import math
import docx
import os
from langchain.output_parsers import OutputFixingParser
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize chat model
chat_llm = ChatOpenAI(temperature=0.0)

st.title("Policy Prompting")


def generation(user_input,text):
    if os.path.exists('Policy_Document.doc'):
        doc = docx.Document('Policy_Document.doc')
    else:
        doc = docx.Document()
    title_template = user_input
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(topic=text)
    response = chat_llm(messages)
    content = str(response.content)
    doc.add_paragraph(content)
    doc.save('Policy.doc')
    with open('Policy.doc', 'rb') as f:
        doc_data = f.read()
    b64 = base64.b64encode(doc_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="policy.doc">Download Result</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    
    
    
    
    
    
# Function to convert DOC to Text
def convert_doc_to_text(doc_file):
    text = docx2txt.process(doc_file)
    return text

# Upload a DOC file
st.sidebar.header("Upload a Policy file")
doc_file = st.sidebar.file_uploader("Choose a DOC file", type=[".doc", ".docx"])

# Main content area
st.subheader("Policy Document")
if doc_file is not None:
    # Convert the DOC to text
    text = convert_doc_to_text(doc_file)
    st.write(text)
else:
    st.write("Upload a DOC file to convert it to text.")

# Text area for user input
st.subheader("User Prompt")
user_input = st.text_area("Enter your Prompt here:")

# Save user input to a file
if st.button("Submit"):
    generation(user_input,text)
    


