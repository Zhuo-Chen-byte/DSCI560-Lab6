import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain import HuggingFacePipeline
from langchain.llms import LlamaCpp

from dotenv import load_dotenv
load_dotenv()

import os
openai_api_key = os.getenv('OPENAI_API_KEY')


def get_pdf_text(pdf_docs):
    text = ''
    
    for pdf_doc in pdf_docs:
        text += ''.join(page.extract_text() for page in PdfReader(pdf_doc).pages)
        
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore, model_choice):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    if model_choice == 'GPT3.5':
        llm = ChatOpenAI()
    elif model_choice == 'Llama':
        llm = LlamaCpp(
            model_path='models/llama-2-7b-chat.ggmlv3.q4_1.bin', n_ctx=1024, n_batch=512
        )
    else:
        raise ValueError("Invalid model choice. Please select either 'GPT3.5' or 'Llama'.")

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4}),
        memory=memory,
    )
    
    return conversation_chain


def handle_userinput(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print(f'User: {message.content}')
        else:
            print(f'Bot: {message.content}')


def main():
    pdf_docs = []
    
    while True:
        pdf_path = input("Enter the path to a PDF file (or 'done' to finish): ")
        
        if pdf_path.lower() == 'done':
            break
        if os.path.exists(pdf_path):
            pdf_docs.append(pdf_path)
        else:
            print('File not found. Please provide a valid path.')

    if not pdf_docs:
        print('No PDFs provided. Exiting.')
        
        return

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    model_choice = input('Choose the model to process the PDFs with (GPT3.5/Llama): ')
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore, model_choice)

    while True:
        user_question = input("Ask questions about your documents (or 'exit' to quit): ")
        
        if user_question.lower() == 'exit':
            break
            
        handle_userinput(user_question, conversation_chain)


if __name__ == '__main__':
    main()
