import os 
from langchain.llms import OpenAI
import streamlit as st
#importing pdf loader 
from langchain.document_loaders import PyPDFLoader
#import chroma as vector store
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (create_vectorstore_agent,VectorStoreToolkit,VectorStoreInfo)


os.environ['OPENAI_API_KEY'] = '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'


#creating instance of openai llm
llm = OpenAI(temperature=0.9)

#create and load pdf loader
loader = PyPDFLoader('InvestmentBanking.pdf')

#split pages from pdf 
pages = loader.split_pages()
#load document into vector database chroma
store = Chroma.from_documents(pages,collection_name='InvestmentBanking')

#create vector store info object metadata repo

vectorStore_Info = VectorStoreInfo(
    name='Investment_Banking',
    description='an investement banking pdf',
    vectorstore=store

)
#convert the vector store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorStore_Info)


#add the toolkilkit to an end to end LC
agent = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
    )

prompt = st.text_input('write your prompt here')

if prompt:
#passing the prompt to the llm
    response = llm(prompt)

    #swap out the raw llm for a document agent
    response = agent.run(prompt)
#write it in the screen
    st.write(response)

    #with streamlit expander

    with st.expander('See the document'):

        #find the relevent pages
        search = store.similarity_search_with_score(prompt)
        #write out the first
        st.write(search[0][0].page_content)