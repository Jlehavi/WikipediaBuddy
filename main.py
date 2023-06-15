from dotenv import load_dotenv
import os

import streamlit as st
from streamlit_chat import message

import wikipediaapi

from serpapi import GoogleSearch

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

#Set up necessary items

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

serpKey = os.getenv("serpApiKey")

embeddings = OpenAIEmbeddings()
langWiki = WikipediaAPIWrapper()
wiki = wikipediaapi.Wikipedia('en')

st.set_page_config(
    page_title="Wikipedia Learning Buddy"
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi! What topic do you want to learn about today?"]

if 'past' not in st.session_state:
    st.session_state['past'] = []

@st.cache_resource()
def loadModel():
    model = OpenAI(temperature=0.2)
    return model
llm = loadModel()

@st.cache_resource()
def loadChatModel(name):
   model = ChatOpenAI(model_name=name, temperature=0.2)
   return model
chat = loadChatModel("gpt-3.5-turbo")



def findSearch(input):
    urlTemplate = """
    I want you to act as an expert in wikipedia searches. Given the following request, determine what the user is trying to find on wikipedia.
    For context, the user is trying to find a specific topic on wikipedia. Once you determine what the user is trying to find, generate
    the perfect wikipedia search to find the topic's wikipedia page. Only return the search you have generated. The request is {topic}
    """
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=urlTemplate
    )

    prompt.format(topic=input)
    searchChain = LLMChain(llm=llm, prompt=prompt)
    search = searchChain.run(input)
    return search

def confirmSearch(titles):
    confirmText = """
    I've found mutliple results from my wikipedia search. Which of these titles are the one you're looking for? Repond with the number associated with the title you're choosing.
    """

    count = 0
    for title in titles:
        confirmText += str(count) + " : " + title + "\n"
        count+=1

    st.session_state['generated'].append(confirmText)

def getWikiPage(title):
    page = wiki.page(title)
    return page


def handleTitle(title):
    page = getWikiPage(title)
    text = page.text

    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = textSplitter.split_text(text)
    db = FAISS.from_texts(docs, embeddings)

    st.session_state['db'] = db

def generateResponse(query):
    db = st.session_state['db']
    text = db.similarity_search(query)
    fullText = " ".join(t.page_content for t in text)

    sysTemplate = """
    I want you to act as an expert and teacher in {title}. You are to help teach the information in the text provided to you to the user.
    You are also going to answer any questions that the user has based on the text provided to you. If they ask a question, you will answer
    their question in a verbose and detailed way. After answering, you will teach the topic to the user to help them understand further.
    Only user factual information you find within the text to answer your question.

    The text you will base your response off of is the following: {text}
    """

    humanTemplate = """
    I want you to answer the following question: {query}
    """

    systemPrompt = SystemMessagePromptTemplate.from_template(sysTemplate)
    humanPrompt = HumanMessagePromptTemplate.from_template(humanTemplate)

    chatPrompt = ChatPromptTemplate.from_messages(
        [systemPrompt, humanPrompt]
    )

    chain = LLMChain(llm=chat, prompt=chatPrompt)

    response = chain.run(
        title=st.session_state['titles'][0],
        text=fullText,
        query=query
    )

    return response
    #Retriever method for similarity search and generating response
    #retriever = db.as_retriever()
    #qa = RetrievalQA.from_chain_type(chat, chain_type="stuff", retriever=retriever)
    #return qa.run(query)

def setUp():
    st.session_state['titles'] = None

def getInput():
    input = st.text_input("Talk to me here")
    return input

userInput = getInput()

if userInput:
    st.session_state['past'].append(userInput)

    if len(st.session_state['past']) == 1:
        #If first user input, then find wikipedia page they are talking about
        search = findSearch(userInput)
        summaries = langWiki.run(search)
        pages = summaries.split("\n")
        
        titles = []
        #for page in pages:
        title = pages[0][6 :]
        titles.append(title)
        
        st.session_state['titles'] = titles

        if len(titles) > 1:
            confirmSearch(titles)
        else:
            handleTitle(title)
            st.session_state['generated'].append("What would you like to know about " + title)

    elif userInput.isdigit():
        #if not first user input and is a number, then the user is picking their title out of the list of titles
        if int(userInput) > len(st.session_state['titles'])-1 or int(userInput) < 0:
            st.session_state['generated'].append("That number doesn't seem to correlate with a title. Try and input a different number.")
        else:
            title = st.session_state['titles'][int(userInput)]
            st.session_state['generated'].append("What would you like to know about " + title)
            handleTitle(title)
    
    else:
        #have to make sure that a title was chosen and the user is not skipping the step
        response = generateResponse(userInput)
        st.session_state['generated'].append(response)


#Create message system
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        if i < len(st.session_state['past']):
            message(st.session_state['past'][i], is_user=True, key = str(i) + ' _user')