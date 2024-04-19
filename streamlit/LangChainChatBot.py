from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)


# Simple prompt 1 ========================
# output = chat.invoke(
#     [
#         HumanMessage(
#             content="Translate this sentence from English to French: I love programming."
#         )
#     ]
# )
# print(output)

# Simple prompt 2 =======================
# output = chat.invoke(
#     [
#         HumanMessage(content="What did you just say?"),
#     ]
# )
# print(output)

# Combined Prompt 1 and 2
# output = chat.invoke(
#     [
#         HumanMessage(
#             content="Translate this sentence from English to French: I love programming."
#         ),
#         AIMessage(content="J'adore programmer."),
#         HumanMessage(content="What did you just say?"),
#     ]
# )
# print(output)

# Testing out chains

# Here we try giving the chatbot some instructions before it begins answering your prompts
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful japanese assistant. Answer all questions in japanese.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
# chain = prompt | chat
# output = chain.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Translate this sentence from English to French: I love programming."
#             ),
#             AIMessage(content="J'adore la programmation."),
#             HumanMessage(content="What did you just say?"),
#         ],
#     }
# )
# print(output)

# Testing retrievers
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from typing import Dict
from langchain_core.runnables import RunnablePassthrough

# The questions we will be testing for our RAG
# question = "how can langsmith help with testing?"
question = "What are the topics covered under the introduction to generative AI slide decks"

# url = "https://docs.smith.langchain.com/overview"
url = "https://raw.githubusercontent.com/parthasarathydNU/humanitarians-OPT-fork/main/PROMPT_ENGINEERING_AND_AI/Prompt%20Engineering%20for%20Generative%20AI/Readme.md"

# Fetching data from the url
loader = WebBaseLoader(url)
data = loader.load()

# Splitting data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Creating an in-memory vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=4)

docs = retriever.invoke(question)
# # Retrieval chain ends here ===================================

# #  Retrieval Augmented generation ============================
# Defining a pre processor for our chatbot to interact with the user
# This is more like prompting the chatbot to behave in a certain way
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Creating a document chain that now takes this pre processor and appends it 
# as a subsequent step for our chat API 
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# Mocking some chat history
demo_ephemeral_chat_history = ChatMessageHistory()

# Adding the mock question to the chat history
demo_ephemeral_chat_history.add_user_message(question)

# Testing out generation based on retrieved data (docs)
# output = document_chain.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#         "context": docs,
#     }
# )

# print(output)

# RAG Chain ends here 

# We have only done on retrieval till now, but we want to do this as a repetitive process
# Where for each question that the user asks we first fetch information and then pass it to the 
# chatbot as context 


# We now define a larger chain that has the following componente
"""
The retrieval chain uses the RunnablePass through can be seen as a function that 
takes in some input and passes it through a couple of other chains before 
passing it out to the output chain

Our retriever should retrieve information relevant to the last message we 
pass in from the user, so we extract it and use that as input to fetch relevant docs,
which we add to the current chain as context. We pass context plus the previous 
messages into our document chain to generate a final answer.

Here in the below example we see the following happen: 
1. In line 172, we pass in the array of messages
2. The execurtion jumps to line 165 where first the `parse_retriever_input` 
runs and gets the last message form the array which is the user question
3. Then it is passed to the retriever / invokes the retriever with the latest question
4. The output of which is then passed to the `document_chain` which takes the retrieved data,
the question and then generates a response

"""

def parse_retriever_input(params: Dict):
    # retrieving the latest message that we pass in from the user
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context= parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

# response = retrieval_chain.invoke(
#     {
#         # here the latest message will be the question that was asked
#         "messages": demo_ephemeral_chat_history.messages, 
#     }
# )

# print(response)

retrieval_chain_2 = (
    RunnablePassthrough.assign( context= parse_retriever_input | retriever ) | 
    document_chain
)

response2 = retrieval_chain_2.invoke(
    {
        # here the latest message will be the question that was asked
        "messages": demo_ephemeral_chat_history.messages, 
    }
)

print(response2)
