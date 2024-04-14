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

# url = "https://docs.smith.langchain.com/overview"
url = "https://raw.githubusercontent.com/parthasarathydNU/humanitarians-OPT-fork/main/PROMPT_ENGINEERING_AND_AI/Prompt%20Engineering%20for%20Generative%20AI/Readme.md"

loader = WebBaseLoader(url)
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=4)

# question = "how can langsmith help with testing?"
question = "What are the topics covered under the introduction to generative AI slide decks"
docs = retriever.invoke(question)

# Retrieval chain ends here ===================================

#  Retrieval Augmented generation ============================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(question)

output = document_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
        "context": docs,
    }
)

print(output)

# RAG Chain ends here 
