#  Design and implement an LLM-powered chatbot using LangChain

Reference Link: https://python.langchain.com/docs/use_cases/chatbots/quickstart/

Reference Python file: [LangChainChatBot.py](../LangChainChatBot.py)

## Overview
We’ll go over an example of how to design and implement an LLM-powered chatbot. Here are a few of the high-level components we’ll be working with:

- Chat Models. The chatbot interface is based around messages rather than raw text, and therefore is best suited to Chat Models rather than text LLMs. You can use LLMs for chatbots as well, but chat models have a more conversational tone and natively support a message interface.
- Prompt Templates, which simplify the process of assembling prompts that combine default messages, user input, chat history, and (optionally) additional retrieved context.
- Chat History, which allows a chatbot to “remember” past interactions and take them into account when responding to followup questions.
- Retrievers (optional), which are useful if you want to build a chatbot that can use domain-specific, up-to-date knowledge as context to augment its responses. - we will add this at a later stage in this process

### Packages to install 
- langchain 
- langchain-openai 
- langchain-chroma

## Basic Elements 

To first check out how the langchain chat works, we try a simple chat invocation: 
```python
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

output = chat.invoke(
    [
        HumanMessage(
            content="Translate this sentence from English to French: I love programming."
        )
    ]
)

print(output)
```

We see the following output:
```json
{
  "content": "J'adore programmer.",
  "response_metadata": {
    "token_usage": {
      "completion_tokens": 6,
      "prompt_tokens": 19,
      "total_tokens": 25
    },
    "model_name": "gpt-3.5-turbo-1106",
    "system_fingerprint": "fp_77a673219d",
    "finish_reason": "stop",
    "logprobs": "None"
  },
  "id": "run-72c6461c-ba26-4bba-93e3-a161df32ee3f-0"
}
```

But in this app we still have not enabled the memory management aspect, so if we ask a follow up question, like "What did you just say?" this is the outcome that we get to see.

```
content='I apologize if there was any confusion. Is there something specific you would like me to clarify or repeat?' response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 13, 'total_tokens': 34}, 'model_name': 'gpt-3.5-turbo-1106', 'system_fingerprint': 'fp_77a673219d', 'finish_reason': 'stop', 'logprobs': None} id='run-33c74ab7-67f3-49be-870a-e47855c4c51a-0'
```

We clearly see that the chatbot does not remember the prev sentence. So to solve this we send the messages in sequence as follows:
```python
output = chat.invoke(
    [
        HumanMessage(
            content="Translate this sentence from English to French: I love programming."
        ),
        AIMessage(content="J'adore programmer."),
        HumanMessage(content="What did you just say?"),
    ]
)
```

This is the output:
```
content='I said "J\'adore programmer," which means "I love programming" in French.' response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 39, 'total_tokens': 58}, 'model_name': 'gpt-3.5-turbo-1106', 'system_fingerprint': 'fp_77a673219d', 'finish_reason': 'stop', 'logprobs': None} id='run-f77347c9-436e-470a-856e-3020e07f0a2c-0'
```

Here we see that , the response has context of the previous message. This brings our chat application to life.

## Chains

Chains are components that help us perform actions in sequence. For example in the below use case, we want the chatbot so speak in japanese, so we give it the instruction to act as a japanese chatbot. Here is where the way we prompt the chatbot makes a difference in how it responds. We run the command `chain.invoke` rather than `chat.invoke`

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful japanese assistant. Answer all questions in japanese.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | chat
output = chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Translate this sentence from English to French: I love programming."
            ),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ],
    }
)
print(output)
```

Here is the output:
```
content='申し訳ありません、私は日本語でお手伝いをするAIです。英語からフランス語への翻訳を提供しました。
```

## Retrievers 

While this chain can serve as a useful chatbot on its own with just the model’s internal knowledge, it’s often useful to introduce some form of retrieval-augmented generation, or RAG for short, over domain-specific knowledge to make our chatbot more focused. 

We can set up and use a Retriever to pull domain-specific knowledge for our chatbot. To show this, let’s expand the simple chatbot we created above to be able to answer questions about LangSmith.

We set up a simple web based retriever: 
```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=4)
docs = retriever.invoke("how can langsmith help with testing?")
print(docs)
```

And here is the output:
```
[
    Document(
        page_content='Getting started with LangSmith | ğŸ¦œï¸�ğŸ›\xa0ï¸� LangSmith',
        metadata={
            'description': 'Introduction', 
            'language': 'en', 
            'source': 'https://docs.smith.langchain.com/overview', 
            'title': 'Getting started with LangSmith | ğŸ¦œï¸�ğŸ›\xa0ï¸� LangSmith'
            }
    ), 
    Document(
        page_content='Skip to main contentLangSmith API DocsSearchGo to AppQuick StartUser GuideTracingEvaluationProduction Monitoring & AutomationsPrompt HubProxyPricingSelf-HostingCookbookQuick StartOn this pageGetting started with LangSmithIntroductionâ€‹LangSmith is a platform for building production-grade LLM applications. It allows you to closely monitor and evaluate your application, so you can ship quickly and with confidence. Use of LangChain is not necessary - LangSmith works on its own!Install', 
        metadata={'description': 'Introduction', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'Getting started with LangSmith | ğŸ¦œï¸�ğŸ›\xa0ï¸� LangSmith'}
    ), 
    Document(
        page_content='reach out to us at support@langchain.dev.My team deals with sensitive data that cannot be logged. How can I ensure that only my team can access it?â€‹If you are interested in a private deployment of LangSmith or if you need to self-host, please reach out to us at sales@langchain.dev. Self-host....
    
    Document ....
]
```

The above content can then be fed into the chat bot for better context to answer questions regarding to the prompt. This can be added within the chain. 

## RAG

We can take this one step further, now that we have extracted relevant ttext from our saved content, we can now pass this as a context to our chatbot for it to be able to answer our questions better based on the retrieved content

```python
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
```

We used the following link: https://raw.githubusercontent.com/parthasarathydNU/humanitarians-OPT-fork/main/PROMPT_ENGINEERING_AND_AI/Prompt%20Engineering%20for%20Generative%20AI/Readme.md

The question that was asked: "What are the topics covered under the introduction to generative AI slide decks"

Output: 
```
The topics covered under the introduction to generative AI slide decks are:
- Generative AI: Foundations and Use Cases
- Before Transformers: Evolution of Text Generation
- Deep Dive into Transformer Architecture
- Generating Text with Transformers
- Lifecycle of a Generative AI Project
- Quiz questions
- References

You can find more details and resources in the slide decks provided in the link: [Link To Learn about Gen AI](https://docs.google.com/document/d/1FZ3jLeNEbSGRlNr41X-SYt_TS6v1QFg8pK6zSGnbDak/edit)
```

We see this is exactly able to retrieve relevant content from the document that we have passed!!

## Integrating retrieval into the conversation chain

Now instead of having a separate step for retrieving and injecting the context into the next message, we create a chain that does it as one of the intermediate steps. This is a powerful think, because we can fetch relevant context for all of the user's messages and then pass it as context along with all the previous messages in the interaction for the LLM to be able to generate a contextualized response. Here the amount of data available to generate plays a role in the accuracy and usefulness of information available for the LLM to generate answers. 


The code looks like this : 
```python
def parse_retriever_input(params: Dict):
    # retrieving the latest message that we pass in from the user
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context= parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

response = retrieval_chain.invoke(
    {
        # here the latest message will be the question that was asked
        "messages": demo_ephemeral_chat_history.messages, 
    }
)

print(response)
```

And this is how the output looks like : 
```
{
    'messages': [
        HumanMessage(
            content='What are the topics covered under the introduction to generative AI slide decks'
            )
    ], 
    'context': [
        Document(
            page_content='### **References and Further Reading**..../Readme.md'
            }
        )
    ], 
    'answer': 'The introduction to generative AI slide decks cover the following topics:\n\n1. Generative AI: Foundations and Use Cases\n2. Before Transformers: Evolution of Text Generation\n3. Deep Dive into Transformer Architecture\n4. Generating Text with Transformers\n5. Lifecycle of a Generative AI Project\n6. Quiz questions\n7. References\n\nYou can find more details and resources on these topics in the slide decks provided in the project planning.'
    
}
```

By looking at the above output we can understand how this works. As per code, the retrieval chain is invoked on the message, but internally it has been assigned a RunnablePassthrough which also takes in the message and assigns it to the retriever chain and then passes it to the document_chain. 

The retriever_chain is the one that takes in the prompt and fetches relevant data form the in memory database, and the document chain is the one that has been given the instructions on what to do with the retrieved data. Here it has been instructed to do the following: "Answer the user's questions based on the below context:\n\n{context}",

### Skip Printing the intermetidate steps

To avoid printing the intermediate steps in the retrieval process, we can use a slightly different of the way wer define the RunnablePassthrough.

Instead of explicitly calling out the assign key word, we can directly use a "|" operator to chain the `document_chain` along with the passthrough to just see the output. 

```python
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
```

Output: 
```
The topics covered under the introduction to generative AI slide decks are:
- Generative AI: Foundations and Use Cases
- Before Transformers: Evolution of Text Generation
- Deep Dive into Transformer Architecture
- Generating Text with Transformers
- Lifecycle of a Generative AI Project
- Quiz questions
- References

You can find more details and resources on these topics in the slide decks provided in the project planning.
````
