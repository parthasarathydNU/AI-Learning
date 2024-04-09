import streamlit as st
import random
import time

st.title("Streaming chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# The only difference so far is we've changed the title of our app and 
# added imports for random and time. We'll use random to randomly select 
# a response from a list of responses and time to add a delay to simulate 
# the chatbot "thinking" before responding.

# All that's left to do is add the chatbot's responses within the if block.
# We'll use a list of responses and randomly select one to display. We'll 
# also add a delay to simulate the chatbot "thinking" before responding 
# (or stream its response). Let's make a helper function for this and insert 
# it at the top of our app.
# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

# Above, we've added a placeholder to display the chatbot's response. 
# We've also added a for loop to iterate through the response and display 
# it one word at a time. We've added a delay of 0.05 seconds between each 
# word to simulate the chatbot "thinking" before responding. Finally, we 
# append the chatbot's response to the chat history. As you've probably guessed, 
# this is a naive implementation of streaming. 
