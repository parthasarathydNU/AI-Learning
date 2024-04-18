import streamlit as st
from openai import OpenAI

st.title("Rubric Generation ChatBot")

flipped_interaction_prompt = "You are an expert in rubric generation for any given type of assignment. Once a user " \
                             "submits an assignment, use the flipped interaction pattern to ask the user questions " \
                             "about their grading preferences, which areas of the assignment that they want greater " \
                             "emphasis on. The conversation should be engaging to the user. The questions can be " \
                             "regarding: Their style of grading , how strict do they want to be and other questions " \
                             "to arrive at a well defined and clear grading schema without any ambiguity. Further ask " \
                             "questions regarding the user to understand more about their personal as well. Finally " \
                             "based on the gathered preferences, use the persona pattern to take the persona of the " \
                             "user and generate a rubric that matches their style. Start by greeting the user and ask " \
                             "one question at a time. Ask the first question about what is the type of assignment " \
                             "they want help with."


# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # The "with" key word here is used to sort of get hold of the
    # context of the chat message element of the type defined by the
    # role of the message
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize has_rubric if not already initialized
if "has_rubric" not in st.session_state:
    st.session_state["has_rubric"] = "None"

st.session_state.messages.append({"role": "assistant", "content": "Do you already have a rubric for the assignment?"})
has_rubric_input = st.selectbox("Do you have an existing rubric?", ["Select", "Yes", "No"])

if has_rubric_input == "Yes":
    st.session_state["has_rubric"] = "Yes"
else:
    st.session_state["has_rubric"] = "No"

if has_rubric_input == "Yes" or has_rubric_input == "No":
    # Prompt OpenAI API with provided prompt
    response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "assistant", "content": flipped_interaction_prompt}
            ],
        )

    # Extract the assistant's response from the API response
    assistant_response = response.choices[0].message.content.strip()

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)


# Accept user input
if prompt := st.chat_input("What do you need help with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Trigger the following block whenever there is a prompt to be processed
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})


