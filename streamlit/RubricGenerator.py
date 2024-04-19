import streamlit as st
from openai import OpenAI

st.title("Rubric Generation ChatBot")

flipped_interaction_prompt = "You are an expert in rubric generation for any given type of assignment. Once a user " \
                             "submits an assignment, use the flipped interaction pattern to ask the user questions " \
                             "about their grading preferences, which areas of the assignment that they want greater " \
                             "emphasis on.  The conversation should be engaging to the user. The questions can be " \
                             "regarding:  Their style of grading , how strict do they want to be and other questions " \
                             "to arrive at a well defined and clear grading schema without any ambiguity. Further ask " \
                             "questions regarding the user to understand more about their personal as well. Finally " \
                             "based on the gathered preferences, use the persona pattern to take the persona of the " \
                             "user and generate a rubric that matches their style. Start by greeting the user and ask " \
                             "one question at a time. "


# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "intro_message_displayed" not in st.session_state:
    st.session_state["intro_message_displayed"] = None

st.session_state.messages.append({"role": "system", "content": flipped_interaction_prompt})


if st.session_state["intro_message_displayed"] is None:
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm here to help you craft a detailed "
                                                                      "rubric for your assignment. To get started, "
                                                                      "could you tell me if you already have the "
                                                                      "rubric?"})
    st.session_state["intro_message_displayed"] = False


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # Check if the message is the introductory message and it hasn't been displayed yet
    if message["role"] != "system" and not st.session_state["intro_message_displayed"]:
        # Display the introductory message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        # Set the flag to True to indicate that the introductory message has been displayed
        st.session_state["intro_message_displayed"] = True
    elif message["role"] != "system":  # Exclude messages with role "system"
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


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


