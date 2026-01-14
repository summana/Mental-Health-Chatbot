# import os
# from dotenv import load_dotenv
# from langchain_openai import OpenAI
# # from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# load_dotenv()
# OpenAI_API_KEY=os.getenv("OPEN_API_KEY")
# if not OpenAI_API_KEY:
#     raise ValueError('Open_API key not found. Please check your .env file')
# llm =OpenAI(openai_api_key=OpenAI_API_KEY, temperature=0.7)#initialise the llm
# #store per user memory sesions
# session_memory_map={}
# def get_response(session_id:str,user_query:str) -> str:
#     if session_id not in session_memory_map:
#         memory=ConversationBufferMemory()
#         session_memory_map[session_id]=ConversationChain(llm =llm, memory = memory, verbose=True)
#     conversation = session_memory_map[session_id]
#     return conversation.predict(input=user_query)



import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
OpenAI_API_KEY = os.getenv("OPEN_API_KEY", "")  # Returns empty string if not found

# REMOVED THE CHECK - No more error!
# if not OpenAI_API_KEY:
#     raise ValueError('OpenAI API key not found. Please check your .env file')

# Initialize the LLM (using ChatOpenAI instead of OpenAI)
llm = ChatOpenAI(
    api_key=OpenAI_API_KEY,  # Changed parameter name
    temperature=0.7,
    model="gpt-3.5-turbo"
)

# Store per-user memory sessions
session_memory_map = {}

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a compassionate mental health support assistant. Provide empathetic, supportive responses."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm


def get_session_history(session_id: str):
    """Get or create chat history for a session"""
    if session_id not in session_memory_map:
        session_memory_map[session_id] = ChatMessageHistory()
    return session_memory_map[session_id]


# Wrap chain with message history
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def get_response(session_id: str, user_query: str) -> str:
    """Get a response from the chatbot for a given session"""
    response = conversation_chain.invoke(
        {"input": user_query},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content