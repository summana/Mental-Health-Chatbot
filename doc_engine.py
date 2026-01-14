# import os
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.llms.openai import OpenAI as LlamaOpenAI
# llama_llm = LlamaOpenAI(model="gpt-3.5-turbo",api_key=os.getenv("OPENAI_API_KEY"))
# documents = SimpleDirectoryReader("data").load_data()
# index= VectorStoreIndex.from_documents(documents)

# query_engine=index.as_query_engine(llm = llama_llm)

# def query_documents(user_query: str)-> str:
#     return str(query_engine.query(user_query))
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure local embeddings (FREE - no API calls!)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Configure LLM for querying
llama_llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

# Load documents and create index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(llm=llama_llm)


def query_documents(user_query: str) -> str:
    """Query documents using the index"""
    return str(query_engine.query(user_query))