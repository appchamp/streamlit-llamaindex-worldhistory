import os
import streamlit as st
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI

index_name = "./saved_index_wh"
documents_folder = "./data/world-history-txt"

@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    if os.path.exists(index_name):
        st.status("Found existing index on disk.", state="complete")

        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        state = st.status("No exiting index found.. creating index..", expanded=True, state="running")
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        state.update("Index creation DONE !", expanded=False, state="complete")

        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine().query(query_text)
    return str(response)


st.title("🦙 World History - LlamaIndex 🦙")
st.header("Welcome to World History.")
st.write(
    "Your query will be answered using the essay as context, using embeddings from text-ada-002 and LLM completions from gpt-3.5-turbo."
)

index = None

if st.secrets.has_key("openai_key"):
    api_key = st.secrets.openai_key
    st.status("Using existing API Key from st.secrets", state="complete")
else:
    api_key = st.text_input(
        label="#### Your OpenAI API key 👇",
        placeholder="Paste your openAI API key, sk-",
        type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    index = initialize_index(index_name, documents_folder)

if index is None:
    st.warning("Please enter your api key first.")

text = st.text_input("Query text:", value="What do you know about Canton factory?")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}"
        )

    with embed_col:
        st.markdown(
            f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
        )
