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

# one document only
index_name = "../saved_index_wh_1"
documents_folder = "../docs/world-history-txt-1"

@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # use a local model instead
    # service_context = ServiceContext.from_defaults(embed_model="local")

    if os.path.exists(index_name):
        st.status("Found existing index on disk.", state="complete")

        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        with st.status("No exiting index found.. creating index..", expanded=True, state="running") as status:
            documents = SimpleDirectoryReader(documents_folder).load_data()
            status.update(label=f"Reading {len(documents)} file(s)..", state="running")
            index = GPTVectorStoreIndex.from_documents(
                documents, service_context=service_context, show_progress=True
            )
            status.update(label="Index creation DONE !", expanded=False, state="complete")
        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine(verbose='True').query(query_text)
    return str(response)


st.title("World History - LlamaIndex 🦙")
st.header("Welcome to World History")
st.caption('''Using [1 doc](https://apcentral.collegeboard.org/media/pdf/ap22-cr-report-world-history-modern.pdf "ap22-cr-report-world-history-modern.pdf"):''')

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


st.markdown('''
            Try one of these queries (or your own):  
            - Who owned the diamond empire?  
            - what do you know about the incan empire?
        ''')
text = st.text_input("Query text:", value="Who is Biran?")

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
