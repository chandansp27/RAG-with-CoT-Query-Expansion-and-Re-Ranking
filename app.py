import os
import streamlit as st
import tempfile
from utils import FILE_TYPES, FILE_STORAGE_DIR, EMBEDDING_MODEL_NAME
import shutil
import functions
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore # type: ignore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # type: ignore
from llama_index.core import StorageContext
import logging
import sys
import utils

# logging settings for LLM
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# title and wide mode settings
st.set_page_config(layout="wide")
st.title('RAG-Chat')


# create storage directory for storage of uploaded files
if not os.path.exists(FILE_STORAGE_DIR):
    os.makedirs(FILE_STORAGE_DIR)


# upload file widget
uploaded_file = st.sidebar.file_uploader(
                        label='Upload files',
                        type=FILE_TYPES,
                        accept_multiple_files=False, 
                        disabled=False,
                        label_visibility="collapsed",
                        )


# session state for files
if 'file_data' not in st.session_state:
    st.session_state.file_data = False

if uploaded_file is not None and not st.session_state.file_data:
    if uploaded_file.name.lower().endswith('.pdf'):
        with tempfile.NamedTemporaryFile(dir=FILE_STORAGE_DIR, delete=False, suffix='.txt', mode='w') as temp_file, st.status(
        "Loading file...",
        expanded=False,
        state='running'
    ):
            pdf_content = functions.processPdfFile([uploaded_file])
            temp_file.write(pdf_content)
    else:
        with tempfile.NamedTemporaryFile(dir=FILE_STORAGE_DIR, delete=False) as temp_file, st.status(
            "Loading file...",
            expanded=False,
            state='running'
        ):
            temp_file.write(uploaded_file.getbuffer())
        st.write("File uploaded successfully!")
    st.session_state.file_data = True


# cache files processed by llama-index
@st.cache_data
def loadDocuments(file_storage_dir):
    try:
        documents = SimpleDirectoryReader(file_storage_dir).load_data()
        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)
        return (documents, nodes)
    except Exception as error:
        return f'{error}'

# cache embedding model throughout the session
@st.cache_resource
def loadEmbeddingModel(model_name):
    try:
        embedding_model = HuggingFaceEmbedding(model_name=model_name)
        return embedding_model
    except Exception as error:
        return f'{error}'

# cache files embeddings throughout the chat
@st.cache_resource
def _embedDocs(_documents, _embed_model):
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection('embedding_docs')
    except Exception:
        pass
    chroma_collection = chroma_client.create_collection("embedding_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        _documents, storage_context=storage_context, embed_model=_embed_model, embed_batch_size=100
    )
    return index


# load files, and create vector db if files uploaded by user
if uploaded_file:
    embedding_model = loadEmbeddingModel(EMBEDDING_MODEL_NAME)
    loaded_data = loadDocuments(FILE_STORAGE_DIR)
    if loaded_data:
        documents, nodes = loaded_data[0], loaded_data[1]
        index = _embedDocs(documents, embedding_model)
    else:
        st.write('Document processing failed')

    # chat functionality
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            st.markdown(message['content'])
        
    if prompt := st.chat_input("Enter your query"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        cot_nodes = functions.retrieveNodes(prompt, nodes, utils.K_RETRIEVE)
        best_cot_nodes = functions.rerankNodes(cot_nodes, utils.K_RERANK)

        # clean nodes, provide only text to the LLM
        cot_context = '\n'.join([node.text for node in best_cot_nodes])
        task1 = functions.chainOfThoughtPrompt(prompt)
        
        # request chain of thought answer using the provided context for query expansion
        cot_request = functions.getResponse(task1, cot_context)
        cot_cleaned = functions.cleanResponce(task1, prompt, cot_request)

        # retrieve nodes using expanded cot-query
        qa_nodes = functions.retrieveNodes(f'{prompt} {cot_cleaned}', nodes, utils.K_RETRIEVE)
        rerrank_qa_nodes = functions.rerankNodes(qa_nodes, utils.K_RERANK)
        
        qa_context = '\n'.join([node.text for node in rerrank_qa_nodes])
        task2 = functions.QAGenerationPrompt(prompt)
        qa_request = functions.getResponse(task2, qa_context)

        qa_cleaned = functions.cleanResponce(task2, prompt, qa_request)
        with st.chat_message('assistant'):
            with st.popover('View CoT'):
                st.markdown(f'{cot_cleaned}')
            st.write_stream(functions.textToGenerator(qa_cleaned))
        st.session_state.messages.append({'role': 'assistant', 'content': qa_cleaned})
else:
    st.markdown(""" ### RAG with :red[Query-expansion] and :blue[CoT] """)
    st.markdown(""" #### ***LLM: :green[Mistral-7b-Instruct-v2]*** ####""")
    st.markdown(""" #### Select a file to get started #### """)
    
    # reset chat, when the user clicks on 'x', clears session data, cache and temporary file storage directory
    def reset_chat(FILE_STORAGE_DIR):
        st.session_state.messages = []
        st.session_state.context = None
        shutil.rmtree(FILE_STORAGE_DIR) 
        os.makedirs('uploaded_files')
        try:
            loadDocuments.clear()
            _embedDocs.clear()
        except Exception as e:
            pass
    st.session_state.file_data = False
    reset_chat(FILE_STORAGE_DIR)
