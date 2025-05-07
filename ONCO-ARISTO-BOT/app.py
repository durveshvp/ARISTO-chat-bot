import os
import streamlit as st
import tempfile
# import hashlib # Not actively used, can be removed if no other use case
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader # DirectoryLoader not explicitly used for initial load
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms.huggingface_hub import HuggingFaceHub # Replaced by Gemini
from langchain_google_genai import ChatGoogleGenerativeAI # Import Gemini
from dotenv import load_dotenv
import re
import uuid
import glob

# --- Configuration ---
st.set_page_config(layout="wide")

# PDF directory and index paths
PDF_DIR = "pdf"  # Directory containing pre-loaded documents
INDEX_DIR = "index"  # Directory to store the index
PRELOADED_INDEX_PATH = os.path.join(INDEX_DIR, "preloaded_index")  # Path for preloaded documents

# --- Environment Variable Loading ---
load_dotenv()
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") # Still needed if you use HF for other things, but not for the LLM here
# MODEL_ID = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3") # Not used for Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🔴 GOOGLE_API_KEY not found in environment variables! Please set it in your .env file.")
    st.stop()


# --- Custom CSS ---
st.markdown("""
    <style>
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #2c3e50;
        color: white !important;
        border-left: 5px solid #3498db;
    }
    .bot-message {
        background-color: #e6f3ff;
        color: #000000 !important;
        border-left: 5px solid #2ecc71;
    }
    .user-message b, .user-message p, .user-message span {
        color: white !important;
    }
    .bot-message b, .bot-message p, .bot-message span, .bot-message div {
        color: #000000 !important;
    }
    .st-expander {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Create necessary directories ---
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Model Initialization with Streamlit Caching ---
@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace sentence transformer embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm_model():
    """Loads the Google Gemini LLM."""
    # MODEL_NAME for Gemini can be e.g. "gemini-pro", "gemini-1.5-flash-latest"
    # Check Google AI documentation for the latest and most suitable models.
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, convert_system_message_to_human=True)

embedding_model = load_embedding_model()
llm = load_llm_model()

if embedding_model is None or llm is None:
    st.error("🔴 Failed to load models. Please check your API keys and model configurations.")
    st.stop()

# --- Title ---
st.markdown("""
    <h1 style='text-align: center;'>📄 Chat with Documents using FAISS + Gemini</h1>
""", unsafe_allow_html=True)


# --- Helper functions ---
def extract_answer_only(text):
    """Extracts the answer part from the LLM's response if prefixed."""
    # This regex looks for "Answer:" (case-insensitive) and captures everything after it.
    # It handles multi-line answers due to re.DOTALL.
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If "Answer:" prefix is not found, return the original text,
    # as Gemini might not always use this prefix.
    return text.strip()


def get_session_index_path(session_id):
    """Generates a unique path for session-specific FAISS indexes."""
    return os.path.join(INDEX_DIR, f"session_{session_id}_index")

def load_document_from_path(file_path, file_type):
    """Loads documents from a given file path based on file type."""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type in ["docx", "doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    else:
        st.error(f"Unsupported file type: {file_type} for file {file_path}")
        return [] # Return empty list on error

def get_documents_in_directory(directory_path):
    """Get list of PDF and Word documents in the specified directory."""
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    doc_files = glob.glob(os.path.join(directory_path, "*.doc")) + glob.glob(os.path.join(directory_path, "*.docx"))
    return pdf_files + doc_files

def process_directory_documents(directory_to_scan, _embedding_model):
    """
    Processes all documents in the specified directory, creates embeddings,
    and saves a FAISS index.
    """
    source_documents = get_documents_in_directory(directory_to_scan)
    if not source_documents:
        return None, 0

    all_chunks = []
    files_processed_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_path in enumerate(source_documents):
        status_text.text(f"Processing file: {os.path.basename(file_path)} ({i+1}/{len(source_documents)})")
        try:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            loaded_docs = load_document_from_path(file_path, file_type)

            if not loaded_docs:
                st.warning(f"Could not load document: {file_path}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(loaded_docs)

            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['source'] = os.path.basename(file_path)

            all_chunks.extend(chunks)
            files_processed_count += 1
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
        progress_bar.progress((i + 1) / len(source_documents))

    status_text.text("Creating FAISS index...")
    if all_chunks:
        faiss_index = FAISS.from_documents(all_chunks, _embedding_model)
        faiss_index.save_local(PRELOADED_INDEX_PATH) # Save to the preloaded path
        status_text.text(f"FAISS index created and saved with {files_processed_count} documents.")
        return faiss_index, files_processed_count
    else:
        status_text.text("No chunks were generated from the documents.")
        return None, 0

# --- Streamlit Caching for FAISS index loading ---
@st.cache_resource
def load_faiss_index_from_disk(path, _embedding_model):
    """Loads FAISS index from disk if it exists."""
    if os.path.exists(path) and os.path.isdir(path): # Check if it's a directory
        try:
            # The allow_dangerous_deserialization=True is needed for FAISS with custom embeddings.
            # Ensure you trust the source of your index files.
            return FAISS.load_local(path, _embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading FAISS index from {path}: {e}. Will try to rebuild.")
            # Potentially delete the corrupted index directory so it can be rebuilt
            # import shutil
            # shutil.rmtree(path, ignore_errors=True)
            return None
    return None

# --- Initialize session state ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
# No need for preloaded_index_loaded or user_index_loaded if we rely on st.session_state.faiss_index directly


# --- Load or Process Documents on Startup ---
if st.session_state.faiss_index is None: # Only attempt to load or process if not already in session
    # Try to load the pre-built index first
    with st.spinner("🔄 Loading document index..."):
        st.session_state.faiss_index = load_faiss_index_from_disk(PRELOADED_INDEX_PATH, embedding_model)

    if st.session_state.faiss_index:
        initial_docs_in_dir = get_documents_in_directory(PDF_DIR)
        st.session_state.files_processed = len(initial_docs_in_dir) # Or better, count from index metadata if possible
        st.success(f"✅ Preloaded index with approximately {st.session_state.files_processed} documents loaded from disk!")
    else:
        st.info("Preloaded index not found or failed to load. Processing documents from PDF directory to create a new index. This might take some time...")
        # Process documents in directory and create the index for the first time
        faiss_index_created, num_files_processed = process_directory_documents(PDF_DIR, embedding_model)
        if faiss_index_created:
            st.session_state.faiss_index = faiss_index_created
            st.session_state.files_processed = num_files_processed
            st.success(f"✅ Directory documents processed and index created with {num_files_processed} documents.")
        else:
            st.warning("⚠️ No documents were found or processed in the PDF directory. Please add documents to the 'pdf' folder.")


# --- Two-column layout ---
left_col, right_col = st.columns([3, 5]) # Adjusted column ratio for better balance

# --- Left column: File upload for additional documents ---
with left_col:
    st.subheader("📁 Document Status & Uploads")
    st.write(f"**Current Total Documents Indexed: {st.session_state.files_processed}**")
    if st.session_state.faiss_index and st.session_state.files_processed > 0:
        st.markdown(f"<span style='color:green'>✅ Index active with {st.session_state.files_processed} documents.</span>", unsafe_allow_html=True)
    elif not get_documents_in_directory(PDF_DIR):
         st.warning(f"⚠️ No documents found in the '{PDF_DIR}' directory. Please add some or upload below.")


    st.markdown("---")
    st.write("**Upload additional PDF/Word documents (max 10 files):**")
    uploaded_files = st.file_uploader(
        "Choose files to add to the current session's knowledge base",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("🚫 You can upload a maximum of 10 additional files at a time.")
        else:
            if st.button("➕ Process Uploaded Documents", key="process_uploaded_button"):
                with st.spinner("⏳ Processing uploaded documents..."):
                    all_new_chunks = []
                    additional_processed_count = 0
                    temp_file_paths = []

                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                            temp_file_paths.append(tmp_path) # Keep track for cleanup

                        try:
                            loaded_docs = load_document_from_path(tmp_path, file_type)
                            if not loaded_docs:
                                st.warning(f"Could not load uploaded file: {uploaded_file.name}")
                                continue

                            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                            chunks = splitter.split_documents(loaded_docs)

                            for chunk in chunks:
                                if not chunk.metadata:
                                    chunk.metadata = {}
                                chunk.metadata['source'] = f"UPLOADED: {uploaded_file.name}" # Distinguish uploaded

                            all_new_chunks.extend(chunks)
                            additional_processed_count += 1
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        # No finally os.unlink(tmp_path) here, do it after processing all files

                    if all_new_chunks:
                        if st.session_state.faiss_index:
                            st.session_state.faiss_index.add_documents(all_new_chunks, embedding=embedding_model)
                            st.session_state.files_processed += additional_processed_count
                            st.success(f"✅ {additional_processed_count} uploaded documents added to the index!")
                        else: # Should not happen if initial loading worked, but as a fallback
                            st.session_state.faiss_index = FAISS.from_documents(all_new_chunks, embedding_model)
                            st.session_state.files_processed = additional_processed_count
                            st.success(f"✅ Index created with {additional_processed_count} uploaded documents.")
                        # Optionally, save this new combined index if persistence for uploads is desired
                        # user_session_index_path = get_session_index_path(st.session_state.session_id)
                        # st.session_state.faiss_index.save_local(user_session_index_path)
                        # st.info(f"Updated session index saved to {user_session_index_path}")
                    else:
                        st.error("No content could be processed from the uploaded files.")

                    # Clean up all temporary files
                    for path in temp_file_paths:
                        try:
                            os.unlink(path)
                        except Exception as e:
                            st.warning(f"Could not delete temp file {path}: {e}")
                    st.rerun() # Rerun to update UI elements correctly after processing

# --- Right column: Chat interface ---
with right_col:
    st.subheader("💬 Chat with Your Documents")
    if st.session_state.faiss_index:
        # Display conversation history
        chat_container = st.container() # Use a container for chat messages
        with chat_container:
            for i, (query, answer) in enumerate(st.session_state.conversation_history):
                st.markdown(f'<div class="chat-message user-message"><b>You:</b><br>{query}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message bot-message"><b>Gemini:</b><br>{answer}</div>', unsafe_allow_html=True)

        # User query input
        user_query = st.text_input("Ask a question about the content of your documents:", key="user_query_input")

        if st.button("Submit Question", key="submit_query_button") and user_query:
            with st.spinner("🔎 Searching and thinking..."):
                try:
                    relevant_docs = st.session_state.faiss_index.similarity_search(user_query, k=5) # k=5 for top 5 docs

                    context_parts = []
                    for doc in relevant_docs:
                        source = doc.metadata.get('source', 'Unknown source')
                        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
                    context = "\n\n---\n\n".join(context_parts) # Separator for clarity

                    prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context. If the answer is not found in the context, say "I could not find an answer in the provided documents." Do not make up information.

Context:
{context}

Question: {user_query}
Answer:"""

                    response_obj = llm.invoke(prompt) # Gemini response object
                    raw_answer = response_obj.content.strip() # Access .content
                    clean_answer = extract_answer_only(raw_answer) # Your existing cleaner

                    st.session_state.conversation_history.append((user_query, clean_answer))
                    st.rerun() # Rerun to update the chat display

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # You might want to log the full error for debugging
                    # print(f"Error during LLM invocation or processing: {traceback.format_exc()}")


        if st.session_state.conversation_history:
            if st.button("🗑️ Clear Conversation", key="clear_chat_button"):
                st.session_state.conversation_history = []
                st.rerun()
    else:
        st.warning("👈 Please ensure documents are loaded or uploaded to activate the chat.")
        if not get_documents_in_directory(PDF_DIR) and not uploaded_files:
            st.info(f"Add documents to the '{PDF_DIR}' folder and restart, or upload files using the panel on the left.")
