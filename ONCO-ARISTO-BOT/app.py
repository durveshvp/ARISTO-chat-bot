import os
import streamlit as st
import tempfile
import hashlib
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from dotenv import load_dotenv
import re
import uuid
import glob


st.set_page_config(layout="wide")  # Optional: make layout wide by default

# Inject custom CSS to expand full width
# Inject custom CSS with fixed color contrast
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
        color: #000000 !important;  /* Ensure black text for bot messages */
        border-left: 5px solid #2ecc71;
    }
    
    /* Ensure text inside messages is appropriately colored */
    .user-message b, .user-message p, .user-message span {
        color: white !important;
    }
    
    .bot-message b, .bot-message p, .bot-message span, .bot-message div {
        color: #000000 !important;
    }
    
    /* Fix source view text */
    .st-expander {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

# PDF directory and index paths
PDF_DIR = "pdf"  # Directory containing pre-loaded documents
INDEX_DIR = "index"  # Directory to store the index
PRELOADED_INDEX_PATH = os.path.join(INDEX_DIR, "preloaded_index")  # Path for preloaded documents

# Create necessary directories
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceHub(
    repo_id=MODEL_ID,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

# Title
st.markdown("""
    <h1 style='text-align: center;'>📄 Chat with Documents using FAISS + HuggingFace</h1>
""", unsafe_allow_html=True)


# Helper functions
def extract_answer_only(text):
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Answer not found."

def get_session_index_path(session_id):
    return os.path.join(INDEX_DIR, f"session_{session_id}")

def load_document(file_path, file_type):
    if file_type == "pdf":
        return PyPDFLoader(file_path).load()
    elif file_type in ["docx", "doc"]:
        return UnstructuredWordDocumentLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def get_documents_in_directory():
    """Get list of PDF and Word documents in the PDF_DIR directory"""
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    doc_files = glob.glob(os.path.join(PDF_DIR, "*.doc")) + glob.glob(os.path.join(PDF_DIR, "*.docx"))
    return pdf_files + doc_files

def process_directory_documents():
    """Process all documents in the PDF_DIR directory"""
    documents = get_documents_in_directory()
    if not documents:
        return None, 0
    
    all_chunks = []
    files_processed = 0
    
    for file_path in documents:
        try:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            documents = load_document(file_path, file_type)
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            
            # Add source file information to metadata
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['source'] = os.path.basename(file_path)
            
            all_chunks.extend(chunks)
            files_processed += 1
            
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
    
    # Create and save the index if documents were processed
    if all_chunks:
        faiss_index = FAISS.from_documents(all_chunks, embedding_model)
        faiss_index.save_local(PRELOADED_INDEX_PATH)
        return faiss_index, files_processed
    
    return None, 0

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'preloaded_index_loaded' not in st.session_state:
    st.session_state.preloaded_index_loaded = False
if 'user_index_loaded' not in st.session_state:
    st.session_state.user_index_loaded = False
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Check if preloaded index exists or process directory documents
if not st.session_state.preloaded_index_loaded:
    if os.path.exists(PRELOADED_INDEX_PATH):
        st.session_state.faiss_index = FAISS.load_local(PRELOADED_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        st.session_state.preloaded_index_loaded = True
        
        # Count preloaded documents
        documents = get_documents_in_directory()
        st.session_state.files_processed = len(documents)
    else:
        # Process documents in directory
        faiss_index, files_processed = process_directory_documents()
        if faiss_index:
            st.session_state.faiss_index = faiss_index
            st.session_state.preloaded_index_loaded = True
            st.session_state.files_processed = files_processed

# Two-column layout
left_col, right_col = st.columns([3, 5])

# Left column: File upload for additional documents
with left_col:
    st.write(f"**Current Documents: {st.session_state.files_processed}**")
    if st.session_state.preloaded_index_loaded:
        st.success(f"✅ {st.session_state.files_processed} documents from directory processed and ready!")
    
    st.write("Upload additional PDF and Word documents (maximum 10 files):")
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check if number of files exceeds the limit
        if len(uploaded_files) > 10:
            st.error("You can upload a maximum of 10 additional files.")
        else:
            # Process only if we have files and user index is not already loaded
            if not st.session_state.user_index_loaded and st.button("Process Additional Documents"):
                with st.spinner("Processing uploaded documents..."):
                    all_chunks = []
                    additional_processed = 0
                    
                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Load document based on file type
                            documents = load_document(tmp_path, file_type)
                            
                            # Split into chunks
                            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                            chunks = splitter.split_documents(documents)
                            
                            # Add source file information
                            for chunk in chunks:
                                if not chunk.metadata:
                                    chunk.metadata = {}
                                chunk.metadata['source'] = uploaded_file.name
                            
                            all_chunks.extend(chunks)
                            additional_processed += 1
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temp file
                            os.unlink(tmp_path)
                    
                    # Add to existing index or create new one
                    if all_chunks:
                        user_index_path = get_session_index_path(st.session_state.session_id)
                        
                        if st.session_state.faiss_index:
                            # Add to existing index
                            st.session_state.faiss_index.add_documents(all_chunks)
                        else:
                            # Create new index
                            st.session_state.faiss_index = FAISS.from_documents(all_chunks, embedding_model)
                        
                        # Save the updated index
                        st.session_state.faiss_index.save_local(user_index_path)
                        st.session_state.user_index_loaded = True
                        st.session_state.files_processed += additional_processed
                        st.success(f"✅ {additional_processed} additional documents processed!")
                    else:
                        st.error("No additional documents were successfully processed.")

# Right column: Chat interface
with right_col:
    if st.session_state.faiss_index:
        # Display conversation history
        for i, (query, answer) in enumerate(st.session_state.conversation_history):
            # User message
            st.markdown(f'<div class="chat-message user-message"><b>You:</b><br>{query}</div>', 
                        unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f'<div class="chat-message bot-message"><b>Answer:</b><br>{answer}</div>', 
                        unsafe_allow_html=True)

        # Text area for user query
        user_query = st.text_input("Ask a question about the documents:")
        
        # Submit button
        if st.button("Submit Question") and user_query:
            with st.spinner("🔎 Searching..."):
                relevant_docs = st.session_state.faiss_index.similarity_search(user_query, k=5)
                
                # Create context with source information when available
                context_parts = []
                for doc in relevant_docs:
                    source = doc.metadata.get('source', 'Unknown document')
                    context_parts.append(f"[Source: {source}]\n{doc.page_content}")
                
                context = "\n\n".join(context_parts)
                prompt = f"""Use the following context to answer the question:

{context}

Question: {user_query}
Answer:"""

                response = llm.invoke(prompt).strip()
                clean_answer = extract_answer_only(response)
                
                # Add to conversation history
                st.session_state.conversation_history.append((user_query, clean_answer))
                
                # Display the latest response
                # st.markdown(f'<div class="chat-message user-message"><b>You:</b><br>{user_query}</div>', 
                #             unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message bot-message" style="color: #000000 !important;"><b style="color: #000000 !important;">Answer:</b><br><span style="color: #000000 !important;">{clean_answer}</span></div>', 
            unsafe_allow_html=True)
                
                # Show sources used
                with st.expander("View sources"):
                    for i, doc in enumerate(relevant_docs):
                        source = doc.metadata.get('source', 'Unknown document')
                        st.markdown(f"**Source {i+1}: {source}**")
                        st.write(doc.page_content)
                        st.markdown("---")
    elif not get_documents_in_directory() and not uploaded_files:
        st.warning("👈 No documents found in the 'pdf' directory. Please add documents to the directory or upload them through the interface.")
    elif not uploaded_files:
        st.warning("👈 Processing documents from directory...")
    else:
        st.warning("👈 Please click 'Process Additional Documents' to analyze your files.")

    # Add button to clear conversation history
    if st.session_state.conversation_history and st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()