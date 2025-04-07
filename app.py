import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import glob
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="MediSynapse - Clinical Notes RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "knowledge_graphs" not in st.session_state:
    st.session_state.knowledge_graphs = {}
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {
        "knowledge_graphs_loaded": False,
        "notes_processed": False,
        "index_created": False
    }

# Pre-loaded example queries
EXAMPLE_QUERIES = [
    "What are the symptoms of myocardial infarction?",
    "How to treat pneumonia?",
    "What are the risk factors for diabetes?",
    "Describe the treatment protocol for hypertension",
    "What are the diagnostic criteria for heart failure?"
]

# Custom CSS
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #f0f2f6;
        color: #000000;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000;
    }
    
    /* Text color for all elements */
    .stMarkdown, .stText, .stChatMessageContent, .stChatMessage {
        color: #000000 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #000000 !important;
    }
    
    /* Input text */
    .stTextInput>div>div>input {
        color: #000000 !important;
    }
    
    /* Subheader text */
    .stSubheader {
        color: #000000 !important;
    }
    
    /* Links */
    a {
        color: #000000 !important;
        text-decoration: underline;
    }
    
    /* Chat input placeholder */
    .stChatInputContainer input::placeholder {
        color: #666666 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg {
        color: #000000 !important;
    }
    
    /* Make sure all text in the app is black */
    * {
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the models
@st.cache_resource
def get_models():
    # Initialize GROQ model for general responses
    groq_model = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Initialize Roberta-based QA model
    qa_model = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Initialize Mistral-7B model
    mistral_model = HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        device=0 if torch.cuda.is_available() else -1,
        model_kwargs={"temperature": 0.7, "max_length": 1000}
    )
    
    return groq_model, qa_model, mistral_model

# Initialize embeddings with specific model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

# Create chat prompt template
@st.cache_resource
def get_chat_prompt():
    return ChatPromptTemplate.from_template("""
    You are a medical expert assistant analyzing clinical notes using the MIMIC-IV-Ext-DiReCT dataset. Your role is to:
    1. Analyze clinical notes and identify diagnostic procedures
    2. Extract relevant observations and rationales
    3. Map findings to the knowledge graph structure
    4. Provide detailed explanations of diagnostic reasoning
    5. Maintain a professional and empathetic tone

    Knowledge Graph Context: {knowledge_graph}
    Clinical Notes: {context}
    Question: {question}
    Answer: 
    """)

# Function to load knowledge graphs
def load_knowledge_graphs(directory: str) -> Dict[str, Dict]:
    knowledge_graphs = {}
    for file_path in glob.glob(os.path.join(directory, "*.json")):
        with open(file_path, 'r') as f:
            disease_category = Path(file_path).stem
            knowledge_graphs[disease_category] = json.load(f)
    return knowledge_graphs

# Function to process clinical notes with specific chunking
def process_notes(notes: List[Dict]) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Extract relevant information from notes
    processed_texts = []
    for note in notes:
        # Combine all input sections
        combined_text = "\n".join([
            f"Chief Complaint: {note.get('input1', '')}",
            f"History of Present Illness: {note.get('input2', '')}",
            f"Past Medical History: {note.get('input3', '')}",
            f"Physical Examination: {note.get('input4', '')}",
            f"Laboratory Results: {note.get('input5', '')}",
            f"Pertinent Results: {note.get('input6', '')}"
        ])
        processed_texts.append(combined_text)
    
    # Split text into chunks
    chunks = text_splitter.split_text("\n".join(processed_texts))
    return chunks

# Function to create FAISS vector store
def create_vector_store(texts: List[str], embeddings):
    # Create embeddings
    embeddings_model = embeddings
    vectors = embeddings_model.embed_documents(texts)
    
    # Convert to numpy array
    vectors = np.array(vectors)
    
    # Create FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    # Create vector store
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Function to process query and generate response
def process_query(query: str, vector_store, knowledge_graphs: Dict[str, Dict]) -> Tuple[str, List[str]]:
    # Get models
    groq_model, qa_model, mistral_model = get_models()
    
    # Get relevant knowledge graph context
    relevant_graph = next(iter(knowledge_graphs.values()))
    
    # Retrieve relevant chunks
    relevant_chunks = vector_store.similarity_search(query, k=3)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Step 1: Extract precise answers using Roberta
    qa_result = qa_model(
        question=query,
        context=context,
        max_answer_len=100
    )
    
    # Step 2: Generate detailed response using Mistral
    mistral_prompt = f"""
    Based on the following medical context and question, provide a detailed clinical response:
    
    Context: {context}
    Question: {query}
    Precise Answer: {qa_result['answer']}
    
    Please provide a comprehensive medical response that:
    1. Explains the condition or treatment in detail
    2. Includes relevant medical terminology
    3. Provides context from the clinical notes
    4. Maintains a professional medical tone
    """
    
    detailed_response = mistral_model(mistral_prompt)
    
    # Step 3: Generate final response using GROQ
    final_prompt = f"""
    Based on the following information, provide a clear and concise medical response:
    
    Question: {query}
    Precise Answer: {qa_result['answer']}
    Detailed Response: {detailed_response}
    Knowledge Graph Context: {json.dumps(relevant_graph)}
    
    Please provide a final response that:
    1. Is clear and concise
    2. Includes the most relevant information
    3. Maintains medical accuracy
    4. Is easy to understand
    """
    
    final_response = groq_model.invoke(final_prompt)
    
    return final_response, [chunk.page_content for chunk in relevant_chunks]

# Function to generate clinical notes
def generate_clinical_notes(prompt: str) -> str:
    _, _, mistral_model = get_models()
    
    clinical_prompt = f"""
    Generate a detailed clinical note based on the following information:
    
    {prompt}
    
    Please include:
    1. Chief Complaint
    2. History of Present Illness
    3. Past Medical History
    4. Physical Examination
    5. Laboratory Results
    6. Assessment and Plan
    """
    
    return mistral_model(clinical_prompt)

# Main application
def main():
    st.title("🏥 MediSynapse - Clinical Notes RAG System")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Clinical Query Answering", "Medical Text Generation"])
    
    with tab1:
        st.subheader("💬 Ask Questions About Clinical Notes")
        
        # Sidebar for data upload
        with st.sidebar:
            st.subheader("📊 Data Processing & Indexing")
            
            # Upload knowledge graphs
            st.subheader("🌐 Diagnostic Knowledge Graphs")
            kg_directory = st.text_input("Enter path to knowledge graphs directory:", "diagnostic_kg")
            if os.path.exists(kg_directory):
                with st.spinner("Loading knowledge graphs..."):
                    st.session_state.knowledge_graphs = load_knowledge_graphs(kg_directory)
                    st.session_state.processing_status["knowledge_graphs_loaded"] = True
                    st.success(f"✅ Loaded {len(st.session_state.knowledge_graphs)} knowledge graphs")
            
            # Upload clinical notes
            st.subheader("🏥 Annotated Clinical Notes")
            uploaded_file = st.file_uploader("Upload clinical notes (JSON)", type=["json"])
            
            if uploaded_file:
                with st.spinner("Processing clinical notes..."):
                    notes_data = json.load(uploaded_file)
                    if isinstance(notes_data, list):
                        texts = process_notes(notes_data)
                        st.session_state.processing_status["notes_processed"] = True
                        st.success(f"✅ Processed {len(texts)} text chunks")
                        
                        with st.spinner("Creating FAISS index..."):
                            embeddings = get_embeddings()
                            st.session_state.vector_store = create_vector_store(texts, embeddings)
                            st.session_state.processing_status["index_created"] = True
                            st.success("✅ FAISS index created successfully!")
        
        # Example queries
        st.subheader("⚡ Example Queries")
        col1, col2, col3 = st.columns(3)
        for i, query in enumerate(EXAMPLE_QUERIES):
            if i % 3 == 0:
                col1.button(query, key=f"query_{i}")
            elif i % 3 == 1:
                col2.button(query, key=f"query_{i}")
            else:
                col3.button(query, key=f"query_{i}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- {source}")
        
        # Chat input
        if query := st.chat_input("Ask a question about the clinical notes..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Process query and display response
            if all(st.session_state.processing_status.values()):
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing..."):
                        response, sources = process_query(
                            query, 
                            st.session_state.vector_store,
                            st.session_state.knowledge_graphs
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
            else:
                missing_steps = [k for k, v in st.session_state.processing_status.items() if not v]
                st.warning(f"⚠️ Please complete the following steps first: {', '.join(missing_steps)}")
    
    with tab2:
        st.subheader("📝 Generate Clinical Notes")
        
        # Input for clinical note generation
        note_prompt = st.text_area(
            "Enter details for clinical note generation:",
            placeholder="Patient is a 45-year-old male presenting with chest pain..."
        )
        
        if st.button("Generate Clinical Note"):
            with st.spinner("Generating clinical note..."):
                clinical_note = generate_clinical_notes(note_prompt)
                st.text_area("Generated Clinical Note:", clinical_note, height=300)

if __name__ == "__main__":
    main() 