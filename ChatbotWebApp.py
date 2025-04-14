import os
import re
import uuid
import nltk
import PyPDF2
import textstat
import math
import numpy as np
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Ensure required NLTK packages are available
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

llmcache = {}

def loadandpreprocess(uploadfile):
    _, ext = os.path.splitext(uploadfile.name)
    text = ""
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    elif ext.lower() == ".txt":
        try:
            text = uploadfile.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
    
    return [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]

def simplifytext(text, client, patientcontext=None):
    prompt = (
        f"Patient Context:\n{patientcontext}\n\n"
        f"Medical Instructions:\n{text}\n\n"
        "Instructions:\n"
        "1. Read and interpret the medical instructions above carefully.\n"
        "2. Rewrite the discharge instructions in plain, patient-friendly language. Use simple and clear language suitable for someone with limited medical knowledge.\n"
        "3. Organize the output into a logically structured list, with each item representing a distinct task, follow-up, or instruction.\n"
        "4. Ensure that all essential medical details are maintained and accurate, particularly relating to medication, follow-up appointments, and critical actions.\n"
        "5. Aim for a final Flesch Reading Ease score between 80 and 90. Do not include any readability score or technical details in the output.\n"
        "6. Verify that the reformulated text adheres to medical accuracy and clarity: check that instructions do not conflict and that any scheduled appointments, medication dosages, or conditions are clearly described.\n\n"
        "Output:\n"
        "- A bullet-point list that clearly separates tasks and key instructions.\n"
        "- Medical statements must be accurate, clear, and easy to understand."
       # "Convert the following discharge instructions into plain, patient-friendly language, ensuring accuracy with respect to the MTSamples discharge summary. "
       # "Retain all essential details while reformulating the text so that it achieves a Flesch Reading Ease score between 80 and 90. Dont output Flesch Reading Ease score check "
       # "final simplified text should be focused on list of tasks, follow-ups, and their importance from the discharge instructions."
       # "Below is a simplified version of discharge instructions. Please examine the text sentence by sentence and extract only those sentences that contain at least one of the following actionable keywords: 'follow', 'call', 'take', 'return', 'appointment', 'contact', 'schedule', or 'medication'. Return your answer as a list of the relevant sentences."
    )
    if prompt in llmcache:
        return llmcache[prompt]
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  
            top_p=1        
        )
        result = response.choices[0].message.content
        llmcache[prompt] = result
        return result
    except Exception as e:
        return f"[OpenRouter Error] {e}"

def evaluatereadability(simplifiedtext):
    score = textstat.flesch_reading_ease(simplifiedtext)
    return score

def chunk_text(text, chunk_size=500):
    """
    Splits the input text into chunks containing approximately chunk_size words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Create a dedicated embeddings client using the official OpenAI endpoint.
embeddings_client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text):
    try:
        response = embeddings_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        # Convert the raw embedding to a NumPy array with 32-bit precision.
        raw_embedding = response.data[0].embedding
        embedding = np.array(raw_embedding, dtype=np.float32)
        
        # Replace any NaN or infinite values with 0.0.
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verify the embedding has the expected dimension (1536 for text-embedding-ada-002).
        expected_dim = 1536
        if embedding.shape[0] != expected_dim:
            st.error(f"Embedding dimension mismatch: expected {expected_dim} but got {embedding.shape[0]}")
            return None
        
        # Normalize the embedding vector (L2 normalization).
        norm = np.linalg.norm(embedding)
        if norm == 0:
            st.error("Embedding vector has zero norm.")
            return None
        normalized = embedding / norm

        # Round the normalized vector to reduce precision issues.
        normalized = np.round(normalized, decimals=8)
        
        return normalized.tolist()
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None
def upsert_chunks(chunks, index):
    """
    Generates embeddings for each chunk and upserts them into the provided Pinecone index.
    """
    vectors = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb:
            vector_id = str(uuid.uuid4())
            vectors.append((vector_id, emb, {"text": chunk}))
    if vectors:
        index.upsert(vectors)
    return index

def retrieve_relevant_chunks(query, index, top_k=5):
    query_emb = get_embedding(query)
    if query_emb is None:
        st.error("No valid query embedding generated.")
        return []
    try:
        result = index.query(queries=[query_emb], top_k=top_k, include_metadata=True)
        matches = result["results"][0]["matches"]
        retrieved_chunks = [match["metadata"]["text"] for match in matches]
        return retrieved_chunks
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return []

##########################
# Streamlit App Interface
##########################

st.title("Discharge Instruction with RAG Enhancement")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

if uploadfile is not None:
    data = loadandpreprocess(uploadfile)
    if data:
        originaltext = " ".join(data)
        
        with st.spinner("Initializing OpenRouter client..."):
            # The OpenRouter client is used for completions only.
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        
        # Initialize Pinecone using the new API via the Pinecone class
        with st.spinner("Initializing Pinecone Vector DB..."):
            index_name = "discharge-instructions"
            # Create a Pinecone instance (serverless)
            pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
            # Check if index exists; if not, create one with the proper dimension (1536) and metric ("cosine")
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=st.secrets["PINECONE_CLOUD"],    # e.g., "aws" or "gcp"
                        region=st.secrets["PINECONE_REGION"]     # e.g., "us-west1-gcp" or "us-west-2"
                    )
                )
            index = pc.Index(index_name)
        
        # Chunk the original text into manageable pieces
        chunks = chunk_text(originaltext, chunk_size=500)
        
        with st.spinner("Upserting text chunks into vector DB..."):
            index = upsert_chunks(chunks, index)
        
        # Allow user to enter additional patient context (optional)
        patientcontext = st.text_input("Enter patient context (optional):")
        
        # Allow user to optionally ask a specific query about the instructions
        query = st.text_input("Enter a query regarding the discharge instructions (optional):")
        st.write("Query entered:", query)
        
        if query:
            with st.spinner("Retrieving relevant text chunks based on your query..."):
                # Debug: check the query embedding before running the Pinecone query.
                query_emb = get_embedding(query)
                if query_emb:
                    st.write("Query embedding sample (first 10 values):", query_emb[:10])
                else:
                    st.error("Failed to generate a valid query embedding.")
                
                # Try retrieving relevant chunks if query embedding is valid.
                if query_emb is not None:
                    try:
                        relevant_chunks = retrieve_relevant_chunks(query, index, top_k=5)
                    except Exception as e:
                        st.error(f"Error in retrieving chunks: {e}")
                        relevant_chunks = []
                else:
                    relevant_chunks = []
            
            # Combine retrieved chunks into a single text block.
            combined_text = " ".join(relevant_chunks)
            
            with st.spinner("Simplifying text based on retrieved relevant chunks..."):
                simplifiedtext = simplifytext(combined_text, client, patientcontext=patientcontext)
        else:
            with st.spinner("Simplifying the complete text..."):
                simplifiedtext = simplifytext(originaltext, client, patientcontext=patientcontext)
                
        st.subheader("Simplified Text")
        st.write(simplifiedtext)
        
        readability = evaluatereadability(simplifiedtext)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)
    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file (PDF or TXT).")
