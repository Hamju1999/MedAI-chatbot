import os
import re
import uuid
import nltk
import PyPDF2
import textstat
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
    # DO NOT change the prompt below
    prompt = (
        f"Patient Context:\n{patientcontext}\n\n"
        f"Medical Instructions:\n{text}\n\n"
        "Use simple, clear language that someone with limited medical knowledge can easily understand.\n\n"
        "Convert the following discharge instructions into plain, patient-friendly language, ensuring accuracy with respect to the MTSamples discharge summary. "
        "Retain all essential details while reformulating the text so that it achieves a Flesch Reading Ease score between 80 and 90. Dont output Flesch Reading Ease score check "
        "final simplified text should be focused on list of tasks, follow-ups, and their importance from the discharge instructions."
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

def extractkeyinfo(simplifiedtext):
    sentences = nltk.sent_tokenize(simplifiedtext)
    keywords = ['follow', 'call', 'take', 'return', 'appointment', 'contact', 'schedule', 'medication']
    keyphrases = [sent for sent in sentences if any(keyword in sent.lower() for keyword in keywords)]
    return keyphrases

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

embeddings_client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def upsert_chunks(chunks, client, index):
    """
    Generates embeddings for each chunk and upserts them into the provided Pinecone index.
    """
    vectors = []
    for chunk in chunks:
        emb = get_embedding(chunk, client)
        if emb:
            vector_id = str(uuid.uuid4())
            vectors.append((vector_id, emb, {"text": chunk}))
    if vectors:
        index.upsert(vectors)
    return index

def retrieve_relevant_chunks(query, index, client, top_k=5):
    """
    Retrieves the top_k most relevant chunks from the Pinecone index given the query.
    """
    query_emb = get_embedding(query, client)
    if query_emb is None:
        return []
    result = index.query(queries=[query_emb], top_k=top_k, include_metadata=True)
    matches = result["results"][0]["matches"]
    retrieved_chunks = [match["metadata"]["text"] for match in matches]
    return retrieved_chunks

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
            index = upsert_chunks(chunks, client, index)
        
        # Allow user to enter additional patient context (optional)
        patientcontext = st.text_input("Enter patient context (optional):")
        
        # Allow user to optionally ask a specific query about the instructions
        query = st.text_input("Enter a query regarding the discharge instructions (optional):")
        
        if query:
            with st.spinner("Retrieving relevant text chunks based on your query..."):
                relevant_chunks = retrieve_relevant_chunks(query, index, client, top_k=5)
            # Combine retrieved chunks into a single text block
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
