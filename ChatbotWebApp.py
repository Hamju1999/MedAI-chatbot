import os
import re
import nltk
import PyPDF2
import textstat
import streamlit as st
from openai import OpenAI

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

    # Clean each line by removing extra spaces/non-ASCII characters
    lines = [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]
    return lines

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

def evaluatereadability(simplifiedtext):
    return textstat.flesch_reading_ease(simplifiedtext)

##########################
# Streamlit App Interface
##########################

st.title("Discharge Instruction Simplifier")

# 1. File uploader (similar look to "Browse files")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

# 2. If no file uploaded, we stop here.
if not uploadfile:
    st.stop()

# 3. Process the file
data = loadandpreprocess(uploadfile)
if not data:
    st.warning("No valid data found in the file.")
    st.stop()

originaltext = " ".join(data)

# 4. Initialize the OpenRouter client
with st.spinner("Initializing OpenRouter client..."):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"]
    )

# 5. Simplify the text
with st.spinner("Simplifying the text..."):
    # For Clinician, we’ll allow them to enter patient context.
    # For Patient, we’ll set context = None automatically below.
    # But the user can switch tabs at any time, so we handle both in the tab sections.
    simplifiedtext_no_context = simplifytext(originaltext, client, patientcontext=None)

# 6. Create horizontal tabs
tabs = st.tabs(["Patient View", "Clinician View"])

# --------------------------
#   PATIENT VIEW TAB
# --------------------------
with tabs[0]:
    st.subheader("Simplified Text (Patient View)")
    # Show the simplified text with NO patient context
    st.write(simplifiedtext_no_context)

    # Show reading ease
    readability_score = evaluatereadability(simplifiedtext_no_context)
    st.subheader("Readability Score (Flesch Reading Ease)")
    st.write(readability_score)

# --------------------------
#   CLINICIAN VIEW TAB
# --------------------------
with tabs[1]:
    # Let the clinician optionally provide context in real-time
    st.subheader("Clinician Tools")
    patientcontext_input = st.text_input("Enter patient context (optional):")

    # If they provide or modify context, re-generate the simplified text on the fly
    # We'll do so only if the user clicks a button to confirm.
    if st.button("Simplify with Context"):
        with st.spinner("Re-simplifying with Clinician's patient context..."):
            simplifiedtext_with_context = simplifytext(
                originaltext,
                client,
                patientcontext=patientcontext_input
            )
        # Display results
        st.subheader("Original Discharge Instructions (Clinician View)")

        # Show each paragraph as a normal text block, no bullet points
        for paragraph in data:
            # If you want headings in bold, you can do something like:
            # if paragraph.isupper() or paragraph.endswith(":"):
            #     st.markdown(f"**{paragraph}**")
            # else:
            #     st.write(paragraph)
            st.write(paragraph)
            st.write("")

        if patientcontext_input:
            st.subheader("Patient Context Provided")
            st.write(patientcontext_input)

        st.subheader("Simplified Text (With Context)")
        st.write(simplifiedtext_with_
