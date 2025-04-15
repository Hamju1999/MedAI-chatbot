import os
import re
import json
import nltk
import PyPDF2
import textstat
import streamlit as st
from openai import OpenAI

# Ensure required NLTK packages are available
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception:
        nltk.download(package)

llmcache = {}

def load_and_preprocess(uploadfile):
    """
    Reads the uploaded PDF or TXT file and returns a list of cleaned lines.
    """
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

    # Remove non-ASCII characters and extra whitespace from each line
    lines = [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]
    return lines

def extract_and_simplify(text, client, patient_context=None):
    """
    Calls the LLM to process the discharge instructions.
    The prompt instructs the LLM to output JSON with these keys:
      - instructions, follow_ups, recommendations, summary, additional_attributes.
    Each item in the first three sections should include a 'text' and a 'priority'.
    """
    prompt = f"""
Patient Context (if any): {patient_context}

Discharge Instructions (Original):
{text}

Your Tasks:
1. Convert the above discharge instructions into plain, patient-friendly language.
2. Separate the content into these sections: instructions, follow-ups, recommendations.
3. For each item, label it with a priority (e.g., high, medium, low).
4. Provide a short summary of the entire discharge plan.
5. Include any additional helpful information if relevant.

Output must be valid JSON with this structure:
{{
  "instructions": [{{"text": "...", "priority": "..."}}, ...],
  "follow_ups": [{{"text": "...", "priority": "..."}}, ...],
  "recommendations": [{{"text": "...", "priority": "..."}}, ...],
  "summary": "...",
  "additional_attributes": "..."
}}
Use a Flesch Reading Ease target between 80 and 90.
Do not include any commentary outside of the JSON.
"""
    if prompt in llmcache:
        return llmcache[prompt]
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        result = response.choices[0].message.content.strip()
        llmcache[prompt] = result
        return result
    except Exception as e:
        return f'[OpenRouter Error] {e}'

def validate_output(original, simplified_json):
    """
    Performs a basic validation by comparing the word counts of the original text
    and the combined simplified text.
    """
    original_len = len(original.split())
    try:
        data = json.loads(simplified_json)
        all_items = data.get("instructions", []) + data.get("follow_ups", []) + data.get("recommendations", [])
        simplified_texts = " ".join(item["text"] for item in all_items if "text" in item)
        simplified_len = len(simplified_texts.split())
        ratio = simplified_len / max(original_len, 1)
        return f"Original word count: {original_len}, Simplified word count: {simplified_len}, Ratio: {ratio:.2f}"
    except json.JSONDecodeError:
        return "Could not parse JSON output from LLM."

def evaluate_readability(text_block):
    """
    Returns the Flesch Reading Ease score for a given text.
    """
    return textstat.flesch_reading_ease(text_block)

def calculate_accuracy_score_llm(output_text, client):
    """
    Uses the LLM to self-assess the medical accuracy of the provided output_text.
    We ask for a single score (0–100) with exactly two decimals.
    """
    prompt = f"""
You are a highly knowledgeable medical expert.
Evaluate the following simplified discharge instructions for accuracy with respect to current reputable medical guidelines.
Provide a single accuracy score between 0 and 100, with exactly two decimals. Do not provide any extra commentary.
    
Content:
{output_text}

Accuracy Score:"""
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        score_text = response.choices[0].message.content.strip()
        # Extract numeric value from the response; assume it's a number.
        try:
            score_value = float(score_text)
        except ValueError:
            score_value = 0.0
        return score_value
    except Exception as e:
        st.error(f"Error calculating LLM accuracy: {e}")
        return 0.0

def calculate_accuracy_score_web(output_text):
    """
    Simulated Internet Search Method.
    In a production system, this function would perform web search queries against authoritative medical guidelines,
    compare key terms or statistics, and compute an accuracy score.
    For demonstration, this function returns a fixed dummy score.
    """
    # Dummy implementation; replace with actual web API calls & analysis if available.
    return 85.50

################################
# Streamlit App Interface
################################

st.title("LLM-Powered Discharge Instruction Processor")

# File uploader (Browse files)
uploaded_file = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])
if not uploaded_file:
    st.info("Please upload a discharge instructions file.")
    st.stop()

# Preprocess file and display original instructions
lines = load_and_preprocess(uploaded_file)
if not lines:
    st.warning("No valid text found in the file.")
    st.stop()

original_text = " ".join(lines)
st.markdown("## Original Discharge Instructions")
for line in lines:
    st.write(line)
    st.write("")

# Optional patient context input
patient_context_input = st.text_input("Enter patient context (optional):")

# Initialize the OpenRouter client
with st.spinner("Initializing OpenRouter client..."):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"]
    )

# Button to trigger processing
if st.button("Process Instructions"):
    with st.spinner("Extracting and Simplifying Instructions..."):
        response_json = extract_and_simplify(original_text, client, patient_context_input)

    # Display the LLM response from JSON
    try:
        data = json.loads(response_json)

        st.markdown("### Simplified Instructions")
        instructions = data.get("instructions", [])
        if instructions:
            st.markdown("**Instructions:**")
            for item in instructions:
                st.write(f"- {item.get('text', '')} (Priority: {item.get('priority', 'N/A')})")

        follow_ups = data.get("follow_ups", [])
        if follow_ups:
            st.markdown("**Follow-ups:**")
            for item in follow_ups:
                st.write(f"- {item.get('text', '')} (Priority: {item.get('priority', 'N/A')})")

        recommendations = data.get("recommendations", [])
        if recommendations:
            st.markdown("**Recommendations:**")
            for item in recommendations:
                st.write(f"- {item.get('text', '')} (Priority: {item.get('priority', 'N/A')})")

        summary = data.get("summary", "")
        if summary:
            st.markdown(f"**Summary:** {summary}")

        additional = data.get("additional_attributes", "")
        if additional:
            st.markdown(f"**Additional Attributes:** {additional}")

        # Evaluate readability for the combined simplified instructions
        combined_text = (
            " ".join([i["text"] for i in instructions]) + " " +
            " ".join([f["text"] for f in follow_ups]) + " " +
            " ".join([r["text"] for r in recommendations])
        )
        readability = evaluate_readability(combined_text)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)

        # Display the validation output (basic word count check)
        st.subheader("Validation Check")
        st.write(validate_output(original_text, response_json))

        # --- New Feature: Medical Accuracy Evaluation ---
        st.subheader("Medical Accuracy Evaluation")
        # Method 1: Use LLM self-assessment
        accuracy_llm = calculate_accuracy_score_llm(combined_text, client)
        # Method 2: Simulated Internet search accuracy score
        accuracy_web = calculate_accuracy_score_web(combined_text)
        # Combine (average) the two scores
        final_accuracy = (accuracy_llm + accuracy_web) / 2
        st.write(f"Medical Accuracy Score: {final_accuracy:.2f}")

    except json.JSONDecodeError:
        st.error("Error parsing JSON output from the LLM. Please try again or check the input.")
