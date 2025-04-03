import os
import json
import PyPDF2
import requests
import nltk
import textwrap
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from google import genai
from googlesearch import search as GoogleSearch
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK data (do this only once)
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

# Load sentence embedding model (do this only once)
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

similaritymodel = load_similarity_model()

class MedAI:
    def __init__(self):
        self.conversation_history = []
        self.GPTclient = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.deepseekclient = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        self.googleclient = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
        self.llamaclient = OpenAI(api_key=st.secrets["LLAMA_API_KEY"], base_url="https://api.llama-api.com")

    def cleantext(self, text: str) -> str:
        for char in ["---", "**", "*", "#"]:
            text = text.replace(char, "")
        return text

    def loadpdf(self, file_bytes):
        try:
            reader = PyPDF2.PdfReader(file_bytes)
            pdftext = ""
            for page in reader.pages:
                pdftext += page.extract_text() or ""
            st.info("PDF loaded successfully.")
            return pdftext
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    def pdfquery(self, query: str, pdftext: str) -> str:
        if pdftext:
            return (
                f"Patient Note:\n{pdftext}\n\n"
                f"Medical Query:\n{query}"
            )
        else:
            return query

    def analyzequery(self, query: str) -> dict:
        historycontext = "\n".join(
            [f"{turn['role']}: {turn['content']}" for turn in self.conversation_history[-4:]]
        )
        prompt = (
            "Analyze the following medical query. Identify the query type (diagnosis, treatment, prognosis, factual), "
            "the user intent (information, advice, etc.), and check for the presence of specific medical terminology. "
            "Return a JSON with keys: 'complexity' (true/false), 'query_type', 'intent', 'medical_terms_present' (true/false), "
            "and 'parsed_query'.\n\n"
            f"Conversation History (if any):\n{historycontext}\n\n"
            f"Query: {query}"
        )
        try:
            response = self.GPTclient.chat.completions.create(
                model="o1-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            result = json.loads(content)
        except Exception as e:
            commonmedterms = ["diabetes", "hypertension", "cancer", "infection", "diagnosis", "treatment", "prognosis"]
            medicalpresent = any(term in query.lower() for term in commonmedterms)
            result = {
                "complexity": len(query) > 50 or medicalpresent,
                "query_type": "factual",
                "intent": "information",
                "medical_terms_present": medicalpresent,
                "parsed_query": query.strip()
            }
        return result

    def primaryprompt(self, query: str, pdftext: str, analysis: dict) -> str:
        context = "\n".join(
            [f"{turn['role']}: {turn['content']}" for turn in self.conversation_history[-4:]]
        )
        basequery = self.pdfquery(query, pdftext)
        if context:
            return f"{context}\n\n{basequery}"
        else:
            return basequery

    def gpt4(self, query: str) -> str:
        prompt = (
            "Answer the following medical query with detailed reasoning, accurate information, and proper medical terminology. "
            "Cross-check your response with reliable medical knowledge:\n\n" + query
        )
        try:
            response = self.GPTclient.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[GPT-4 Error] {e}"

    def gpto1(self, query: str) -> str:
        prompt = (
            "Answer the following medical query succinctly with proper medical terminology and factual accuracy: "
            "Cross-check your response with reliable medical knowledge:\n\n" + query
        )
        try:
            response = self.GPTclient.chat.completions.create(
                model="o1-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[o1-mini Error] {e}"

    def gemini(self, query: str) -> str:
        prompt = (
            "Answer the following medical query with detailed reasoning and proper medical terminology. "
            "Verify facts against reliable medical sources:\n\n" + query
        )
        try:
            response = self.googleclient.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"[Gemini Error] {e}"
        finally:
            try:
                if hasattr(self.googleclient, "close"):
                    self.googleclient.close()
            except Exception:
                pass

    def gemma(self, query: str) -> str:
        prompt = (
            "Answer the following medical query with detailed reasoning and proper medical terminology. "
            "Cross-check facts with reliable sources:\n\n" + query
        )
        try:
            response = self.googleclient.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"[Gemma Error] {e}"
        finally:
            try:
                if hasattr(self.googleclient, "close"):
                    self.googleclient.close()
            except Exception:
                pass

    def deepseek(self, query: str) -> str:
        prompt = (
            "Answer the following medical query with detailed reasoning, ensuring factual accuracy and proper medical terminolog: "
            "Cross-check facts with reliable sources:\n\n" + query
        )
        try:
            response = self.deepseekclient.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[DeepSeek Error] {e}"

    def llama(self, query: str) -> str:
        prompt = (
            "Answer the following medical query with detailed reasoning and proper medical terminology: "
            "Cross-check facts with reliable sources:\n\n" + query
        )
        try:
            response = self.llamaclient.chat.completions.create(
                model="llama3.3-70b",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Llama Error] {e}"

    def aggregate(self, resultsdict: dict) -> str:
        seentexts = set()
        labeledtexts = []
        for modelname, text in resultsdict.items():
            if text not in seentexts:
                seentexts.add(text)
                labeledtexts.append(f"{modelname}:\n{text}")
        return "\n\n".join(labeledtexts)

    def gemrefine(self, query: str) -> str:
        prompt = (
            "Refine the following aggregated medical answer succinctly, ensuring factual accuracy. "
            "Cross-check against reliable medical sources (e.g., PubMed, MedlinePlus, CDC, Mayo Clinic, NIH) and use precise medical terminology:\n\n"
            f"{query}\n\nFinal Refined Answer:"
        )
        try:
            response = self.googleclient.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"[Gemini Refine Error] {e}"
        finally:
            try:
                if hasattr(self.googleclient, "close"):
                    self.googleclient.close()
            except Exception:
                pass

    def refine(self, answer: str) -> str:
        return self.gemrefine(answer)

    def searchmedical(self, query: str, num_results: int = 5) -> list:
        urls = []
        medicalquery = f"{query} site:pubmed.ncbi.nlm.nih.gov, site:medlineplus.gov, site:cdc.gov, site:mayoclinic.org, site:nih.gov"
        try:
            for url in GoogleSearch(medicalquery):
                urls.append(url)
                if len(urls) >= num_results:
                    break
        except Exception as e:
            st.error(f"Error during medical search: {e}")
        return urls

    def fetchurl(self, url: str) -> str:
        if not (url.startswith("http://") or url.startswith("https://")):
            st.warning(f"Skipping invalid URL: {url}")
            return ""
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        }
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return text
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error fetching {url}: {http_err}")
        except Exception as e:
            st.error(f"Error fetching {url}: {e}")
        return ""

    def splitsentences(self, text: str) -> list:
        return nltk.sent_tokenize(text)

    def fetchtexts(self, urls: list) -> list:
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futuretourl = {executor.submit(self.fetchurl, url): url for url in urls}
            for future in concurrent.futures.as_completed(futuretourl):
                url = futuretourl[future]
                try:
                    text = future.result()
                    if text:
                        results.append((url, text))
                except Exception as e:
                    st.error(f"Error processing URL {url}: {e}")
        return results

    def verifyinfo(self, refinedanswer: str, texts: list, threshold: float = 0.75) -> list:
        matches = []
        try:
            refinedembed = similaritymodel.encode(refinedanswer, convert_to_tensor=True)
        except Exception as e:
            st.error(f"Error encoding refined answer: {e}")
            return matches

        for url, text in texts:
            sentences = self.splitsentences(text)
            if not sentences:
                continue
            try:
                sentenceembed = similaritymodel.encode(sentences, convert_to_tensor=True)
                cosinescores = util.cos_sim(refinedembed, sentenceembed)[0]
                for i, score in enumerate(cosinescores):
                    if score.item() >= threshold:
                        matches.append({
                            'url': url,
                            'sentence': sentences[i],
                            'similarity': score.item()
                        })
            except Exception as e:
                st.error(f"Error processing text from {url}: {e}")
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches

    def verifyrefined(self, refinedanswer: str, verificationquery: str) -> dict:
        st.info(f"Starting verification using query: {verificationquery}")
        urls = self.searchmedical(verificationquery, num_results=5)
        if not urls:
            st.warning("No URLs found for verification.")
            return {"matches": [], "confidence": 0.0}
        urltexts = self.fetchtexts(urls)
        if not urltexts:
            st.warning("No content fetched for verification.")
            return {"matches": [], "confidence": 0.0}
        matches = self.verifyinfo(refinedanswer, urltexts, threshold=0.75)
        if matches:
            avgscore = sum(match['similarity'] for match in matches) / len(matches)
            confidence = avgscore * min(len(matches), 5)
        else:
            confidence = 0.0
        return {"matches": matches, "confidence": confidence}

    def synthesizeverifiedinfo(self, verificationmatches: list) -> str:
        if not verificationmatches:
            return ""
        matchedsentences = [match["sentence"] for match in verificationmatches]
        combinedtext = "\n".join(matchedsentences)
        synthesisprompt = (
            "Combine the following sentences extracted from verified medical sources. "
            "Provide a detailed consensus summary that is fact-checked and uses proper medical terminology. "
            "Ensure the final summary cross-references reliable sources (e.g., PubMed, MedlinePlus, CDC, Mayo Clinic, NIH):\n\n"
            f"{combinedtext}\n\nConsensus Summary:"
        )
        consensus = self.gemrefine(synthesisprompt)
        return consensus

    def combinerefinedconsensus(self, refined: str, consensus: str) -> str:
        combinedtext = f"Refined Answer:\n{refined}\n\nConsensus Summary:\n{consensus}"
        prompt = (
            "Compare the two texts provided below and extract the common verified information. "
            "Combine this information and refine it into a final, concise verified answer using proper medical terminology. "
            "Ensure factual accuracy by cross-referencing reliable medical databases:\n\n"
            f"{combinedtext}\n\nFinal Verified Answer:"
        )
        finalverified = self.gemrefine(prompt)
        return finalverified

    def addtohistory(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

# Streamlit UI
st.sidebar.title("Select Mode")
mode = st.sidebar.selectbox("Choose an application mode", ["Chatbot", "Patient Simulation"])

if mode == "Chatbot":
    st.title("MedAI")
    
    # Initialize chatbot only if API keys are available as environment variables
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("GOOGLE_API_KEY") and os.environ.get("LLAMA_API_KEY"):
        chatbot = MedAI()
    
        uploadedfile = st.file_uploader("Upload Health Record", type="pdf")
        if uploadedfile is not None:
            pdftext = chatbot.loadpdf(uploadedfile)
            st.session_state['pdftext'] = pdftext
        else:
            st.session_state['pdftext'] = ""
    
        query = st.text_input("Enter your medical query:")
        if st.button("Get Answer"):
            if not query:
                st.warning("Please enter a medical query.")
            else:
                chatbot.addtohistory("User", query)
                pdftext = st.session_state.get('pdftext', "")
    
                with st.spinner("Analyzing query..."):
                    analysis = chatbot.analyzequery(query)
                    complexity = analysis.get("complexity", False)
                    intent = analysis.get("intent", "information")
                    medical_terms_present = analysis.get("medical_terms_present", False)
                    parsedquery = analysis.get("parsed_query", query)
    
                with st.spinner("Generating primary answer..."):
                    if complexity or medical_terms_present or intent.lower() == "advice":
                        st.info("Detected complex or advice-seeking query. Using detailed model.")
                        primary = chatbot.gpto1(chatbot.primaryprompt(parsedquery, pdftext, analysis))
                    else:
                        st.info("Detected simple query. Using advanced reasoning model.")
                        primary = chatbot.gpt4(chatbot.primaryprompt(parsedquery, pdftext, analysis))
                    primaryclean = chatbot.cleantext(primary)
                    chatbot.addtohistory("AI", primaryclean)
                   # st.subheader("Primary Answer:")
                   # st.write(primaryclean)
    
                with st.spinner("Querying multiple models in parallel..."):
                    resultsdict = {}
                    with ThreadPoolExecutor(max_workers=6) as executor:
                        futures = {
                            'ChatGPT4': executor.submit(chatbot.gpt4, chatbot.primaryprompt(parsedquery, pdftext, analysis)),
                            'ChatGPTo1': executor.submit(chatbot.gpto1, chatbot.primaryprompt(parsedquery, pdftext, analysis)),
                            'Gemini': executor.submit(chatbot.gemini, chatbot.primaryprompt(parsedquery, pdftext, analysis)),
                            'Gemma': executor.submit(chatbot.gemma, chatbot.primaryprompt(parsedquery, pdftext, analysis)),
                            'DeepSeek': executor.submit(chatbot.deepseek, chatbot.primaryprompt(parsedquery, pdftext, analysis)),
                            'Llama': executor.submit(chatbot.llama, chatbot.primaryprompt(parsedquery, pdftext, analysis)),
                        }
                        for modelname, future in futures.items():
                            resultsdict[modelname] = future.result()
                    aggregated = chatbot.aggregate(resultsdict)
                    aggregatedclean = chatbot.cleantext(aggregated)
                   # st.subheader("Aggregated Answer:")
                   # st.write(aggregatedclean)
    
                   # st.subheader("Accuracy Check of Individual Models (using gemrefine):")
                    for model_name, answer in resultsdict.items():
                       # with st.spinner(f"Checking accuracy of {model_name}'s answer..."):
                            accuracy_check_prompt = (
                                f"Evaluate the factual accuracy of the following medical answer. "
                                f"Cross-reference with reliable medical sources and provide a brief assessment:\n\n"
                                f"{answer}\n\nAccuracy Assessment for {model_name}:"
                            )
                            accuracy_assessment = chatbot.gemrefine(accuracy_check_prompt)
                          #  st.markdown(f"**{model_name} Accuracy Assessment:**")
                          #  st.write(accuracy_assessment)
    
                with st.spinner("Refining aggregated answer..."):
                    refined = chatbot.refine(aggregatedclean)
                    refinedclean = chatbot.cleantext(refined)
                    st.subheader("Refined Answer:")
                    st.write(refinedclean)
    
                with st.spinner("Verifying refined answer against medical sources..."):
                    verificationresult = chatbot.verifyrefined(refinedclean, parsedquery)
                    verificationmatches = verificationresult.get("matches", [])
                    confidencescore = verificationresult.get("confidence", 0.0)
                    st.subheader("Refined Answer Accuracy (Confidence Score):")
                    st.write(f"{confidencescore:.2f}")
    
                    filtered_matches = []
                    for match in verificationmatches:
                        try:
                            response = chatbot.fetchurl(match['url'])
                            if "Could not find that page" not in response and "Page not found" not in response and "Page or document not found" not in response and "Error" not in response:
                                filtered_matches.append(match)
                            else:
                                st.warning(f"Filtered out URL due to 'could not find page' or error: {match['url']}")
                        except Exception as e:
                            st.error(f"Error checking URL {match['url']}: {e}")
    
                    if filtered_matches:
                        st.subheader("Verification Matches (Top results):")
                        for idx, match in enumerate(filtered_matches[:10], 1):
                            st.markdown(f"**Match {idx}:**")
                            st.write(f"Source URL: {match['url']}")
                            st.write(f"Similarity Score: {match['similarity']:.2f}")
                            st.write(f"Matching Sentence: {match['sentence']}")
                    else:
                        st.info("No valid verification matches found after filtering.")
    
                with st.spinner("Synthesizing verified information..."):
                    consensusverified = chatbot.synthesizeverifiedinfo(verificationmatches)
                    if consensusverified:
                        st.subheader("Verified Information:")
                        st.write(consensusverified)
                    else:
                        st.info("No verified information generated.")
    
                if refinedclean and consensusverified:
                    with st.spinner("Combining refined and verified information..."):
                        finalverified = chatbot.combinerefinedconsensus(refinedclean, consensusverified)
                        finalverifiedclean = chatbot.cleantext(finalverified)
                        st.subheader("Final Verified Information:")
                        st.write(finalverifiedclean)
                else:
                    st.info("Insufficient data to combine and refine for the final answer.")
    else:
        st.warning("API keys are not configured. Please set them as secrets in Streamlit Cloud.")

if mode == "Patient Simulation":
    def parse_transcript(transcript_text: str) -> dict:
    # Extract Chief Complaint from the transcript
    cc_match = re.search(r"Chief Complaint:\s*(.*)", transcript_text)
    chief_complaint = cc_match.group(1).strip() if cc_match else "Shortness of breath and swelling in my legs."
    
    # Extract History of Present Illness (HPI)
    hpi_match = re.search(r"History of Present Illness \(HPI\):\s*(.*?)\n\n", transcript_text, re.DOTALL)
    history_of_present_illness = hpi_match.group(1).strip() if hpi_match else transcript_text[:100]
    
    # Extract Past Medical History (PMH)
    pmh_match = re.search(r"Past Medical History.*?:\s*(.*?)\n\n", transcript_text, re.DOTALL)
    past_medical_history_text = pmh_match.group(1).strip() if pmh_match else ""
    past_medical_history = re.split(r'\n|\r', past_medical_history_text)
    past_medical_history = [line.strip() for line in past_medical_history if line.strip()]
    
    # Extract Medications
    med_match = re.search(r"Medications:\s*(.*?)\n\n", transcript_text, re.DOTALL)
    medications_text = med_match.group(1).strip() if med_match else ""
    meds_lines = medications_text.splitlines()
    medications = [re.sub(r"^\d+\.\s*", "", line).strip() for line in meds_lines if line.strip()]
    
    typical_responses = {
        "how are you feeling today?": "I'm feeling quite breathless today, and my legs are really swollen.",
        "can you describe your shortness of breath?": "It feels like I can't get enough air, especially when I try to lie flat.",
        "have you checked your weight recently?": "Yes, I've gained about 5 pounds in the last week.",
        "are you taking all your medications?": "Yes, I haven't missed any doses.",
        "any chest pain?": "No, no chest pain.",
    }
    
    return {
        "chief_complaint": chief_complaint,
        "history_of_present_illness": history_of_present_illness,
        "past_medical_history": past_medical_history,
        "medications": medications,
        "typical_responses": typical_responses
    }

st.title("Interactive AI Patient Simulation")

# Upload the clinical transcript (TXT file)
uploaded_file = st.file_uploader("Upload the clinical transcript (TXT file)", type=["txt"])
if uploaded_file is not None:
    transcript_text = uploaded_file.read().decode("utf-8")
    st.subheader("Transcript Content")
    st.text_area("Transcript", transcript_text, height=200)
    simulated_patient_case = parse_transcript(transcript_text)
else:
    # Default simulated patient case if no transcript is provided.
    simulated_patient_case = {
        "chief_complaint": "Shortness of breath and swelling in my legs.",
        "history_of_present_illness": "The patient reports increasing shortness of breath over the past week, especially when lying down. They also noticed swelling in their ankles and legs. They feel tired more easily.",
        "past_medical_history": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril", "Metformin"],
        "typical_responses": {
            "how are you feeling today?": "I'm feeling quite breathless today, and my legs are really swollen.",
            "can you describe your shortness of breath?": "It feels like I can't get enough air, especially when I try to lie flat.",
            "have you checked your weight recently?": "Yes, I've gained about 5 pounds in the last week.",
            "are you taking all your medications?": "Yes, I haven't missed any doses.",
            "any chest pain?": "No, no chest pain.",
        }
    }

# -----------------------------------------------------------------------------
# 2. Initialize or Retrieve Chat Session State
# -----------------------------------------------------------------------------
if "simulation_messages" not in st.session_state:
    st.session_state["simulation_messages"] = [{
        "role": "assistant",
        "content": f"Hello doctor, I'm here because of {simulated_patient_case['chief_complaint']}."
    }]

for msg in st.session_state.simulation_messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------------------------------------------------------
# 3. Chat Input and AI Simulation Response
# -----------------------------------------------------------------------------
prompt = st.chat_input(key="simulation_input")
if prompt:
    # Add user's prompt to conversation history.
    st.session_state.simulation_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Build conversation context from the chat history.
    conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.simulation_messages])
    
    # Construct the LLM prompt using the simulated case and conversation so far.
    llm_prompt = textwrap.dedent(f"""
        You are a patient with heart failure. Your chief complaint is {simulated_patient_case['chief_complaint']}".
        Your history includes: {', '.join(simulated_patient_case['past_medical_history'])}.
        You are currently taking: {', '.join(simulated_patient_case['medications'])}.
        
        Here is the conversation so far:
        {conversation_context}
        
        Respond to the last message as the patient would, drawing from your simulated details and typical responses. Be concise and realistic.
    """)
    
    # Retrieve the OpenAI API key from secrets.
    openai_api_key_simulation = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key_simulation:
        st.error("OpenAI API key is required for the simulation.")
        st.stop()
    
    # Initialize OpenAI client.
    client_simulation = OpenAI(api_key=openai_api_key_simulation)
    
    try:
        response = client_simulation.chat.completions.create(
            model="gpt-4o",  # Change to your preferred model if needed.
            messages=[
                {"role": "system", "content": "You are a patient in a medical simulation."},
                {"role": "user", "content": llm_prompt},
            ]
        )
        ai_response = response.choices[0].message.content
        st.session_state.simulation_messages.append({"role": "assistant", "content": ai_response})
        st.chat_message("assistant").write(ai_response)
    except Exception as e:
        st.error(f"An error occurred in the simulation: {e}")
