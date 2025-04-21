# Libraries
import os
import re
import io
import datetime
import locale
import base64
import requests
import json
import streamlit as st
import pandas as pd

# optional readability scoring
try:
    import textstat
except ImportError:
    textstat = None

# PDF and Word export
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

try:
    from docx import Document
except ImportError:
    Document = None

# Voice transcription
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# --- Text extraction ---
def extract_text_from_file(file) -> str:
    if file.type == "application/pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            st.error("PDF support requires PyPDF2. Please `pip install PyPDF2`.")
            return ""
        try:
            reader = PdfReader(file)
        except Exception:
            import io as _io
            reader = PdfReader(_io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    else:
        data = file.read()
        for enc in ("utf-8","latin-1"): 
            try:
                return data.decode(enc)
            except:
                pass
        return ""

# --- UI setup ---
st.set_page_config(page_title="Discharge Summary Simplifier")
st.title("Discharge Summary Simplifier")
st.subheader("Transforming Patient Understanding with AI")
st.write("Upload or paste a hospital discharge summary and let the AI simplify it.")

# --- Offline caching & run flag ---
for key, default in [
    ("cached_summary", None),
    ("cached_sections", {}),
    ("cached_concise", None),
    ("run_summary", False),
    ("symptoms", []),
    ("faq_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- API key handling ---
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your OpenRouter API Key", type="password")
if not api_key and st.session_state["cached_summary"]:
    st.info("Offline mode: Using cached summary.")
elif not api_key:
    st.info("Please provide your OpenRouter API key.")
    st.stop()

# --- Input mode selector ---
mode = st.radio(
    "How provide the discharge text?",
    ("Enter text", "Upload file", "Voice note"),
    horizontal=True
)
if "discharge_text" not in st.session_state:
    st.session_state["discharge_text"] = ""

if mode == "Enter text":
    curr = st.session_state["discharge_text"]
    h = min(max((curr.count("\n")+1)*25, 200), 800)
    st.text_area("Discharge Instructions Text:", height=h, key="discharge_text")
elif mode == "Upload file":
    f = st.file_uploader("Upload Discharge Summary (TXT/PDF)", type=["txt","pdf"])
    if f:
        txt = extract_text_from_file(f).strip()
        if not txt:
            st.error("Invalid file.")
        else:
            st.session_state["discharge_text"] = txt
else:
    audio = st.file_uploader("Upload voice note (mp3/wav)", type=["mp3","wav"])
    if audio and sr:
        r = sr.Recognizer()
        with sr.AudioFile(audio) as src:
            aud = r.record(src)
        try:
            txt = r.recognize_google(aud)
            st.session_state["discharge_text"] = txt
        except:
            st.warning("Transcription failed.")
    elif audio:
        st.info("Install speech_recognition for transcription.")

discharge_text = st.session_state["discharge_text"]
if discharge_text:
    st.markdown("#### Original Text")
    st.write(discharge_text)
else:
    st.info("Provide text above, then click ‘Simplify Discharge Instructions’")
    st.stop()

# --- Sidebar & options ---
font_size = st.sidebar.slider("Font size", 12,24,16)
high_contrast = st.sidebar.checkbox("High Contrast Mode")
if high_contrast:
    st.markdown(
        """
        <style>
        /* Root containers */
        html, body, .stApp, [data-testid="stSidebar"] {
            background-color: #000 !important;
            color: #fff !important;
        }
        /* All text elements */
        .stApp * {
            color: #fff !important;
            background-color: transparent !important;
        }
        /* Inputs, buttons, sliders, text areas */
        .stButton button,
        .stDownloadButton button,
        .stTextInput>div>input,
        .stTextArea>div>textarea,
        .stSlider>div {
            background-color: #333 !important;
            color: #fff !important;
            border: 1px solid #555 !important;
        }
        /* Links */
        a {
            color: #0ff !important;
        }
        /* Tables and dataframes */
        .stDataFrame, .stTable {
            background-color: #000 !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
st.markdown(f"<style>*{{font-size:{font_size}px}}</style>", unsafe_allow_html=True)
user_locale = locale.getdefaultlocale()[0] or ""
af_lang = user_locale.split("_")[0].capitalize()
if af_lang not in ["English","Spanish","Chinese","French","German"]:
    af_lang = None
reading_level = st.sidebar.slider("Target Reading Level (US Grade)", min_value=3, max_value=12, value=6)
language = st.sidebar.selectbox("Output Language", ["English", "Spanish", "Chinese", "French", "German"], index=0)

# --- Glossary ---
glossary = {
    "otoscope exam": "An exam where a doctor looks inside your ear",
    "mri": "Magnetic Resonance Imaging scan"
}
def apply_tooltips(line:str)->str:
    for term,defi in glossary.items():
        line = re.sub(rf"\b({re.escape(term)})\b",
                      rf"<span title='{defi}' style='border-bottom:1px dotted;'>\1</span>",
                      line, flags=re.IGNORECASE)
    return line

# --- LLM calls ---
def generate_concise_summary(text:str, lang:str)->dict:
    prompt = (
        f"Simplify the following discharge instructions into a concise, include all essential "
        f"information related to the patient especially the diagnosis and the reason, patient-friendly overview.\n"
        f"Output only a short paragraph (no sections):\n\n\"\"\"{text}\"\"\""
    )
    payload = {"model":"deepseek/deepseek-r1",
               "messages":[{"role":"user","content":prompt}],
               "temperature":0.0,"top_p":1.0}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                         json=payload)
    return {"status":resp.status_code,
            "raw_json":resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text}

def summarize_discharge(text:str, lvl:int, lang:str)->dict:
    prompt = (
        f"The following is a hospital discharge summary. Simplify it to a {lvl}th-grade reading level in {lang}.\n"
        "Break into sections with these headings:\n"
        "- **Simplified Instructions:** bullet-points\n"
        "- **Importance:** why each instruction matters\n"
        "- **Follow-Up Appointments or Tasks:** tasks/visits\n"
        "- **Medications:** with simple dosing notes\n"
        "- **Precautions:** warning signs, activities to avoid\n"
        "- **References:** brief reasons/explanations\n\n"
        f"Now simplify:\n\"\"\"{text}\"\"\""
    )
    payload = {"model":"deepseek/deepseek-r1",
               "messages":[{"role":"user","content":prompt}],
               "temperature":0.0,"top_p":1.0}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                         json=payload)
    return {"status":resp.status_code,
            "raw_json":resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text}

def generate_ics(evt:str)->str:
    dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nSUMMARY:{evt}\nDTSTART:{dt}\nEND:VEVENT\nEND:VCALENDAR"

# --- Main action ---
if st.button("Simplify Discharge Instructions"):
    with st.spinner("Simplifying Discharge Instructions…"):
        # concise
        if not api_key and st.session_state["cached_concise"]:
            concise = st.session_state["cached_concise"]
        else:
            r = generate_concise_summary(discharge_text, language)
            if r["status"] != 200:
                st.error(f"API {r['status']}"); st.stop()
            concise = r["raw_json"]["choices"][0]["message"]["content"].strip()
            st.session_state["cached_concise"] = concise

        # detailed (cache, but hidden)
        if not api_key and st.session_state["cached_summary"]:
            pass
        else:
            r2 = summarize_discharge(discharge_text, reading_level, language)
            if r2["status"] != 200:
                st.error(f"API {r2['status']}"); st.stop()
            detailed = r2["raw_json"]["choices"][0]["message"]["content"].strip()
            st.session_state["cached_summary"] = detailed
            # parse into sections
            sections = {
                "Simplified Instructions": [],
                "Importance": [],
                "Follow-Up Appointments or Tasks": [],
                "Medications": [],
                "Precautions": [],
                "References": []
            }
            
            # compile flexible header patterns (now stripping trailing ** too)
            header_patterns = {
                name: re.compile(
                    rf"^\s*\**\s*{re.escape(name)}\s*:?\s*\**\s*$",
                    re.IGNORECASE
                )
                for name in sections
            }
            
            current = None
            for line in detailed.splitlines():
                stripped = line.strip()
                text = re.sub(r"^[\-\*\s]+", "", stripped)
            
                # header detection
                matched = False
                for sec_name, pat in header_patterns.items():
                    if pat.match(text):
                        current = sec_name
                        matched = True
                        break
                if matched:
                    continue
            
                # content lines
                if current and text:
                    clean_text = re.sub(r"[:\*]+$", "", text).strip()
                    sections[current].append(clean_text)
            
            # fallback if no sections found
            if all(len(items) == 0 for items in sections.values()):
                sections["Simplified Instructions"] = [
                    ln.strip() for ln in detailed.splitlines() if ln.strip()
                ]
            
            st.session_state["cached_sections"] = sections
        st.session_state["run_summary"] = True

# --- Persistent display ---
if st.session_state["run_summary"]:
    concise = st.session_state["cached_concise"]
    sections = st.session_state["cached_sections"]

    # Simplified Summary
    st.markdown("---")
    st.subheader("Simplified Summary")
    for ln in concise.splitlines():
        st.markdown(apply_tooltips(ln), unsafe_allow_html=True)

    # --- Categorization & Actions ---
    st.markdown("---")
    st.subheader("Categorization & Actions")
    
    # 1) Render each section without leftover "**" or bullets
    for h, its in sections.items():
        if not its:
            continue
    
        # Clean header (strip any stray asterisks/colons)
        header_clean = re.sub(r"[:\*]+$", "", h).strip()
        st.markdown(f"**{header_clean}**")
    
        # Render items
        for raw in its:
            # strip bullets/spaces
            item = re.sub(r'^[\u2022\-\*\s]+', '', raw)
            # strip Task: prefix
            item = re.sub(r'^(Task:?\**)\s*', '', item, flags=re.IGNORECASE)
            # strip trailing colons or stars
            item = re.sub(r'[:\*]+$', '', item).strip()
            st.markdown(f"- {apply_tooltips(item)}", unsafe_allow_html=True)
    
            # Follow‑Up calendar buttons (and strip any leading "Task:" prefix)
            if header_clean == "Follow‑Up Appointments or Tasks":
                for raw in its:
                    # 1) remove bullets/spaces
                    fu = re.sub(r'^[\u2022\-\*\s]+', '', raw)
                    # 2) remove leading "Task:" or "Task:**"
                    fu = re.sub(r'^(Task:?\**)\s*', '', fu, flags=re.IGNORECASE)
                    # 3) strip any trailing colons or stars
                    fu = re.sub(r'[:\*]+$', '', fu).strip()
                    ics = generate_ics(fu)
                    st.download_button(
                        label=f"Add '{fu}' to Calendar",
                        data=ics,
                        file_name="event.ics",
                        mime="text/calendar"
                    )
    
    # 2) Medication checklist only once, after loop
    meds = sections.get("Medications", [])
    if meds:
        st.markdown("")  # spacing
        st.subheader("Medication Checklist & Reminders")
        for raw in meds:
            m = re.sub(r"^[\u2022\-\*\s]+", "", raw)
            m = re.sub(r"[:\*]+$", "", m).strip()
            st.checkbox(m, key=f"med_{m}")
        if st.button("Schedule Med Reminders", key="med_reminders_btn"):
            st.success("Medication reminders scheduled!")
    
    # 3) Overall reading level once at the end
    if textstat:
        all_text = " ".join(
            re.sub(r"[:\*]+$", "", re.sub(r"^[\u2022\-\*\s]+","", itm)).strip()
            for items in sections.values() for itm in items
        )
        grade = textstat.flesch_kincaid_grade(all_text)
        st.markdown(f"*Overall reading level: {grade:.1f}th grade*")
        
    # --- Symptom Tracker with auto‑extracted symptoms ---
    st.markdown("---")
    st.subheader("Symptom Tracker")
    
    # 1) extract symptom keywords from the text
    COMMON_SYMPTOMS = ["pain", "swelling", "fever", "nausea", "headache", "dizziness", "fatigue"]
    found = [
        term for term in COMMON_SYMPTOMS
        if re.search(rf"\b{re.escape(term)}\b", discharge_text, flags=re.IGNORECASE)
    ]
    
    if found:
        st.markdown("**Detected symptoms in summary:** " + ", ".join(found).capitalize())
        selected = st.multiselect(
            "Select symptoms to log",
            options=[s.capitalize() for s in found],
            default=[s.capitalize() for s in found],
            key="selected_symptoms"
        )
    else:
        st.info("No common symptoms detected in the summary.")
        selected = []
    
    # 2) date input
    d = st.date_input("Date", datetime.date.today(), key="symp_date")
    
    # 3) sliders for each selected symptom
    levels = {}
    for sym in selected:
        key = f"level_{sym.lower()}"
        levels[sym] = st.slider(
            f"{sym} level",
            min_value=0,
            max_value=10,
            value=0,
            key=key
        )
    
    # 4) log button
    if st.button("Log Symptoms", key="log_symptoms_btn"):
        entry = {"date": str(d)}
        for sym, lvl in levels.items():
            entry[sym.lower()] = lvl
        st.session_state["symptoms"].append(entry)
        st.success("Symptoms logged!")
    
    # 5) show existing log
    if st.session_state["symptoms"]:
        df = pd.DataFrame(st.session_state["symptoms"])
        st.dataframe(df)
    # timeline
    if st.checkbox("Show Recovery Timeline", key="show_timeline"):
        if st.session_state["symptoms"]:
            df=pd.DataFrame(st.session_state["symptoms"])
            df["date"]=pd.to_datetime(df["date"])
            st.line_chart(df.set_index("date")[["pain","swelling"]])
        else:
            st.info("No symptom data.")
    # download log
    if st.session_state["symptoms"]:
        csv = pd.DataFrame(st.session_state["symptoms"]).to_csv(index=False)
        st.download_button("Download Symptom Log", csv, "symptoms.csv","text/csv", key="dl_symptoms")

    # Send Feedback
    st.markdown("---")
    st.subheader("Send Feedback to Provider")
    fb = st.text_area("Your message", key="feedback_msg")
    if st.button("Send Message", key="send_feedback"):
        fb = st.session_state["feedback_msg"]
        mailto_fb = (
            f"mailto:hamzapiracha@live.com"
            f"?subject={requests.utils.quote('Patient Feedback')}"
            f"&body={requests.utils.quote(fb)}"
        )
        st.markdown(f"[Click here to send feedback]({mailto_fb})")

    # --- Emergency Contacts ---
    st.markdown("---")
    st.subheader("Emergency Contacts")
    
    with st.form("emergency_form"):
        ec_name = st.text_input("Contact Name", key="ec_name_input")
        ec_num  = st.text_input("Contact Number", key="ec_num_input")
        save_ec = st.form_submit_button("Save Contact")
        if save_ec:
            st.session_state["emergency"] = {"name": ec_name, "number": ec_num}
            st.success("Emergency contact saved")
    
    # Always show the saved contact below
    if "emergency" in st.session_state:
        em = st.session_state["emergency"]
        st.markdown(f"[Call {em['name']}](tel:{em['number']})")

    # ——— Export All Data as JSON ———
    st.markdown("---")
    st.subheader("Export All Data as JSON")
    data_export = {
        "discharge_text": discharge_text,
        "concise_summary": st.session_state["cached_concise"],
        "sections": st.session_state["cached_sections"],
        "symptoms": st.session_state["symptoms"],
        "feedback_messages": st.session_state["faq_log"],
        "emergency_contact": st.session_state.get("emergency", {}),
    }
    # display it in‑page
    st.markdown("#### JSON Preview")
    st.json(data_export)
    # download button
    st.download_button(
        label="Download All Data (JSON)",
        data=json.dumps(data_export, indent=2),
        file_name="app_data.json",
        mime="application/json",
    )
    if st.button("Clear All Data", key="clear_data"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k,None)
        st.success("Cleared")
