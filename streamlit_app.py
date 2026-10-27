import streamlit as st
import os
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from src import llm
from src import retriever
from src import scraper
from src import build_kb

# Page config
st.set_page_config(
    page_title="PlacidWay Medical Tourism Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "lead_captured" not in st.session_state:
    st.session_state.lead_captured = False
if "show_lead_form" not in st.session_state:
    st.session_state.show_lead_form = False
if "log_file" not in st.session_state:
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    st.session_state.log_file = os.path.join(log_dir, f"unanswered_{datetime.now().strftime('%Y%m%d')}.jsonl")

@st.cache_resource
def load_retriever():
    kb_path = os.path.join(BASE_DIR, "data", "kb_index.json")
    if not os.path.exists(kb_path):
        st.error("Knowledge base not found. Please run scraper.py and build_kb.py first.")
        st.stop()
    return retriever.Retriever(kb_path)

def log_unanswered(question, session_id):
    """Log questions the bot could not answer"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answered": False
        }
        with open(st.session_state.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        pass

def log_answered(question, session_id, had_sources):
    """Log answered questions"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answered": True,
            "had_sources": had_sources
        }
        log_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        answered_file = os.path.join(log_dir, f"answered_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(answered_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        pass

def refresh_content():
    """Refresh bot content"""
    try:
        with st.spinner("Scraping pages..."):
            scraper.main()
        with st.spinner("Building knowledge base..."):
            build_kb.main()
        st.cache_resource.clear()
        st.success("Content refreshed successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error refreshing: {e}")

def main():
    st.title("🏥 PlacidWay Medical Tourism Chatbot")
    st.markdown("*Grounded answers from authorized PlacidWay pages only - Answers in any language*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Refresh Content", use_container_width=True):
            refresh_content()
        
        st.divider()
        
        st.header("💬 Chat Options")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.header("📊 Features")
        st.markdown("""
        - ✅ Content from 7 authorized URLs
        - ✅ LangChain RAG with citations
        - ✅ Multilingual support
        - ✅ Follow-up questions with history
        - ✅ Honest "I don't know"
        - ✅ Cost questions with currency
        - ✅ Medical disclaimer when relevant
        - ✅ Lead capture
        - ✅ Question logging
        """)
        
        st.divider()
        st.caption("Powered by LangChain + OpenAI")
    
    # Load retriever
    try:
        ret = load_retriever()
    except Exception as e:
        st.error(f"Error loading retriever: {e}")
        st.stop()
    
    # Lead capture form
    if st.session_state.show_lead_form and not st.session_state.lead_captured:
        with st.container():
            st.info("📧 Want a free quote from PlacidWay? Leave your details below.")
            with st.form("lead_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                submitted = st.form_submit_button("Submit")
                if submitted and name and email:
                    lead_data = {
                        "timestamp": datetime.now().isoformat(),
                        "name": name,
                        "email": email,
                        "phone": phone,
                        "session_id": st.session_state.session_id
                    }
                    lead_dir = os.path.join(BASE_DIR, "leads")
                    os.makedirs(lead_dir, exist_ok=True)
                    lead_file = os.path.join(lead_dir, f"leads_{datetime.now().strftime('%Y%m%d')}.jsonl")
                    with open(lead_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(lead_data) + "\n")
                    st.session_state.lead_captured = True
                    st.session_state.show_lead_form = False
                    st.success("Thank you! We'll contact you soon.")
                    st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in message["sources"]:
                            st.markdown(f"- [{src['title']}]({src['url']})")
                if message.get("show_disclaimer"):
                    st.caption("⚠️ Medical Disclaimer: This information is for educational purposes only and is not medical advice. Always consult with qualified healthcare professionals.")
    
    # Chat input
    if prompt := st.chat_input("Ask in any language about ITC treatments, packages, prices..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                # Rewrite query with history for follow-ups
                history = []
                for msg in st.session_state.messages[:-1]:  # Exclude current
                    history.append({"role": msg["role"], "content": msg["content"]})
                
                search_q = llm.rewrite_query(history, prompt)
                hits = ret.search(search_q, top_k=6)
                chunks = [c for c, _ in hits]
                
                result = llm.answer(chunks, prompt, history)
                
                answer = result["answer"]
                sources = result.get("sources", [])
                
                st.markdown(answer)
                
                # Check if needs disclaimer
                medical_keywords = ['treatment', 'therapy', 'cancer', 'medicine', 'drug', 'dose', 'combine', 'hormone']
                show_disclaimer = any(kw in prompt.lower() or kw in answer.lower() for kw in medical_keywords)
                if show_disclaimer:
                    st.caption("⚠️ Medical Disclaimer: This information is for educational purposes only and is not medical advice. Always consult with qualified healthcare professionals.")
                
                # Check if should show lead capture
                if any(word in prompt.lower() for word in ['quote', 'appointment', 'book', 'contact', 'price']) and not st.session_state.lead_captured:
                    if st.button("📧 Get Free Quote - Leave Details"):
                        st.session_state.show_lead_form = True
                        st.rerun()
                
                if sources:
                    with st.expander("📚 Sources"):
                        for src in sources:
                            st.markdown(f"- [{src['title']}]({src['url']})")
                
                # Log
                if not sources or "don't know" in answer.lower() or "i don't know" in answer.lower():
                    log_unanswered(prompt, st.session_state.session_id)
                else:
                    log_answered(prompt, st.session_state.session_id, bool(sources))
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "show_disclaimer": show_disclaimer
                })

if __name__ == "__main__":
    main()
