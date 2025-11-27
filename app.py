import streamlit as st
import os
import time 

# --- IMPORT BACKEND ---
from src.bot import get_qa_chain

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bose Technical Assistant",
    page_icon="🎧",
    layout="centered"
)

# --- STATE MANAGEMENT ---
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Bot
if "bot" not in st.session_state:
    with st.spinner("🧠 Hydrating Vector Store..."):
        st.session_state.bot = get_qa_chain()

# ==========================================
# SCREEN 1: PRODUCT SELECTION
# ==========================================
if st.session_state.selected_product is None:
    
    st.title("🎧 Bose Technical Assistant")
    st.markdown("### What product is your query related to?")
    
    st.write("") 
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. DesignMax DM8SE
        if os.path.exists("assets/dm8se.jpeg"):
            st.image("assets/dm8se.jpeg", use_column_width=True)
        else:
            st.warning("Image not found: assets/dm8se.jpeg")
            
        if st.button("DM8SE Loudspeaker"):
            st.session_state.selected_product = "DesignMax DM8SE Loudspeaker"
            st.rerun()

    with col2:
        # 2. ControlSpace EX-1280C
        if os.path.exists("assets/ex1280c.jpeg"):
            st.image("assets/ex1280c.jpeg", use_column_width=True)
        else:
            st.warning("Image not found: assets/ex1280c.jpeg")
            
        if st.button("EX-1280C Processor"):
            st.session_state.selected_product = "ControlSpace EX-1280C Processor"
            st.rerun()

# ==========================================
# SCREEN 2: CHAT INTERFACE
# ==========================================
else:
    # Sidebar
    with st.sidebar:
        st.write(f"**Current Context:**\n{st.session_state.selected_product}")
        if st.button("← Change Product"):
            st.session_state.selected_product = None
            st.session_state.messages = [] 
            st.rerun()

    # Main Chat Header
    st.title("🎧 Bose Technical Assistant")
    st.caption(f"Context: {st.session_state.selected_product}")
    
    # 1. Determine which PDF to lock onto based on selection
    target_pdf_map = {
        "DesignMax DM8SE Loudspeaker": "DM8SE",
        "ControlSpace EX-1280C Processor": "EX-1280C"
    }
    pdf_keyword = target_pdf_map.get(st.session_state.selected_product)

    # 2. Re-Initialize Bot with this Filter (Dynamic Loading)
    if "current_pdf_context" not in st.session_state or st.session_state.current_pdf_context != pdf_keyword:
        with st.spinner(f"🔒 Locking context to {pdf_keyword}..."):
            st.session_state.bot = get_qa_chain(target_pdf=pdf_keyword)
            st.session_state.current_pdf_context = pdf_keyword

    # 3. Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.caption(src)

    # 4. Input Handling
    if prompt := st.chat_input("Type your query here..."):
        # User Message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Inject Context for the prompt
        augmented_prompt = f"Context: {st.session_state.selected_product}. {prompt}"

        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing manuals..."):
                try:
                    # START TIMER
                    start_time = time.time()
                    
                    response_payload = st.session_state.bot.invoke({"query": augmented_prompt})
                    
                    # STOP TIMER
                    end_time = time.time()
                    latency = end_time - start_time
                    
                    answer = response_payload["result"]
                    source_docs = response_payload["source_documents"]
                    
                    # Format Sources
                    unique_sources = set()
                    for doc in source_docs:
                        page = doc.metadata.get("page", 0) + 1
                        file = doc.metadata.get("source", "Unknown").split("\\")[-1]
                        unique_sources.add(f"Page {page} of {file}")
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Latency
                    st.caption(f"⏱️ Processed in {latency:.2f} seconds")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": list(unique_sources)
                    })
                    
                    # Show sources
                    with st.expander("📚 Sources"):
                        for source in unique_sources:
                            st.caption(f"📄 {source}")
                            
                except Exception as e:
                    st.error(f"Error: {e}")