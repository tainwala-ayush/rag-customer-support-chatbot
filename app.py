import streamlit as st
import time
from rag import convertFile, semanticSearch, buildPrompt, ask_bot
from sentence_transformers import SentenceTransformer
embeddingModel = SentenceTransformer("BAAI/bge-small-en-v1.5")

st.title("AI CHAT BOT")

uploaded_files = st.file_uploader("Upload the files here!", type=".txt", accept_multiple_files=True)
query = st.text_input("Enter your query here: ")
if st.button("Ask"):
    all_chunks, metadata = convertFile(uploaded_files)
    text_embeddings = embeddingModel.encode(all_chunks, normalize_embeddings=True)
    query_embeddings = embeddingModel.encode([query], normalize_embeddings=True)
    st.session_state["chunks"] = all_chunks
    st.session_state["metadata"] = metadata
    st.session_state["textEmb"] = text_embeddings
    
    results, maxScore = semanticSearch(all_chunks, metadata, text_embeddings, query_embeddings)
    answer = ask_bot(query, results, maxScore)

    if isinstance(answer, tuple):
        response, sim, filename, text = answer
    else:
        response = answer
        sim, filename, text = [], [], []
        
    with st.spinner("Thinking..."):
        time.sleep(2)
        st.write(response)
    with st.expander("See context:"):
        i = 1
        for f,s,t in zip(filename, sim, text):
            st.write(f"CHUNK #{i}")
            st.write(f"Filename: {f} with a similarity of {s*100:.2f}%")
            i+=1
            with st.expander("View the retrieved chunks:"):
                st.write(t[:300])