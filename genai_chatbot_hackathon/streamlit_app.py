import streamlit as st
import requests

st.title("GenAI KB Chatbot 💻")

issue = st.text_area("Describe the issue:")
dry_run = st.checkbox("Dry Run", True)

if st.button("Run Chatbot"):
    with st.spinner("Fetching resolution..."):
        res = requests.post("http://localhost:5000/chat", json={"issue": issue, "dry_run": dry_run})
        if res.ok:
            data = res.json()
            st.success("Resolution Found")
            st.text_area("KB Used", data['kb_used'], height=150)
            st.text_area("Commands Extracted", data['commands'], height=150)
            st.text_area("Execution Result", data['execution_result'], height=200)
        else:
            st.error(res.text)