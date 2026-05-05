import streamlit as st
from prediction_module import get_full_response

st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥")

st.title("🧑‍⚕️ Healthcare Chatbot")
st.write("Enter your symptoms in simple English.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Describe symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            res = get_full_response(prompt)

            if res.get("error"):
                st.warning(res["message"])
            else:
                st.markdown(f"### Disease : {res['disease']}")
                st.markdown(f"**Confidence :** {res['confidence']:.2f}%")
                st.markdown(f"**Description :** {res['description']}")
                st.markdown(f"**Precautions :** {res['precautions']}")
                st.markdown(f"**Detected Symptoms :** {', '.join(res['detected_symptoms'])}")