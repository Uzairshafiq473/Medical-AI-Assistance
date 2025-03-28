import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import re
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_length=512,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"device_map": "auto"}
    )

def format_response(response):
    """Format the answer and sources in a user-friendly way with numbered points"""
    result = response["result"]
    sources = response["source_documents"]
    
    # Split the answer by lines and clean up existing numbering
    answer_lines = [line.strip() for line in result.split('\n') if line.strip()]
    
    formatted_answer = "**Answer:**\n"
    
    for i, line in enumerate(answer_lines, 1):
        # Remove any existing numbering at the start of the line
        line = re.sub(r'^\d+\.\s*', '', line)
        formatted_answer += f"{i}. {line}\n"
    
    formatted_sources = "\n**Sources:**\n"
    
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get('source', 'Unknown source').split('\\')[-1]
        page = doc.metadata.get('page', 'N/A')
        formatted_sources += f"{i}. {source} (Page {page})\n"
    
    return formatted_answer + formatted_sources

def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="💊")
    st.title("💊 Medical Chatbot")
    st.markdown("Ask me about medical conditions and treatments!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your medical question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query
        with st.spinner("Searching medical knowledge..."):
            try:
                HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                HF_TOKEN = os.environ.get("HF_TOKEN")

                vectorstore = get_vectorstore()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt()}
                )

                response = qa_chain.invoke({'query': prompt})
                formatted_response = format_response(response)

            except Exception as e:
                formatted_response = f"❌ Error: {str(e)}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(formatted_response, unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    main()