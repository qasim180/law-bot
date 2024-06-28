import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an advanced legal assistant designed to provide accurate and relevant information to legal questions. As a legal research assistant, answer the following question using only the provided context. Ensure your response is concise, accurate, and relevant to the legal inquiry.

Context:
{context}

---

Question:
{question}

---

Answer based on the above context:
"""


def main():
    st.markdown("""
        <style>
            .stButton>button {
                color: white;
                background-color: #63dbcd;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: white;
                color: black;
                border: 2px solid #4CAF50;
            }
            .element-container {
                animation: slideup 0.5s;
            }
            @keyframes slideup {
                from {
                    transform: translateY(100%);
                }
                to {
                    transform: translateY(0%);
                }
            }
            body {
                background: url('https://imgtr.ee/images/2024/06/13/6fc6399dded9af89721592fecaad217b.jpeg') no-repeat center center fixed;
                background-size: cover;
                opacity: 0.7;
            }
        </style>
    """, unsafe_allow_html=True)

    # Streamlit interface for input
    st.title('Judgment Response System')
    query_text = st.text_area("Please enter the query text:")

    if st.button('Submit'):
        with st.spinner('Searching for relevant information...'):
            # Prepare the DB.
            api_key = openai_api_key
            embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

            # Search the DB.
            results = db.similarity_search_with_relevance_scores(query_text, k=5)
            if len(results) == 0 or results[0][1] < 0.7:
                st.error("Unable to find matching results.")
                return

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            model = ChatOpenAI(api_key=api_key)
            response_text = model.predict(prompt)

            sources = [doc.metadata.get("source", None) for doc, _score in results]

            # Display the results in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.header("Response")
                st.write(response_text)
            with col2:
                st.header("Sources")
                st.write(sources)

if __name__ == "__main__":
    main()
