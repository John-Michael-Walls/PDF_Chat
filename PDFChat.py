import os
import pdfplumber
import pickle
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
#from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import PromptLayerOpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import promptlayer
import datetime

#promptlayer.api_key = os.getenv('PROMPTLAYER_API_KEY') # if required
openai_api_key = os.getenv('OPENAI_API_KEY')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

page_separator = "NEW_PAGE"

def app():
   
    st.markdown(
        """
        <style>
        .stApp {
            background-color: lightblue;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.title("PDF Chat Bot")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = pdf.pages
            saved_file = page_separator.join([page.extract_text() for page in pages if page.extract_text()])
        
        if not saved_file.strip():
            st.write("There is no text to show. Check the PDF file.")
            return

        filename = uploaded_file.name.split(".")[0] + ".pkl"
        fileoutput = uploaded_file.name.split(".")[0] 

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                vectors = pickle.load(f)
        else:
            text_splitter = CharacterTextSplitter(
                separator=page_separator,
                chunk_size=200,
                chunk_overlap=150,
                length_function=len,
            )
            texts = text_splitter.split_text(saved_file)

            embeddings = OpenAIEmbeddings()
            vectors = FAISS.from_texts(texts, embeddings)

            with open(filename, "wb") as f:
                pickle.dump(vectors, f)
        
       

        temperature = st.slider("Choose which temperature to use:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        model_choice = st.selectbox("Choose which openai model to use:", options=["gpt-4", "gpt-3.5-turbo"])

        st.write("Ask your questions of the PDF below:")

        user_question = st.text_input("Your question:")


        submit_button = st.button("Submit")
        
        if submit_button and user_question:
            model = ChatOpenAI(model=model_choice, temperature=temperature)
            qa = ConversationalRetrievalChain.from_llm(model, retriever=vectors.as_retriever())
            result = qa({"question": user_question, "chat_history": st.session_state.chat_history})
            answer = result['answer']
            st.session_state.chat_history.append((user_question, answer))
          
        for idx, (user_question, answer) in enumerate(st.session_state.chat_history, 1):
            st.write(f"Q{idx}: {user_question}")
            st.write(f"A{idx}: {answer}")
            st.write("---")

        chat_history_str = "\n".join([f"Q{idx}: {question}\nA{idx}: {answer}\n---\n" for idx, (question, answer) in enumerate(st.session_state.chat_history, 1)])
        chat_history_str = f"PDF Name:  {fileoutput}. Date created: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n" + chat_history_str
        st.download_button(
            label="Download chat history as .txt file",
            data=chat_history_str.encode('utf-8'),
            file_name="chat_history.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    app()
