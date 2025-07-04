# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
app = Flask(__name__, static_folder='.') 
CORS(app)



# google_api_key = os.getenv("GOOGLE_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY") 
#if not google_api_key:
#   print("Warning: GOOGLE_API_KEY is not set. LLM calls will fail.")
#

try:
    pdf_path = "DATA BASE MANAGEMENT SYSTEMS NOTES.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = FAISS.from_texts(texts, embeddings)

    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    print("PDF processed and QA chain initialized successfully.")

except Exception as e:
    print(f"Error during PDF processing or QA chain setup: {e}")
 

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    if qa_chain is None:
        return jsonify({"error": "Backend not initialized. Check server logs for errors."}), 500

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        print(f"Received query: {query}")
        result = qa_chain.run(query)
        print(f"Answer: {result}")
        return jsonify({"answer": result})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
