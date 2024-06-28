from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import pdfplumber
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# from unstructured.document_loaders import PDFLoader
CHROMA_PATH = "chroma"
DATA_PATH = "data/pdfs"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)




# DATA_PATH = "data/pdfs"
#
def load_documents():
    # Find all PDF files in the directory
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    documents = []

    # Iterate over each file and extract text
    for filename in pdf_files:
        path = os.path.join(DATA_PATH, filename)
        try:
            with pdfplumber.open(path) as pdf:
                full_text = ''
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"  # Extract text and add a newline after each page
                
                # Create a Document object for each PDF file
                document = Document(
                    page_content=full_text,
                    metadata={"source": filename}
                )
                documents.append(document)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return documents



def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(api_key=openai_api_key), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
