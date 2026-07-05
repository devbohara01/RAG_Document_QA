import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(folder_path="data"):
    """
    Load all PDF files from the given folder
    and split them into chunks.
    """

    all_documents = []

    # Read every PDF in data folder
    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            print(f"📄 Loading: {file}")

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            all_documents.extend(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_documents)

    print(f"\n✅ Total Pages : {len(all_documents)}")
    print(f"✅ Total Chunks: {len(chunks)}")

    return chunks