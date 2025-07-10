from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveChracterTextSplitter
from langchain_openai import OpenAIEMbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def qa_agent(open_api_key,memory,uploaded_file,question):
    model=ChatOpenAI(model="gpt-3.5-turbo",open_api_key=open_api_key)
    file_content=uploaded_file.read()
    temp_file_path="temp.pdf"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)
    loader=PyPDFLoader(temp_file_path)
    docs=loader.load()
    text_splitter=RecursiveChracterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separator=['\n','。','!',"？"]
    )
    texts=text_splitter.split_documents(docs)
    embedding_model=OpenAIEMbeddings()
    db=FAISS.from_documents(texts,embedding_model)
    retriever=db.as_retriever()
    qa=ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    reponse=qa.invoke({"question":question,"chat_history":memory})
    return reponse
