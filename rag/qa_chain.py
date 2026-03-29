from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def get_qa_chain(vector_db):
    retriever = vector_db.as_retriever()

    llm = ChatOpenAI(temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context:\n\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain