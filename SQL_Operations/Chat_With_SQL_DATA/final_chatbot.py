import asyncio
from dotenv import load_dotenv
from os import getenv
import ast
import re
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
from typing_extensions import TypedDict, Annotated, List
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv(dotenv_path="/Users/a2024/Desktop/Playground/.env")


# Define the unified state
class UnifiedState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    relevance: str
    query: str
    result: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


# Initialize Qdrant client for web chat
def initialize_qdrant_client():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="web_collection",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    return client


# Initialize embeddings and vector store for web chat
def initialize_web_vector_store(client):
    web_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    web_vector_store = QdrantVectorStore(
        client=client,
        collection_name="web_collection",
        embedding=web_embeddings,
    )
    return web_vector_store


# Load and chunk contents of the blog for web chat
async def load_documents():
    urls = ["https://www.oraczen.ai"]
    loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
    docs = await loader.aload()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits


# Initialize database for product search
def initialize_database(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    return SQLDatabase(engine=engine)


# Query data as list for product search
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


# Initialize vector store for product search
def initialize_vector_store(data, embedding_model="text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_texts(data)
    return vector_store


# Create retriever tool for product search
def create_retriever_tool_from_store(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    description = (
        "Use to look up values to filter on. Input is an approximate spelling "
        "of the proper noun, output is valid proper nouns. Use the noun most "
        "similar to the search."
    )
    return create_retriever_tool(
        retriever, name="search_proper_nouns", description=description
    )


# Initialize LLM and tools for product search
def initialize_llm_with_tools(retriever_tool):
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm.bind_tools([retriever_tool])
    return llm


# Function to decide which agent to use (product search or web chat) using LLM
def decide_agent(state: UnifiedState):
    relevance_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a decision-making assistant. Your task is to determine whether a user's question is related to 'product search' or 'web chat'."
                "If the question is about products, categories, prices, inventory, or anything related to a product database, respond with 'product_search'."
                "If the question is about a website, contact details, services, blogs, or anything related to web content, respond with 'web_chat'."
                "Respond ONLY with 'product_search' or 'web_chat' based on the question.",
            ),
            ("user", "Question: {question}"),
        ]
    )
    relevance_chain = relevance_prompt | llm
    response = relevance_chain.invoke({"question": state["question"]})
    decision = response.content.strip().lower()
    if decision not in ["product_search", "web_chat"]:
        decision = (
            "product_search"  # Default to product search if the response is invalid
        )
    return {"relevance": decision}


# Product search functions
def write_query(state: UnifiedState):
    query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


def execute_query(state: UnifiedState):
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def generate_product_answer(state: UnifiedState):
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    query_gen_system = """You are a customer service chatbot named 'ANTAR AI.'
        Using the SQL tools, generate the relevant answer. Be concise and clear in your responses, polite, and helpful.
        If the answer is not found, tell the user that the information is not available.
    """
    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", query_gen_system), ("placeholder", "{prompt}")]
    )
    query_gen = query_gen_prompt | llm
    response = query_gen.invoke({"prompt": [HumanMessage(content=prompt)]})
    return {"answer": response.content}


# Web chat functions
def retrieve_web(state: UnifiedState):
    retrieved_docs = web_vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate_web_answer(state: UnifiedState):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    template = (
        "Use the following pieces of context to answer the question at the end."
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "Use three sentences maximum and keep the answer as concise as possible."
        "Always say 'thanks for asking!' at the end of the answer."
        f"Context: {docs_content}\n"
        f'Question: {state["question"]}'
    )
    system_msg = """ You are a customer service CHATBOT your name is 'ORACZEN AI'
    company details and service chatbot generate the relevent answer so clear in your responses. Be polite and helpful.
    If answer not found tell the user that the information is not available.
    dont answer for unknown questions and dont make up answers. simply say I dont know if the question is not related to the context.
    """
    query_generate = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("placeholder", "{template}")]
    )
    web_llm = ChatOpenAI(model="gpt-4o-mini")
    output = query_generate | web_llm
    response = output.invoke({"template": [HumanMessage(content=template)]})
    return {"answer": response.content}


# Build the graph
def build_graph():
    graph_builder = StateGraph(UnifiedState)
    graph_builder.add_node("decide_agent", decide_agent)
    graph_builder.add_node("write_query", write_query)
    graph_builder.add_node("execute_query", execute_query)
    graph_builder.add_node("generate_product_answer", generate_product_answer)
    graph_builder.add_node("retrieve_web", retrieve_web)
    graph_builder.add_node("generate_web_answer", generate_web_answer)
    graph_builder.set_entry_point("decide_agent")
    graph_builder.add_conditional_edges(
        "decide_agent",
        lambda state: (
            "write_query" if state["relevance"] == "product_search" else "retrieve_web"
        ),
    )
    graph_builder.add_edge("write_query", "execute_query")
    graph_builder.add_edge("execute_query", "generate_product_answer")
    graph_builder.add_edge("generate_product_answer", END)
    graph_builder.add_edge("retrieve_web", "generate_web_answer")
    graph_builder.add_edge("generate_web_answer", END)
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


# Main function
def main():
    # Initialize components
    global db, llm, web_vector_store, graph
    client = initialize_qdrant_client()
    web_vector_store = initialize_web_vector_store(client)
    all_splits = asyncio.run(load_documents())
    _ = web_vector_store.add_documents(documents=all_splits)
    db = initialize_database("products.db")
    product = query_as_list(db, "SELECT product_name FROM products")
    category_name = query_as_list(db, "SELECT category_name FROM products")
    vector_store = initialize_vector_store(product + category_name)
    retriever_tool = create_retriever_tool_from_store(vector_store)
    llm = initialize_llm_with_tools(retriever_tool)
    graph = build_graph()

    # Main loop
    print("Welcome to ORACZEN AI Chatbot! Type 'exit' to end the chat.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("ORACZEN AI: Goodbye! Have a great day!")
            break
        state = {"question": question}
        config = {"configurable": {"thread_id": "chat_session"}}
        try:
            result = graph.invoke(state, config=config)
            answer = result["answer"]
            print(f"ORACZEN AI: {answer}")
        except Exception as e:
            print(
                f"ORACZEN AI: An error occurred while processing your request. Details: {e}"
            )


if __name__ == "__main__":
    main()
