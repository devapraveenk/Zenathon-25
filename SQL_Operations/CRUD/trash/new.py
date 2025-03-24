from langgraph.graph import START, StateGraph, END
from typing import List, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from Notes.create_notes import NoteManager

load_dotenv()


class Decision(TypedDict):
    operation: str
    schema_name: str  # Renamed from 'schema' to avoid conflict


class State(TypedDict):
    question: str
    operation: str
    schema_name: str


llm = ChatOpenAI(
    model="gpt-4o-mini",  # llm
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# Create
def create_note(state: State):
    print("Creating a new note")
    try:
        manager = NoteManager()
        manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_lead(state: State):
    print("Creating a new lead")
    pass


def create_contact(state: State):
    print("Creating a new contact")
    pass


def create_organization(state: State):
    print("Creating a new organization")
    pass


def create_opportunity(state: State):
    print("Creating a new opportunity")
    pass


def create_task(state: State):
    print("Creating a new task")
    pass


def create_conditional(state: State):

    if state["schema_name"] == "lead":
        return "create_lead"
    elif state["schema_name"] == "note":
        return "create_note"
    elif state["schema_name"] == "contact":
        return "create_contact"
    elif state["schema_name"] == "organization":
        return "create_organization"
    elif state["schema_name"] == "opportunity":
        return "create_opportunity"
    elif state["schema_name"] == "task":
        return "create_task"
    else:
        return "unknown"


# Read
def read_lead(state: State):
    print("Reading a lead")
    pass


def read_note(state: State):
    print("Reading a note")
    pass


def read_contact(state: State):
    print("Reading a contact")
    pass


def read_organization(state: State):
    print("Reading a organization")
    pass


def read_opportunity(state: State):
    print("Reading a opportunity")
    pass


def read_task(state: State):
    print("Reading a task")
    pass


def read_conditional(state: State):

    if state["schema_name"] == "lead":
        return "read_lead"
    elif state["schema_name"] == "note":
        return "read_note"
    elif state["schema_name"] == "contact":
        return "read_contact"
    elif state["schema_name"] == "organization":
        return "read_organization"
    elif state["schema_name"] == "opportunity":
        return "read_opportunity"
    elif state["schema_name"] == "task":
        return "read_task"
    else:
        return "unknown"


# Update
def update_lead(state: State):
    print("Updating a lead")
    pass


def update_note(state: State):
    print("Updating a note")
    pass


def update_contact(state: State):
    print("Updating a contact")
    pass


def update_organization(state: State):
    print("Updating a organization")
    pass


def update_opportunity(state: State):
    print("Updating a opportunity")
    pass


def update_task(state: State):
    print("Updating a task")
    pass


def update_conditional(state: State):

    if state["schema_name"] == "lead":
        return "update_lead"
    elif state["schema_name"] == "note":
        return "update_note"
    elif state["schema_name"] == "contact":
        return "update_contact"
    elif state["schema_name"] == "organization":
        return "update_organization"
    elif state["schema_name"] == "opportunity":
        return "update_opportunity"
    elif state["schema_name"] == "task":
        return "update_task"
    else:
        return "unknown"


# Delete
def delete_lead(state: State):
    print("Deleting a lead")
    pass


def delete_note(state: State):
    print("Deleting a note")
    pass


def delete_contact(state: State):
    print("Deleting a contact")
    pass


def delete_organization(state: State):
    print("Deleting a organization")
    pass


def delete_opportunity(state: State):
    print("Deleting a opportunity")
    pass


def delete_task(state: State):
    print("Deleting a task")
    pass


def delete_conditional(state: State):

    if state["schema_name"] == "lead":
        return "delete_lead"
    elif state["schema_name"] == "note":
        return "delete_note"
    elif state["schema_name"] == "contact":
        return "delete_contact"
    elif state["schema_name"] == "organization":
        return "delete_organization"
    elif state["schema_name"] == "opportunity":
        return "delete_opportunity"
    elif state["schema_name"] == "task":
        return "delete_task"
    else:
        return "unknown"


def unknown(state: State):
    print("Unknown operation")
    pass


def conditional_edge(state: State):

    # Routing the user query to the respective operation

    # Create
    if state["operation"] == "create":
        create_conditional(state)
    # Read
    elif state["operation"] == "read":
        if state["schema_name"] == "lead":
            return "read_lead"
        elif state["schema_name"] == "note":
            return "read_note"
        elif state["schema_name"] == "contact":
            return "read_contact"
        elif state["schema_name"] == "organization":
            return "read_organization"
        elif state["schema_name"] == "opportunity":
            return "read_opportunity"
        elif state["schema_name"] == "task":
            return "read_task"
    # Update
    elif state["operation"] == "update":
        if state["schema_name"] == "lead":
            return "update_lead"
        elif state["schema_name"] == "note":
            return "update_note"
        elif state["schema_name"] == "contact":
            return "update_contact"
        elif state["schema_name"] == "organization":
            return "update_organization"
        elif state["schema_name"] == "opportunity":
            return "update_opportunity"
        elif state["schema_name"] == "task":
            return "update_task"
    # Delete
    elif state["operation"] == "delete":
        if state["schema_name"] == "lead":
            return "delete_lead"
        elif state["schema_name"] == "note":
            return "delete_note"
        elif state["schema_name"] == "contact":
            return "delete_contact"
        elif state["schema_name"] == "organization":
            return "delete_organization"
        elif state["schema_name"] == "opportunity":
            return "delete_opportunity"
        elif state["schema_name"] == "task":
            return "delete_task"
    else:
        return "unknown"


def decide_intent(state: State):
    "Decide the operation to be performed based on the user query."

    system_message = """You are a CRM System Manager for Antar CRM.
    Here There are several operations CREATE, READ, UPDATE and DELETE.
    With Several Schema like Lead, Contact, Organization, Opportunity, Task, Note.
    User input will pass through the system and you need to decide which operation to perform.
    if in the user query any one of the information not available then return 'unknown'.
        e.g. what is salary then it should return 'unknown'.
    
    FOR EXAMPLE:
    If the user query is "Create a new lead", then you need to decide the operation as 'create' and schema as 'lead'.
    
    YOUR TASK:
        1. Extract the OPERATION NAME from the user query, and schema name.
        2. Return the operation name and the schema name.
        if related to create then it return with 'create' and so on and schema name as 'lead'.
        if the task is not related to any of the above operations then return 'unknown'.
    """

    prompt = ChatPromptTemplate(
        [("system", system_message), ("user", "this is the user input:{query}")]
    )
    chain = prompt | llm.with_structured_output(Decision)
    response = chain.invoke({"query": state["question"]})

    print(
        "__________________________________ USER QUERY __________________________________"
    )
    print(state["question"])
    print(
        "__________________________________ SYSTEM RESPONSE __________________________________"
    )
    state["operation"] = response["operation"]
    state["schema_name"] = response["schema_name"]
    print(state)
    return state


def build_graph():
    graph = StateGraph(State)
    graph.add_node("decide_intent", decide_intent)
    graph.add_node("create_lead", create_lead)
    graph.add_node("create_note", create_note)
    graph.add_node("create_contact", create_contact)
    graph.add_node("create_organization", create_organization)
    graph.add_node("create_opportunity", create_opportunity)
    graph.add_node("create_task", create_task)
    graph.add_node("read_lead", read_lead)
    graph.add_node("read_note", read_note)
    graph.add_node("read_contact", read_contact)
    graph.add_node("read_organization", read_organization)
    graph.add_node("read_opportunity", read_opportunity)
    graph.add_node("read_task", read_task)
    graph.add_node("update_lead", update_lead)
    graph.add_node("update_note", update_note)
    graph.add_node("update_contact", update_contact)
    graph.add_node("update_organization", update_organization)
    graph.add_node("update_opportunity", update_opportunity)
    graph.add_node("update_task", update_task)
    graph.add_node("delete_lead", delete_lead)
    graph.add_node("delete_note", delete_note)
    graph.add_node("delete_contact", delete_contact)
    graph.add_node("delete_organization", delete_organization)
    graph.add_node("delete_opportunity", delete_opportunity)
    graph.add_node("delete_task", delete_task)
    graph.add_node("unknown", unknown)

    graph.set_entry_point("decide_intent")

    graph.add_conditional_edges(
        "decide_intent",
        conditional_edge,
        {
            "create_note": "create_note",
            "create_lead": "create_lead",
            "create_contact": "create_contact",
            "create_organization": "create_organization",
            "create_opportunity": "create_opportunity",
            "create_task": "create_task",
            "read_lead": "read_lead",
            "read_note": "read_note",
            "read_contact": "read_contact",
            "read_organization": "read_organization",
            "read_opportunity": "read_opportunity",
            "read_task": "read_task",
            "update_lead": "update_lead",
            "update_note": "update_note",
            "update_contact": "update_contact",
            "update_organization": "update_organization",
            "update_opportunity": "update_opportunity",
            "update_task": "update_task",
            "delete_lead": "delete_lead",
            "delete_note": "delete_note",
            "delete_contact": "delete_contact",
            "delete_organization": "delete_organization",
            "delete_opportunity": "delete_opportunity",
            "delete_task": "delete_task",
            "unknown": "unknown",
        },
    )
    graph.add_edge("create_note", END)
    graph.add_edge("create_lead", END)
    graph.add_edge("create_contact", END)
    graph.add_edge("create_organization", END)
    graph.add_edge("create_opportunity", END)
    graph.add_edge("create_task", END)
    graph.add_edge("read_lead", END)
    graph.add_edge("read_note", END)
    graph.add_edge("read_contact", END)
    graph.add_edge("read_organization", END)
    graph.add_edge("read_opportunity", END)
    graph.add_edge("read_task", END)
    graph.add_edge("update_lead", END)
    graph.add_edge("update_note", END)
    graph.add_edge("update_contact", END)
    graph.add_edge("update_organization", END)
    graph.add_edge("update_opportunity", END)
    graph.add_edge("update_task", END)
    graph.add_edge("delete_lead", END)
    graph.add_edge("delete_note", END)
    graph.add_edge("delete_contact", END)
    graph.add_edge("delete_organization", END)
    graph.add_edge("delete_opportunity", END)
    graph.add_edge("delete_task", END)
    graph.add_edge("unknown", END)
    final_graph = graph.compile()
    return final_graph


if __name__ == "__main__":
    # Create an instance of the State dictionary
    question = input("> ").strip()
    state = {"question": question}

    graph = build_graph()
    while True:
        if question.lower() == "exit":
            break
        graph.invoke(state)
