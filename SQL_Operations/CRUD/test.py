from langgraph.graph import START, StateGraph, END
from typing import List, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from Notes.create_notes import CreateNoteManager
from Organization.create_organization import CreateOrganizationManager
from Opportunity.create_opportunity import CreateOpportunityManager
from Product.create_product import CreateProductManager
from Task.create_task import CreateTaskManager
from Contact.create_contact import CreateContactManager
from Lead.create_lead import CreateLeadManager
from Lead.delete_lead import DeleteLeadManager
from Contact.delete_contact import DeleteContactManager

load_dotenv()


class Decision(TypedDict):
    operation: str
    schema_name: str  # Renamed from 'schema' to avoid conflict


class State(TypedDict):
    question: str
    operation: str
    schema_name: str
    next_node: str


llm = ChatOpenAI(
    model="gpt-4o-mini",  # llm
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# Create
def create_note(state: State):
    print("Creating a new note")
    try:
        create_note_manager = CreateNoteManager()
        create_note_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_lead(state: State):

    try:
        create_lead_manager = CreateLeadManager()
        create_lead_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_contact(state: State):

    try:
        create_contact_manager = CreateContactManager()
        create_contact_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_organization(state: State):
    try:
        create_organization_manager = CreateOrganizationManager()
        create_organization_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_opportunity(state: State):
    print("Creating a new opportunity")
    try:
        create_opportunity_manager = CreateOpportunityManager()
        create_opportunity_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_product(state: State):

    try:
        create_product_manager = CreateProductManager()
        create_product_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_task(state: State):

    try:
        create_task_manager = CreateTaskManager()
        create_task_manager.run_interactive_session(state)
    except Exception as e:
        print(f"Application error: {e}")


def create_conditional(state: State):
    if state["schema_name"] == "lead":
        return {"next_node": "create_lead"}
    elif state["schema_name"] == "note":
        return {"next_node": "create_note"}
    elif state["schema_name"] == "contact":
        return {"next_node": "create_contact"}
    elif state["schema_name"] == "organization":
        return {"next_node": "create_organization"}
    elif state["schema_name"] == "opportunity":
        return {"next_node": "create_opportunity"}
    elif state["schema_name"] == "task":
        return {"next_node": "create_task"}
    elif state["schema_name"] == "product":
        return {"next_node": "create_product"}
    else:
        return {"next_node": "unknown"}


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


def read_product(state: State):
    print("Reading a product")
    pass


def read_conditional(state: State):
    if state["schema_name"] == "lead":
        return {"next_node": "read_lead"}
    elif state["schema_name"] == "note":
        return {"next_node": "read_note"}
    elif state["schema_name"] == "contact":
        return {"next_node": "read_contact"}
    elif state["schema_name"] == "organization":
        return {"next_node": "read_organization"}
    elif state["schema_name"] == "opportunity":
        return {"next_node": "read_opportunity"}
    elif state["schema_name"] == "task":
        return {"next_node": "read_task"}
    elif state["schema_name"] == "product":
        return {"next_node": "read_product"}
    else:
        return {"next_node": "unknown"}


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


def update_product(state: State):
    print("Updating a product")
    pass


def update_conditional(state: State):
    if state["schema_name"] == "lead":
        return {"next_node": "update_lead"}
    elif state["schema_name"] == "note":
        return {"next_node": "update_note"}
    elif state["schema_name"] == "contact":
        return {"next_node": "update_contact"}
    elif state["schema_name"] == "organization":
        return {"next_node": "update_organization"}
    elif state["schema_name"] == "opportunity":
        return {"next_node": "update_opportunity"}
    elif state["schema_name"] == "task":
        return {"next_node": "update_task"}
    elif state["schema_name"] == "product":
        return {"next_node": "update_product"}
    else:
        return {"next_node": "unknown"}


# Delete
def delete_lead(state: State):

    try:
        manager = DeleteLeadManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")


def delete_note(state: State):
    print("Deleting a note")
    pass


def delete_contact(state: State):

    try:
        manager = DeleteContactManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")


def delete_organization(state: State):
    print("Deleting a organization")
    pass


def delete_opportunity(state: State):
    print("Deleting a opportunity")
    pass


def delete_task(state: State):
    print("Deleting a task")
    pass


def delete_product(state: State):
    print("Deleting a product")
    pass


def delete_conditional(state: State):
    if state["schema_name"] == "lead":
        return {"next_node": "delete_lead"}
    elif state["schema_name"] == "note":
        return {"next_node": "delete_note"}
    elif state["schema_name"] == "contact":
        return {"next_node": "delete_contact"}
    elif state["schema_name"] == "organization":
        return {"next_node": "delete_organization"}
    elif state["schema_name"] == "opportunity":
        return {"next_node": "delete_opportunity"}
    elif state["schema_name"] == "task":
        return {"next_node": "delete_task"}
    elif state["schema_name"] == "product":
        return {"next_node": "delete_product"}
    else:
        return {"next_node": "unknown"}


def unknown(state: State):
    print("Unknown operation")
    pass


def decide_intent(state: State):
    """Decide the operation to be performed based on the user query."""
    system_message = """You are a CRM System Manager for Antar CRM.
    Here There are several operations CREATE, READ, UPDATE and DELETE.
    With Several Schema like Lead, Contact, Organization, Opportunity, Products, Task, Note.
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

    # print(
    #     "__________________________________ USER QUERY __________________________________"
    # )
    # print(state["question"])
    # print(
    #     "__________________________________ SYSTEM RESPONSE __________________________________"
    # )
    state["operation"] = response["operation"]
    state["schema_name"] = response["schema_name"]

    # Add the next_node key based on the operation
    if state["operation"] == "create":
        state["next_node"] = "create_conditional"
    elif state["operation"] == "read":
        state["next_node"] = "read_conditional"
    elif state["operation"] == "update":
        state["next_node"] = "update_conditional"
    elif state["operation"] == "delete":
        state["next_node"] = "delete_conditional"
    else:
        state["next_node"] = "unknown"

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
    graph.add_node("create_product", create_product)

    graph.add_node("create_conditional", create_conditional)

    graph.add_node("read_lead", read_lead)
    graph.add_node("read_note", read_note)
    graph.add_node("read_contact", read_contact)
    graph.add_node("read_organization", read_organization)
    graph.add_node("read_opportunity", read_opportunity)
    graph.add_node("read_task", read_task)
    graph.add_node("read_product", read_product)

    graph.add_node("read_conditional", read_conditional)

    graph.add_node("update_lead", update_lead)
    graph.add_node("update_note", update_note)
    graph.add_node("update_contact", update_contact)
    graph.add_node("update_organization", update_organization)
    graph.add_node("update_opportunity", update_opportunity)
    graph.add_node("update_task", update_task)
    graph.add_node("update_product", update_product)

    graph.add_node("update_conditional", update_conditional)

    graph.add_node("delete_lead", delete_lead)
    graph.add_node("delete_note", delete_note)
    graph.add_node("delete_contact", delete_contact)
    graph.add_node("delete_organization", delete_organization)
    graph.add_node("delete_opportunity", delete_opportunity)
    graph.add_node("delete_task", delete_task)
    graph.add_node("delete_product", delete_product)

    graph.add_node("delete_conditional", delete_conditional)

    graph.add_node("unknown", unknown)

    graph.set_entry_point("decide_intent")

    graph.add_conditional_edges(
        "decide_intent",
        lambda state: state["next_node"],
        {
            "create_conditional": "create_conditional",
            "read_conditional": "read_conditional",
            "update_conditional": "update_conditional",
            "delete_conditional": "delete_conditional",
            "unknown": "unknown",
        },
    )

    graph.add_conditional_edges(
        "create_conditional",
        lambda state: state["next_node"],
        {
            "create_lead": "create_lead",
            "create_note": "create_note",
            "create_contact": "create_contact",
            "create_organization": "create_organization",
            "create_opportunity": "create_opportunity",
            "create_task": "create_task",
            "create_product": "create_product",
            "unknown": "unknown",
        },
    )

    graph.add_conditional_edges(
        "read_conditional",
        lambda state: state["next_node"],
        {
            "read_lead": "read_lead",
            "read_note": "read_note",
            "read_contact": "read_contact",
            "read_organization": "read_organization",
            "read_opportunity": "read_opportunity",
            "read_task": "read_task",
            "read_product": "read_product",
            "unknown": "unknown",
        },
    )

    graph.add_conditional_edges(
        "update_conditional",
        lambda state: state["next_node"],
        {
            "update_lead": "update_lead",
            "update_note": "update_note",
            "update_contact": "update_contact",
            "update_organization": "update_organization",
            "update_opportunity": "update_opportunity",
            "update_task": "update_task",
            "update_product": "update_product",
            "unknown": "unknown",
        },
    )

    graph.add_conditional_edges(
        "delete_conditional",
        lambda state: state["next_node"],
        {
            "delete_lead": "delete_lead",
            "delete_note": "delete_note",
            "delete_contact": "delete_contact",
            "delete_organization": "delete_organization",
            "delete_opportunity": "delete_opportunity",
            "delete_task": "delete_task",
            "delete_product": "delete_product",
            "unknown": "unknown",
        },
    )

    graph.add_edge("create_note", END)
    graph.add_edge("create_lead", END)
    graph.add_edge("create_contact", END)
    graph.add_edge("create_organization", END)
    graph.add_edge("create_opportunity", END)
    graph.add_edge("create_task", END)
    graph.add_edge("create_product", END)

    graph.add_edge("read_lead", END)
    graph.add_edge("read_note", END)
    graph.add_edge("read_contact", END)
    graph.add_edge("read_organization", END)
    graph.add_edge("read_opportunity", END)
    graph.add_edge("read_task", END)
    graph.add_edge("read_product", END)

    graph.add_edge("update_lead", END)
    graph.add_edge("update_note", END)
    graph.add_edge("update_contact", END)
    graph.add_edge("update_organization", END)
    graph.add_edge("update_opportunity", END)
    graph.add_edge("update_task", END)
    graph.add_edge("update_product", END)

    graph.add_edge("delete_lead", END)
    graph.add_edge("delete_note", END)
    graph.add_edge("delete_contact", END)
    graph.add_edge("delete_organization", END)
    graph.add_edge("delete_opportunity", END)
    graph.add_edge("delete_task", END)
    graph.add_edge("delete_product", END)

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
        question = input("> ").strip()
        state = {"question": question}
