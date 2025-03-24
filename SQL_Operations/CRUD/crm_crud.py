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
from Notes.delete_notes import DeleteNoteManager
from Organization.delete_organization import DeleteOrganizationManager
from Opportunity.delete_opportunity import DeleteOpportunityManager
from Product.delete_product import DeleteProductManager
from Task.delete_task import DeleteTaskManager
from Contact.read_contact import ReadContactManager
from Lead.read_lead import ReadLeadManager
from Notes.read_notes import ReadNoteManager
from Opportunity.read_opportunity import ReadOpportunityManager
from Organization.read_organization import ReadOrganizationManager
from Product.read_product import ReadProductManager
from Task.read_task import ReadTaskManager


load_dotenv()


class Decision(TypedDict):
    operation: str
    schema_name: str  # Renamed from 'schema' to avoid conflict


class State(TypedDict):
    question: str
    operation: str
    schema_name: str
    next_node: str
    error: str


class CRMGraph:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # llm
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.graph = self.build_graph()

    # Create operations
    def create_note(self, state: State):
        print("Creating a new note")
        try:
            create_note_manager = CreateNoteManager()
            create_note_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_lead(self, state: State):
        try:
            create_lead_manager = CreateLeadManager()
            create_lead_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_contact(self, state: State):
        try:
            create_contact_manager = CreateContactManager()
            create_contact_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_organization(self, state: State):
        try:
            create_organization_manager = CreateOrganizationManager()
            create_organization_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_opportunity(self, state: State):
        try:
            create_opportunity_manager = CreateOpportunityManager()
            create_opportunity_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_product(self, state: State):
        try:
            create_product_manager = CreateProductManager()
            create_product_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_task(self, state: State):
        try:
            create_task_manager = CreateTaskManager()
            create_task_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def create_conditional(self, state: State):
        if state["schema_name"] == "lead":
            state["next_node"] = "create_lead"
        elif state["schema_name"] == "note":
            state["next_node"] = "create_note"
        elif state["schema_name"] == "contact":
            state["next_node"] = "create_contact"
        elif state["schema_name"] == "organization":
            state["next_node"] = "create_organization"
        elif state["schema_name"] == "opportunity":
            state["next_node"] = "create_opportunity"
        elif state["schema_name"] == "task":
            state["next_node"] = "create_task"
        elif state["schema_name"] == "product":
            state["next_node"] = "create_product"
        else:
            state["next_node"] = "unknown"
        return state

    # Read operations
    def read_lead(self, state: State):
        try:
            manager = ReadLeadManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_note(self, state: State):
        try:
            manager = ReadNoteManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_contact(self, state: State):
        try:
            manager = ReadContactManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_organization(self, state: State):

        try:
            manager = ReadOrganizationManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_opportunity(self, state: State):
        try:
            create_opportunity_manager = ReadOpportunityManager()
            create_opportunity_manager.run_interactive_session(state)
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_task(self, state: State):
        try:
            manager = ReadTaskManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_product(self, state: State):
        try:
            manager = ReadProductManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def read_conditional(self, state: State):
        if state["schema_name"] == "lead":
            state["next_node"] = "read_lead"
        elif state["schema_name"] == "note":
            state["next_node"] = "read_note"
        elif state["schema_name"] == "contact":
            state["next_node"] = "read_contact"
        elif state["schema_name"] == "organization":
            state["next_node"] = "read_organization"
        elif state["schema_name"] == "opportunity":
            state["next_node"] = "read_opportunity"
        elif state["schema_name"] == "task":
            state["next_node"] = "read_task"
        elif state["schema_name"] == "product":
            state["next_node"] = "read_product"
        else:
            state["next_node"] = "unknown"
        return state

    # Update operations
    def update_lead(self, state: State):
        print("Updating a lead")
        # Implementation here
        return state

    def update_note(self, state: State):
        print("Updating a note")
        # Implementation here
        return state

    def update_contact(self, state: State):
        print("Updating a contact")
        # Implementation here
        return state

    def update_organization(self, state: State):
        print("Updating a organization")
        # Implementation here
        return state

    def update_opportunity(self, state: State):
        print("Updating a opportunity")
        # Implementation here
        return state

    def update_task(self, state: State):
        print("Updating a task")
        # Implementation here
        return state

    def update_product(self, state: State):
        print("Updating a product")
        # Implementation here
        return state

    def update_conditional(self, state: State):
        if state["schema_name"] == "lead":
            state["next_node"] = "update_lead"
        elif state["schema_name"] == "note":
            state["next_node"] = "update_note"
        elif state["schema_name"] == "contact":
            state["next_node"] = "update_contact"
        elif state["schema_name"] == "organization":
            state["next_node"] = "update_organization"
        elif state["schema_name"] == "opportunity":
            state["next_node"] = "update_opportunity"
        elif state["schema_name"] == "task":
            state["next_node"] = "update_task"
        elif state["schema_name"] == "product":
            state["next_node"] = "update_product"
        else:
            state["next_node"] = "unknown"
        return state

    # Delete operations
    def delete_lead(self, state: State):
        try:
            manager = DeleteLeadManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_note(self, state: State):
        try:
            manager = DeleteNoteManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_contact(self, state: State):
        try:
            manager = DeleteContactManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_organization(self, state: State):
        try:
            manager = DeleteOrganizationManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_opportunity(self, state: State):
        try:
            manager = DeleteOpportunityManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_task(self, state: State):
        try:
            manager = DeleteTaskManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_product(self, state: State):
        try:
            manager = DeleteProductManager()
            manager.run_interactive_session()
            return state
        except Exception as e:
            print(f"Application error: {e}")
            state["error"] = str(e)
            return state

    def delete_conditional(self, state: State):
        if state["schema_name"] == "lead":
            state["next_node"] = "delete_lead"
        elif state["schema_name"] == "note":
            state["next_node"] = "delete_note"
        elif state["schema_name"] == "contact":
            state["next_node"] = "delete_contact"
        elif state["schema_name"] == "organization":
            state["next_node"] = "delete_organization"
        elif state["schema_name"] == "opportunity":
            state["next_node"] = "delete_opportunity"
        elif state["schema_name"] == "task":
            state["next_node"] = "delete_task"
        elif state["schema_name"] == "product":
            state["next_node"] = "delete_product"
        else:
            state["next_node"] = "unknown"
        return state

    def unknown(self, state: State):
        print("Unknown operation")
        state["response"] = "Unknown operation or schema"
        return state

    def decide_intent(self, state: State):
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
        chain = prompt | self.llm.with_structured_output(Decision)
        response = chain.invoke({"query": state["question"]})

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

        # print(state)
        return state

    def build_graph(self):
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("decide_intent", self.decide_intent)

        # Create nodes
        graph.add_node("create_lead", self.create_lead)
        graph.add_node("create_note", self.create_note)
        graph.add_node("create_contact", self.create_contact)
        graph.add_node("create_organization", self.create_organization)
        graph.add_node("create_opportunity", self.create_opportunity)
        graph.add_node("create_task", self.create_task)
        graph.add_node("create_product", self.create_product)
        graph.add_node("create_conditional", self.create_conditional)

        # Read nodes
        graph.add_node("read_lead", self.read_lead)
        graph.add_node("read_note", self.read_note)
        graph.add_node("read_contact", self.read_contact)
        graph.add_node("read_organization", self.read_organization)
        graph.add_node("read_opportunity", self.read_opportunity)
        graph.add_node("read_task", self.read_task)
        graph.add_node("read_product", self.read_product)
        graph.add_node("read_conditional", self.read_conditional)

        # Update nodes
        graph.add_node("update_lead", self.update_lead)
        graph.add_node("update_note", self.update_note)
        graph.add_node("update_contact", self.update_contact)
        graph.add_node("update_organization", self.update_organization)
        graph.add_node("update_opportunity", self.update_opportunity)
        graph.add_node("update_task", self.update_task)
        graph.add_node("update_product", self.update_product)
        graph.add_node("update_conditional", self.update_conditional)

        # Delete nodes
        graph.add_node("delete_lead", self.delete_lead)
        graph.add_node("delete_note", self.delete_note)
        graph.add_node("delete_contact", self.delete_contact)
        graph.add_node("delete_organization", self.delete_organization)
        graph.add_node("delete_opportunity", self.delete_opportunity)
        graph.add_node("delete_task", self.delete_task)
        graph.add_node("delete_product", self.delete_product)
        graph.add_node("delete_conditional", self.delete_conditional)

        graph.add_node("unknown", self.unknown)

        graph.set_entry_point("decide_intent")

        # Add edges
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

        # Add edges to END
        graph.add_edge("create_lead", END)
        graph.add_edge("create_note", END)
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

        # Compile the graph
        final_graph = graph.compile()
        return final_graph

    def process_query(self, query: str):
        """Process a user query using the graph."""
        state = {"question": query}
        self.graph.invoke(state)
        return state


if __name__ == "__main__":
    crm_manager = CRMGraph()
    result = crm_manager.process_query(input("> ").strip())
    print(result)
