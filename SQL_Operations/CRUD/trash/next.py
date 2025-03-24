from langgraph.graph import END
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import Graph
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv("/Users/a2024/Desktop/Antar_CRM/.env")


# Pydantic Model for Lead
class LeadSchema(BaseModel):
    salutation: Optional[str] = Field(None, pattern="^(Mr|Mrs|Ms|Dr)$")
    tenant_id: Optional[str]
    first_name: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="retrieve the name of the user from the input",
    )
    last_name: str = Field(..., min_length=2, max_length=50)
    company: str = Field(..., max_length=100)
    email: str = Field(..., pattern=r"^\S+@\S+\.\S+$")
    gender: Optional[str] = Field(None, pattern="^(Male|Female|Other)$")
    lead_source: Optional[str] = None
    lead_score: Optional[int] = Field(None, ge=0, le=100)
    lead_value: Optional[float] = Field(None, ge=0)
    website: Optional[str] = None
    status: str = Field("new", pattern="^(new|contacted|qualified|lost)$")
    converted: Optional[bool] = False
    contact_id: Optional[str] = None
    employees: Optional[int] = Field(None, ge=0)
    assigned_to: Optional[str]
    territory_id: Optional[str]
    industry_id: Optional[int]
    revenue: Optional[float] = Field(None, ge=0)
    company_constitution: Optional[str] = None
    company_incorporation_date: Optional[str] = None
    mobile_code: Optional[str] = Field(None)
    mobile_no: Optional[str] = Field(None, min_length=10, max_length=15)
    department_id: Optional[str] = None
    changed_by: Optional[str]
    created_by: Optional[str]


# In-memory database (for demonstration purposes)
leads_db: Dict[str, LeadSchema] = {}


class LeadManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.parser = JsonOutputParser(pydantic_object=LeadSchema)
        self.required_fields = ["first_name", "last_name", "company", "email"]
        self.field_prompts = {
            "first_name": "First name",
            "last_name": "Last name",
            "company": "Company name",
            "email": "Email address",
        }

    def detect_intent(self, user_input: str) -> str:
        """Detect the user's intent (Create, Read, Update, Delete, or General Chat)."""
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the following user input and determine the intent:
            - If the user wants to create a new lead, respond with "create".
            - If the user wants to read lead information, respond with "read".
            - If the user wants to update lead information, respond with "update".
            - If the user wants to delete a lead, respond with "delete".
            - If the intent is unclear or unrelated to leads, respond with "general".

            User Input: {input}
            """
        )
        chain = prompt | self.llm
        response = chain.invoke({"input": user_input}).content
        return response.lower()

    def extract_lead_data(self, user_input: str) -> Dict:
        """Extract lead data from user input, filtering out fields not in the schema."""
        prompt = ChatPromptTemplate.from_template(
            """
            Extract lead details from the following input:
            {input}
            Return the extracted details in JSON format.
            """
        )
        chain = prompt | self.llm | self.parser
        try:
            # Extract data from the input
            extracted_data = chain.invoke({"input": user_input})
            # Filter out fields not in the LeadSchema
            valid_fields = LeadSchema.__fields__.keys()
            filtered_data = {
                k: v for k, v in extracted_data.items() if k in valid_fields
            }
            return filtered_data
        except ValidationError as e:
            print(f"Validation error: {e}")
            return {}

    def collect_mandatory_fields(self) -> Dict:
        """Collect mandatory fields one by one from the user."""
        collected_data = {}
        for field in self.required_fields:
            while True:
                user_input = input(f"Enter {self.field_prompts[field]}: ").strip()
                if user_input.lower() == "exit":
                    print("CRM Bot: Operation cancelled.")
                    return None
                if user_input:
                    collected_data[field] = user_input
                    break
                print(f"CRM Bot: {self.field_prompts[field]} is required.")
        return collected_data

    def confirm_details(self, collected_data: Dict) -> bool:
        """Ask the user to confirm the collected details."""
        print("\nCRM Bot: Please review the details:")
        for k, v in collected_data.items():
            print(f"  {k}: {v}")
        while True:
            choice = input("\nConfirm? (yes/edit): ").lower()
            if choice == "yes":
                return True
            elif choice == "edit":
                field = input("Enter field to edit: ").strip()
                if field in collected_data:
                    del collected_data[field]
                    return False
                print("Invalid field name")
            else:
                print("Please enter yes/edit")

    def create_lead(self, user_input: str) -> str:
        """Create a new lead by collecting mandatory fields and confirming details."""
        print(
            "\nCRM Bot: Let's create a new lead. Please provide the following details."
        )
        collected_data = self.collect_mandatory_fields()
        if not collected_data:
            return "Operation cancelled."

        while not self.confirm_details(collected_data):
            print("\nCRM Bot: Let's re-collect the missing fields.")
            collected_data.update(self.collect_mandatory_fields())

        try:
            lead = LeadSchema(**collected_data)
            leads_db[lead.email] = lead
            return f"Lead created successfully: {lead.json()}"
        except ValidationError as e:
            return f"Validation error: {e}"

    def read_lead(self, user_input: str) -> str:
        """Read lead information based on user input."""
        lead_data = self.extract_lead_data(user_input)
        if not lead_data:
            return "Failed to extract lead data. Please provide valid details."
        email = lead_data.get("email")
        if email and email in leads_db:
            return f"Lead details: {leads_db[email].json()}"
        return "Lead not found."

    def update_lead(self, user_input: str) -> str:
        """Update lead information based on user input."""
        lead_data = self.extract_lead_data(user_input)
        if not lead_data:
            return "Failed to extract lead data. Please provide valid details."
        email = lead_data.get("email")
        if email and email in leads_db:
            lead = leads_db[email]
            for key, value in lead_data.items():
                if value is not None:
                    setattr(lead, key, value)
            return f"Lead updated successfully: {lead.json()}"
        return "Lead not found or update failed."

    def delete_lead(self, user_input: str) -> str:
        """Delete a lead based on user input."""
        lead_data = self.extract_lead_data(user_input)
        if not lead_data:
            return "Failed to extract lead data. Please provide valid details."
        email = lead_data.get("email")
        if email and email in leads_db:
            del leads_db[email]
            return f"Lead with email {email} deleted successfully."
        return "Lead not found."

    def general_chat(self, user_input: str) -> str:
        """Handle general chat interactions."""
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Respond to the following: {input}"
        )
        chain = prompt | self.llm
        return chain.invoke({"input": user_input}).content


# LangGraph Workflow Setup
def setup_workflow(manager: LeadManager) -> Graph:
    """Define the workflow using LangGraph."""
    workflow = Graph()

    # Define nodes
    workflow.add_node("detect_intent", manager.detect_intent)
    workflow.add_node("create_lead", manager.create_lead)
    workflow.add_node("read_lead", manager.read_lead)
    workflow.add_node("update_lead", manager.update_lead)
    workflow.add_node("delete_lead", manager.delete_lead)
    workflow.add_node("general_chat", manager.general_chat)

    # Define conditional edges
    workflow.add_conditional_edges(
        "detect_intent",
        lambda x: {
            "create": "create_lead",
            "read": "read_lead",
            "update": "update_lead",
            "delete": "delete_lead",
            "general": "general_chat",
        }[x],
    )

    # Set entry point
    workflow.set_entry_point("detect_intent")

    # Set finish points individually
    workflow.add_edge("create_lead", END)
    workflow.add_edge("read_lead", END)
    workflow.add_edge("update_lead", END)
    workflow.add_edge("delete_lead", END)
    workflow.add_edge("general_chat", END)

    return workflow


# Main Execution
if __name__ == "__main__":
    print("CRM Lead Management Bot")
    print("-----------------------")
    manager = LeadManager()
    workflow = setup_workflow(manager)

    # Compile the workflow
    compiled_workflow = workflow.compile()

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("CRM Bot: Goodbye!")
            break

        # Execute the compiled workflow
        result = compiled_workflow.invoke({"input": user_input})
        print(f"CRM Bot: {result}")
