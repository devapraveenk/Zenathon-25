from dotenv import load_dotenv

load_dotenv("/Users/a2024/Desktop/Antar_CRM/.env")

# Database Setup
DATABASE_URL = "URI"
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.mysql import CHAR
from uuid import uuid4
import json

# Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Lead(Base):
    __tablename__ = "leads"
    lead_id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid4()))
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(100))
    company = Column(String(100))
    assigned_to = Column(String(50))
    territory_id = Column(String(50))  # Ensure this is a string
    industry_id = Column(String(50))  # Ensure this is a string
    created_by = Column(String(50))
    changed_by = Column(String(50))  # Added missing field


Base.metadata.create_all(bind=engine)


# State Definition
class AgentState(TypedDict):
    query: str
    current_field: Optional[str]
    collected_data: Dict[str, Any]
    response: Optional[str]


# Mandatory Fields Configuration
MANDATORY_FIELDS = [
    "first_name",
    "last_name",
    "email",
    "company",
    "assigned_to",  # user id
    "territory_id",
    "industry_id",
    "created_by",
    "changed_by",  # Added to mandatory fields   #key valye pair
]

# Field Prompts
FIELD_PROMPTS = {
    "first_name": "Please provide the first name",
    "last_name": "Please provide the last name",
    "email": "Please provide the email address",
    "company": "Please provide the company name",
    "assigned_to": "Please provide the assigned user",
    "territory_id": "Please provide the territory ID",
    "industry_id": "Please provide the industry ID",
    "created_by": "Please provide the creator's name",
    "changed_by": "Please provide the modifier's name",
}


def get_next_missing_field(collected: dict) -> Optional[str]:
    for field in MANDATORY_FIELDS:
        if field not in collected or not str(collected[field]).strip():
            return field
    return None


def execute_create_operation(data: dict):
    session = SessionLocal()
    try:
        # Validate all fields have non-empty values
        for field in MANDATORY_FIELDS:
            if not str(data.get(field, "")).strip():
                raise ValueError(f"Missing value for {field}")

        data["lead_id"] = str(uuid4())
        lead = Lead(**data)
        session.add(lead)
        session.commit()
        return f"Lead created successfully with ID: {lead.lead_id}"
    except Exception as e:
        session.rollback()
        return f"Error: {str(e)}"
    finally:
        session.close()


def parse_input(state: AgentState):
    current_field = state.get("current_field")
    collected = state.get("collected_data", {})

    if current_field:
        # Validate input isn't empty
        value = state["query"].strip()
        if not value:
            return {
                "collected_data": collected,
                "current_field": current_field,
                "response": f"Invalid empty value. {FIELD_PROMPTS[current_field]}",
            }
        collected[current_field] = value
        return {"collected_data": collected, "current_field": None}

    try:
        # Improved NLP parsing
        prompt = ChatPromptTemplate.from_template(
            """
        Extract ONLY the following fields from: {input}
        Fields: {fields}
        Return JSON with EXISTING VALUES ONLY. Use null for missing fields.
        """
        )

        chain = prompt | ChatOpenAI() | JsonOutputParser()
        result = chain.invoke(
            {"fields": ", ".join(MANDATORY_FIELDS), "input": state["query"]}
        )

        # Clean empty values and ensure all fields are strings
        cleaned_result = {k: str(v) for k, v in result.items() if v is not None}
        return {"collected_data": {**collected, **cleaned_result}}
    except Exception as e:
        return {"collected_data": collected}


def check_completion(state: AgentState):
    collected = state.get("collected_data", {})
    missing = get_next_missing_field(collected)

    if not missing:
        try:
            result = execute_create_operation(collected)
            return {"response": result, "current_field": None}
        except Exception as e:
            return {"response": str(e), "current_field": None}
    else:
        return {"current_field": missing, "response": FIELD_PROMPTS[missing]}


# Workflow Setup
workflow = StateGraph(AgentState)
workflow.add_node("parse_input", parse_input)
workflow.add_node("check_completion", check_completion)
workflow.set_entry_point("parse_input")
workflow.add_edge("parse_input", "check_completion")
workflow.add_conditional_edges(
    "check_completion",
    lambda state: END if not state["current_field"] else "parse_input",
)

app = workflow.compile()

# Chat Interface
print("Bot: Hi! I'm your CRM assistant. How can I help you today?")
state = {"collected_data": {}, "current_field": None}

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    state["query"] = user_input
    result = app.invoke(state)

    # Update state
    state.update(
        {
            "collected_data": result.get("collected_data", state["collected_data"]),
            "current_field": result.get("current_field"),
            "response": result.get("response"),
        }
    )

    if "response" in result:
        print(f"Bot: {result['response']}")
