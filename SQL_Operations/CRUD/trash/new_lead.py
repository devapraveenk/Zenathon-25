import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, validator

load_dotenv("/Users/a2024/Desktop/Antar_CRM/.env")


# Pydantic Model
#
class LeadSchema(BaseModel):
    salutation: Optional[str] = Field(None, pattern="^(Mr|Mrs|Ms|Dr)$")
    tenant_id: Optional[str]
    first_name: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="retrive the name of the user from the input",
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


class LeadCreator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.parser = JsonOutputParser(pydantic_object=LeadSchema)
        self.collected_data: Dict = {}
        self.required_fields = [
            "first_name",
            "last_name",
            "company",
            "email",
        ]
        self.field_prompts = {
            "first_name": "First name",
            "last_name": "Last name",
            "company": "Company name",
            "email": "Email address",
        }

    def parse_input(self, user_input: str) -> Dict:
        try:
            prompt = ChatPromptTemplate.from_template(
                """
            Extract lead fields from: {input}
            Known fields: {current_fields}
            Possible fields: {possible_fields}
            Return JSON with any extracted fields.
            """
            )

            chain = prompt | self.llm | self.parser
            return chain.invoke(
                {
                    "input": user_input,
                    "current_fields": str(self.collected_data),
                    "possible_fields": str(self.required_fields),
                }
            )
        except:
            return {}

    def get_missing_fields(self) -> List[str]:
        return [f for f in self.required_fields if f not in self.collected_data]

    def collect_data(self):
        print(
            "\nCRM Bot: Let's create a new lead. Provide info in any format. Say 'add' when ready."
        )

        while True:
            user_input = input("\nUser: ").strip()

            if user_input.lower() in ["exit", "cancel"]:
                print("CRM Bot: Operation cancelled.")
                return None

            if user_input.lower() == "add":
                if missing := self.get_missing_fields():
                    print(f"CRM Bot: Need more info: {', '.join(missing)}")
                    continue
                return self.confirm_and_save()

            # Process partial input
            new_data = self.parse_input(user_input)
            self.collected_data.update(new_data)

            # Show current status
            missing = self.get_missing_fields()
            collected = len(self.required_fields) - len(missing)
            print(
                f"CRM Bot: Collected {collected}/10 fields. Missing: {', '.join(missing) or 'none'}"
            )

    def confirm_and_save(self):
        print("\nCRM Bot: Please review the details:")
        for k, v in self.collected_data.items():
            print(f"  {k}: {v}")

        while True:
            choice = input("\nConfirm? (yes/edit): ").lower()
            if choice == "yes":
                return self.format_output()
            elif choice == "edit":
                field = input("Enter field to edit: ").strip()
                if field in self.collected_data:
                    del self.collected_data[field]
                    return self.collect_data()
                print("Invalid field name")
            else:
                print("Please enter yes/edit")

    def format_output(self):
        try:
            return LeadSchema(**self.collected_data).dict()
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return None


if __name__ == "__main__":
    print("CRM Lead Creation Bot")
    print("---------------------")
    creator = LeadCreator()

    while True:
        result = creator.collect_data()
        if result:
            print("\nFinal Lead Data:")
            print(json.dumps(result, indent=2))
            break
