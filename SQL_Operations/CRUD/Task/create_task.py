from typing import Dict, Optional, Literal, Any, List, Union
from pydantic import BaseModel, Field, ValidationError
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
import requests
import json
from datetime import datetime


load_dotenv()


class TaskDetails(BaseModel):
    """Complete details of a task in a CRM system."""

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    title: str = Field(
        ..., min_length=2, max_length=100, description="Title of the task."
    )
    description: Optional[str] = Field(
        default="null", description="Description of the task."
    )
    type: Optional[str] = Field(
        default="In Progress",
        pattern="^(In Progress|Done|canceled|Todo|Backlog)$",
        description="Type of the task.",
    )
    big_status: Optional[str] = Field(
        default="Medium",
        pattern="^(Low|Medium|High)$",
        description="High-level status of the task.",
    )
    task_status: Optional[str] = Field(
        default="Meeting",
        pattern="^(Calling|Meeting|Event)$",
        description="Current status of the task.",
    )
    note_text: Optional[str] = Field(
        default="null", description="Additional notes or comments for the task."
    )
    task_data: Optional[str] = Field(
        default="null", description="Date and time for the task."
    )


class CRMAPIClient:
    """Client for interacting with the CRM API."""

    def __init__(self, base_url: str, token: str):
        """Initialize the CRM API client.

        Args:
            base_url: Base URL for the CRM API
            token: Authentication token for the API
        """
        self.base_url = base_url
        self.token = token
        self.headers = {
            "Authorization": token,
            "Content-Type": "application/json",
            "device": "crkiosk",
        }

    def create_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task in the CRM system.

        Args:
            data: Task data to submit

        Returns:
            API response as a dictionary

        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the API returns an invalid response
        """
        url = f"{self.base_url}"
        data["user_id"] = {
            "label": "vidhul",
            "val": "605ed9e2-e554-4b82-b818-e356797ff50a",
        }
        # print("Data:", data)

        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        if not response.text:
            raise ValueError("API returned an empty response")

        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            raise ValueError("API returned a non-JSON response")


class LLMManager:
    """Manager for LLM interactions."""

    def __init__(self, openai_api_key: str, groq_api_key: str):
        """Initialize the LLM manager.

        Args:
            openai_api_key: API key for OpenAI
            groq_api_key: API key for Groq
        """
        self.openai_api_key = openai_api_key
        self.groq_api_key = groq_api_key
        self.openai_llm = self._init_openai_llm()
        self.groq_llm = self._init_groq_llm()
        self.parser = JsonOutputParser(pydantic_object=TaskDetails)
        self.prompt_template = self._create_prompt_template()
        self.chain = self.prompt_template | self.openai_llm | self.parser

    def _init_openai_llm(self) -> ChatOpenAI:
        """Initialize the OpenAI LLM."""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=self.openai_api_key,
        )

    def _init_groq_llm(self) -> Any:
        """Initialize the Groq LLM."""
        return init_chat_model(
            "deepseek-r1-distill-llama-70b",
            model_provider="groq",
            api_key=self.groq_api_key,
        )

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for the LLM."""
        system_prompt = """You Are a CRM system details: extraxter agent you need to collect the following details from the given user query.

so collect the above details from the user query and return it.

You must respond with ONLY a valid JSON object in this exact format:
EXAMPLE:

User Query: create task as board meeting
output: {{"title": "board meeting"}} 

ALREADY COLLECTED DATA:
    as of now The Collected data are: {collected_data}
    IF the data are available dont replace with any other values like "NONE" 
    IF its null or DEFAULT value then collect the data from it
    Your work is parse the datas from the user query only DO NOT Add any new data And DO NOT Hallucinate YourSelf.

STRICTLY FOLLOW THE INSTRUCTIONS:
parse the AVAIALABLE data from the USER QUERY Dont return with None values.
Ensure underscores are NOT escaped.
parse the AVAIALABLE data from the USER QUERY Dont PARSE and RETURN with "None" values LEAVE THE DEFAULT VAULES.

`"You must always return valid JSON fenced by a markdown code block. Do not return any additional text."`
"DO NOT FORMAT like this: 
    {{
    "lead\\_score": 90
    }}"

Answer the user query. Wrap the output in `json` tags {format_instructions}
"""

        return ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("user", "this is the user input:{query}"),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())

    def process_query(
        self, query: str, collected_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a user query with the LLM.

        Args:
            query: User query to process
            collected_data: Previously collected data

        Returns:
            Processed data as a dictionary
        """
        return self.chain.invoke({"query": query, "collected_data": collected_data})


class DataValidator:
    """Validator for task data."""

    @staticmethod
    def clean_dict_keys(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
        """Recursively removes escaped underscores from dictionary keys.

        Args:
            data: Data to clean

        Returns:
            Cleaned data
        """
        if isinstance(data, dict):
            return {
                key.replace(r"\_", "_"): DataValidator.clean_dict_keys(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [DataValidator.clean_dict_keys(item) for item in data]
        else:
            return data

    @staticmethod
    def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against the TaskDetails model.

        Args:
            data: Data to validate

        Returns:
            Validated data

        Raises:
            ValidationError: If validation fails
        """
        return TaskDetails(**data).model_dump()

    @staticmethod
    def validate_data_check(
        data: Dict[str, Any], data_clean: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data against the TaskDetails model.

        Args:
            data: Data to validate

        Returns:
            Validated data

        Raises:
            ValidationError: If validation fails
        """
        return TaskDetails(**{**data, **data_clean})

    @staticmethod
    def handle_validation_error(error: ValidationError) -> str:
        """Generate a user-friendly error message from a Pydantic ValidationError.

        Args:
            error: ValidationError to handle

        Returns:
            User-friendly error message
        """
        errors = []
        for err in error.errors():
            field = err["loc"][0]  # Get the field name
            msg = err["msg"]  # Get the error message

            # Translate error messages into natural language
            if msg == "Input should be a valid string":
                errors.append(
                    f"The value for '{field}' must be text, not a number or other type."
                )
            elif msg == "Input should be a valid integer":
                errors.append(f"The value for '{field}' must be a whole number.")
            elif msg.startswith("String should match pattern"):
                errors.append(f"The value for '{field}' is not in the correct format.")
            elif msg == "ensure this value is greater than or equal to 0":
                errors.append(f"The value for '{field}' must be 0 or greater.")
            elif msg == "ensure this value is less than or equal to 100":
                errors.append(f"The value for '{field}' must be 100 or less.")
            else:
                errors.append(f"Error in '{field}': {msg}")

        return "\n".join(errors)


class CreateTaskManager:
    """Manager for CRM tasks."""

    def __init__(self):
        """Initialize the Task Manager."""
        load_dotenv()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.openai_api_key or not self.groq_api_key:
            raise ValueError("API keys not found in environment variables")

        self.llm_manager = LLMManager(self.openai_api_key, self.groq_api_key)

        # API configuration
        self.api_base_url = os.getenv("TASK_API_BASE_URL")
        self.api_token = os.getenv("API_TOKEN")

        self.api_client = CRMAPIClient(self.api_base_url, self.api_token)
        self.collected_data = {}

    def ask_to_update(self, field: str, current_value: str) -> bool:
        """Ask the user if they want to update an existing field.

        Args:
            field: Field name to update
            current_value: Current value of the field

        Returns:
            True if the user wants to update, False otherwise
        """
        print(f"Current value: {current_value}")
        response = (
            input(
                f"Do you want to update '{field}' (current value: '{current_value}')? (y/n): "
            )
            .strip()
            .lower()
        )
        return response == "y"

    def prompt_for_mandatory_fields(self):
        """Prompt the user for mandatory fields that are missing."""
        mandatory_fields = ["title"]
        missing_fields = [
            field
            for field in mandatory_fields
            if field not in self.collected_data or not self.collected_data[field]
        ]

        if missing_fields:
            print(f"Missing required fields: {', '.join(missing_fields)}")
            for field in missing_fields:
                self.collected_data[field] = input(f"Please enter {field}: ").strip()

        for field, value in self.collected_data.items():
            print(f"Updated {field} to {value}.")

    def process_user_input(self, query: str) -> bool:
        """Process user input to handle commands or extract data.

        Args:
            query: User input to process

        Returns:
            True if processing should continue, False to exit
        """
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            return False

        if query.lower() == "help":
            print("You can ask me to create a task in the CRM system.")
            print("Commands: exit/quit, help, show, clear, submit")
            return True

        if query.lower() == "show":
            try:
                print(
                    json.dumps(
                        DataValidator.validate_data(self.collected_data), indent=4
                    )
                )
            except ValidationError as e:
                print(DataValidator.handle_validation_error(e))
            return True

        if query.lower() == "clear":
            self.collected_data = {}
            print("Data cleared.")
            return True

        if query.lower() == "submit":
            try:
                validated_data = DataValidator.validate_data(self.collected_data)
                print("Check the data before submitting:")
                print(json.dumps(validated_data, indent=4))

                if input("Do you want to submit this data? (y/n): ").lower() == "y":
                    response = self.api_client.create_task(data=validated_data)
                    print("Data submitted successfully.")
                    print(f"Response:", response["message"])
                    return False  # Exit after successful submission
            except (ValidationError, requests.RequestException, ValueError) as e:
                print(f"Error: {e}")
            return True

        else:
            # Process regular query for data extraction
            try:
                # Get parsed data from LLM
                parsed_data = self.llm_manager.process_query(query, self.collected_data)
                cleaned_data = DataValidator.clean_dict_keys(parsed_data)

                # Check if any new information was parsed
                if cleaned_data == self.collected_data:
                    print(
                        "No new information was parsed from your input. Please provide more details."
                    )
                    return True

                # Update fields with confirmation for existing values
                for field, value in cleaned_data.items():
                    if (
                        field in self.collected_data
                        and value != self.collected_data[field]
                    ):
                        if not self.ask_to_update(field, self.collected_data[field]):
                            continue
                    self.collected_data[field] = value

                # Validate the updated data
                try:
                    DataValidator.validate_data_check(self.collected_data, cleaned_data)
                    print("Data validated successfully.")
                except ValidationError as e:
                    print(DataValidator.handle_validation_error(e))

                # Prompt for any missing mandatory fields
                self.prompt_for_mandatory_fields()

            except Exception as e:
                print(f"Error processing input: {e}")

            return True

    def run_interactive_session(self, state):
        """Run an interactive session for task management."""
        print("CRM Task Manager - Interactive Session")
        print("Type 'help' for available commands.")
        print("How can I help you create a task?")
        self.process_user_input(state["question"])

        while True:
            query = input("> ").strip()
            if not self.process_user_input(query):
                break


def main():
    """Main entry point of the application."""
    try:
        manager = CreateTaskManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
