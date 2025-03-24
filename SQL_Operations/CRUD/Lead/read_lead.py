import requests
import json
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for the CRM Lead Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    model_id: str


class ReadLeadManager:
    """A class to manage CRM leads with get operation"""

    def __init__(self):
        """Initialize the Lead Manager with configuration from environment variables"""
        load_dotenv()

        self.config = Config(
            api_url_base=os.getenv("API_URL_BASE", "http://localhost:8088"),
            api_token=os.getenv("API_TOKEN", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            groq_model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            model_id=os.getenv("LEAD_MODEL_ID", "8caca9a9-7c48-4081-9eeb-170009963a5f"),
        )

        self._validate_config()

        # Initialize LLM clients
        self.openai_llm = ChatOpenAI(
            model=self.config.openai_model,
            temperature=0.2,
            openai_api_key=self.config.openai_api_key,
        )

        # Pre-processed data cache
        self.email_to_lead: Dict[str, Dict] = {}
        self.email_to_lead_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_lead_data()

    def _validate_config(self) -> None:
        """Validate that all required configuration is present"""
        required_fields = [
            "api_url_base",
            "api_token",
            "openai_api_key",
            "groq_api_key",
            "model_id",
        ]

        for field in required_fields:
            if not getattr(self.config, field):
                raise ValueError(f"Missing required configuration: {field}")

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for API requests"""
        return {
            "Authorization": self.config.api_token,
            "Content-Type": "application/json",
            "device": "test",
        }

    def _refresh_lead_data(self) -> None:
        """Refresh the cached lead data"""
        lead_data = self.get_leads()
        if lead_data:
            self.email_to_lead = self._preprocess_leads(lead_data)
            self.email_to_lead_id = self._preprocess_lead_ids(lead_data)

    def get_leads(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all leads from the API

        Returns:
            A dictionary containing lead data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_leads?mdl={self.config.model_id}&&role=undefined"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            return None
        except json.JSONDecodeError:
            print("API returned an invalid JSON response")
            return None

    def _preprocess_leads(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess lead data for efficient email-based lookup

        Args:
            data: The raw lead data from the API

        Returns:
            A dictionary mapping email addresses to lead data
        """

        desired_keys = [
            "first_name",
            "email",
            "lead_score",
            "lead_value",
            "status",
            "created_at",
        ]

        email_to_lead = {}
        try:
            for lead in data.get("data", []):
                email = lead.get("email")
                filtered_lead = {key: lead[key] for key in desired_keys if key in lead}
                if email:
                    email_to_lead[email.lower()] = filtered_lead
            return email_to_lead
        except Exception as e:
            print(f"Error preprocessing lead data: {str(e)}")
            return {}

    def _preprocess_lead_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess lead data to map emails to lead IDs

        Args:
            data: The raw lead data from the API

        Returns:
            A dictionary mapping email addresses to lead IDs
        """
        email_to_lead_id = {}
        try:
            for lead in data.get("data", []):
                email = lead.get("email")
                lead_id = lead.get("lead_id")
                if email and lead_id:
                    email_to_lead_id[email.lower()] = lead_id
            return email_to_lead_id
        except Exception:
            return {}

    def find_lead_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Find a lead by email address

        Args:
            email: The email address to search for

        Returns:
            The lead data if found, None otherwise
        """
        if not email:
            return None

        email = email.lower()
        return self.email_to_lead.get(email)

    def get_lead_id_by_email(self, email: str) -> Optional[str]:
        """
        Get a lead ID by email address

        Args:
            email: The email address to search for

        Returns:
            The lead ID if found, None otherwise
        """
        if not email:
            return None

        email = email.lower()
        return self.email_to_lead_id.get(email)

    def analyze_lead_details(self, lead_data: Any) -> str:
        """
        Analyze lead details using an LLM

        Args:
            lead_data: The lead data to analyze

        Returns:
            A human-readable analysis of the lead data
        """
        if not lead_data:
            return "No lead details found."

        if isinstance(lead_data, dict):
            lead_data = json.dumps(lead_data, indent=4)

        try:
            prompt = ChatPromptTemplate(
                [
                    (
                        "system",
                        """You are a User Friendly Chatbot. The user details are passed as input. 
                        Summarize the key information about this user in a clear, professional manner.
                        If result not found then report there are no details found.
                        """,
                    ),
                    ("user", "The user details are: {details}"),
                ]
            )

            llm = init_chat_model(
                self.config.groq_model,
                model_provider="groq",
                api_key=self.config.groq_api_key,
            )

            chain = prompt | llm
            result = chain.invoke({"details": lead_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing lead details: {str(e)}"

    def get_lead(self, email: str) -> bool:
        """
        get a lead by email address

        Args:
            email: The email address of the lead to get

        Returns:
            True if the lead was getd successfully, False otherwise
        """
        if not email:
            print("No email provided for deletion")
            return False

        lead_id = self.get_lead_id_by_email(email)
        if not lead_id:
            print(f"No lead ID found for email: {email}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/find_single_lead?mdl={self.config.model_id}&&id={lead_id}"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after deletion
            self._refresh_lead_data()

            print(f"Successfully getd lead with email: {email}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error Searching lead: {str(e)}")
            return False

    def run_interactive_session(self):
        """Run the interactive get session"""
        print("Starting Lead Manager Interactive Session")

        while True:
            try:
                get_email = input("Enter the email to get (or 'exit' to quit): ")

                if get_email.lower() in ["exit", "cancel", "quit"]:
                    print("Exiting the program.")
                    break

                if not get_email:
                    print("Email cannot be empty.")
                    continue

                lead_data = self.find_lead_by_email(get_email)
                if lead_data:
                    print("\n=== Lead to get ===")
                    analysis = self.analyze_lead_details(lead_data)
                    print(analysis)

                    confirmation = input("Do you want to get this lead? (y/n): ")
                    if confirmation.lower() == "y":
                        success = self.get_lead(get_email)
                        if success:
                            print("Lead successfully getd.")
                            # Ask if user wants to get another lead
                            continue_choice = input("get another lead? (y/n): ")
                            if continue_choice.lower() != "y":
                                break
                        else:
                            print("Failed to get lead.")
                    else:
                        print("Operation cancelled.")
                else:
                    print(f"No lead found for email '{get_email}'.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")


def main():
    """Main entry point of the application."""
    try:
        manager = ReadLeadManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
