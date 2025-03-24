import requests
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from fuzzywuzzy import process


@dataclass
class Config:
    """Configuration for the CRM Contact Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    module_id: str


class DeleteContactManager:
    """A class to manage CRM contacts with API integration and deletion capability"""

    def __init__(self):
        """Initialize the Contact Manager with configuration from environment variables"""
        load_dotenv()

        self.config = Config(
            api_url_base=os.getenv("API_URL_BASE", "http://localhost:8088"),
            api_token=os.getenv(
                "API_TOKEN",
                "",
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            groq_model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            module_id=os.getenv(
                "CONTACT_MODEL_ID", "68d8274e-0bc3-4182-8264-9d218524ab68"
            ),
        )

        self._validate_config()

        # Initialize LLM clients
        self.openai_llm = ChatOpenAI(
            model=self.config.openai_model,
            temperature=0.2,
            openai_api_key=self.config.openai_api_key,
        )

        # Pre-processed data cache
        self.name_to_contact: Dict[str, Dict] = {}
        self.name_to_contact_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_contact_data()

    def _validate_config(self) -> None:
        """Validate that all required configuration is present"""
        required_fields = [
            "api_url_base",
            "api_token",
            "openai_api_key",
            "groq_api_key",
            "module_id",
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

    def _refresh_contact_data(self) -> None:
        """Refresh the cached contact data"""
        contact_data = self.get_contacts()
        if contact_data:
            self.name_to_contact = self._preprocess_contacts(contact_data)
            self.name_to_contact_id = self._preprocess_contact_ids(contact_data)

    def get_contacts(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all contacts from the API

        Returns:
            A dictionary containing contact data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_contacts?mdl={self.config.module_id}&&role=undefined"

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

    def _preprocess_contacts(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess contact data for efficient name-based lookup

        Args:
            data: The raw contact data from the API

        Returns:
            A dictionary mapping first names to contact data
        """
        desired_keys = ["first_name", "email", "created_at"]

        name_to_contact = {}
        try:
            for contact in data.get("data", []):
                first_name = contact.get("first_name")
                filtered_contact = {
                    key: contact[key] for key in desired_keys if key in contact
                }
                if first_name:
                    name_to_contact[first_name] = filtered_contact
            return name_to_contact
        except Exception as e:
            print(f"Error preprocessing contact data: {str(e)}")
            return {}

    def _preprocess_contact_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess contact data to map names to contact IDs

        Args:
            data: The raw contact data from the API

        Returns:
            A dictionary mapping first names to contact IDs
        """
        name_to_contact_id = {}
        try:
            for contact in data.get("data", []):
                first_name = contact.get("first_name")
                contact_id = contact.get("contact_id")
                if first_name and contact_id:
                    name_to_contact_id[first_name] = contact_id
            return name_to_contact_id
        except Exception:
            return {}

    def find_related_names(
        self, user_input: str, names: List[str], limit: int = 5
    ) -> List[str]:
        """
        Find names similar to user input using fuzzy matching

        Args:
            user_input: The user's input to search for
            names: List of all available names
            limit: Maximum number of results to return

        Returns:
            List of names that match the input
        """
        if not names:
            return []

        try:
            related_names = process.extract(user_input, names, limit=limit)
            return [name for name, score in related_names]
        except Exception as e:
            print(f"Error in fuzzy matching: {str(e)}")
            return []

    def find_contact_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a contact by first name

        Args:
            name: The first name to search for

        Returns:
            The contact data if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_contact.get(name)

    def get_contact_id_by_name(self, name: str) -> Optional[str]:
        """
        Get a contact ID by first name

        Args:
            name: The first name to search for

        Returns:
            The contact ID if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_contact_id.get(name)

    def analyze_contact_details(self, contact_data: Any) -> str:
        """
        Analyze contact details using an LLM

        Args:
            contact_data: The contact data to analyze

        Returns:
            A human-readable analysis of the contact data
        """
        if not contact_data:
            return "No contact details found."

        if isinstance(contact_data, dict):
            contact_data = json.dumps(contact_data, indent=4)

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
            result = chain.invoke({"details": contact_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing contact details: {str(e)}"

    def delete_contact(self, name: str) -> bool:
        """
        Delete a contact by first name

        Args:
            name: The first name of the contact to delete

        Returns:
            True if the contact was deleted successfully, False otherwise
        """
        if not name:
            print("No name provided for deletion")
            return False

        contact_id = self.get_contact_id_by_name(name)
        if not contact_id:
            print(f"No contact ID found for name: {name}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/delete_contact?mdl={self.config.module_id}&&id={contact_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after deletion
            self._refresh_contact_data()

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error deleting contact: {str(e)}")
            return False

    def run_interactive_session(self) -> None:
        """Run an interactive session allowing users to search and delete contacts"""
        print("Starting Contact Manager Interactive Session")

        if not self.name_to_contact_id:
            print("No contacts found. Please check your API connection.")
            return

        while True:
            try:
                user_input = input(
                    "\nEnter a name to Delete (or type 'exit' to quit): "
                ).strip()

                if user_input.lower() in ["exit", "cancel", "quit"]:
                    print("Exiting the program.")
                    break

                if not user_input:
                    print("Name cannot be empty.")
                    continue

                related_names = self.find_related_names(
                    user_input, list(self.name_to_contact_id.keys())
                )

                if not related_names:
                    print("No matching names found. Please try again.")
                    continue

                print("\nDid you mean one of these names?")
                for i, name in enumerate(related_names, 1):
                    print(f"{i}. {name}")

                choice = input(
                    "\nEnter the number of the correct name (or '0' to try again): "
                ).strip()

                if choice == "0":
                    continue

                try:
                    choice = int(choice)
                    if 1 <= choice <= len(related_names):
                        selected_name = related_names[choice - 1]
                        contact_id = self.get_contact_id_by_name(selected_name)

                        if not contact_id:
                            print(
                                f"Error: Could not find contact ID for {selected_name}"
                            )
                            continue

                        print(f"\n=== Contact to Delete ===")
                        print(f"Selected name: {selected_name}")

                        contact_data = self.find_contact_by_name(selected_name)
                        if contact_data:
                            analysis = self.analyze_contact_details(contact_data)
                            print(analysis)

                            confirmation = input(
                                "\nDo you want to Delete this contact? (y/n): "
                            ).lower()
                            if confirmation == "y":
                                success = self.delete_contact(selected_name)
                                if success:
                                    print("Contact successfully deleted.")
                                    # Ask if user wants to delete another contact
                                    continue_choice = input(
                                        "Delete another contact? (y/n): "
                                    )
                                    if continue_choice.lower() != "y":
                                        break
                                else:
                                    print("Failed to delete contact.")
                            else:
                                print("Deletion cancelled.")
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")


def main():
    """Main entry point of the application."""
    try:
        manager = DeleteContactManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
