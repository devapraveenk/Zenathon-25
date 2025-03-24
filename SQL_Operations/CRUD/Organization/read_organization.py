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
    """Configuration for the CRM organization Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    module_id: str


class ReadOrganizationManager:
    """A class to manage CRM organizations with API integration and Search capability"""

    def __init__(self):
        """Initialize the organization Manager with configuration from environment variables"""
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
                "ORGANIZATION_MODEL_ID", "cc8989b5-bb0b-4021-b76d-cf16d8f59841"
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
        self.name_to_organization: Dict[str, Dict] = {}
        self.name_to_organization_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_organization_data()

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

    def _refresh_organization_data(self) -> None:
        """Refresh the cached organization data"""
        organization_data = self.get_organizations()
        if organization_data:
            self.name_to_organization = self._preprocess_organizations(
                organization_data
            )
            self.name_to_organization_id = self._preprocess_organization_ids(
                organization_data
            )

    def get_organizations(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all organizations from the API

        Returns:
            A dictionary containing organization data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_organization?mdl={self.config.module_id}"

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

    def _preprocess_organizations(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess organization data for efficient name-based lookup

        Args:
            data: The raw organization data from the API

        Returns:
            A dictionary mapping first names to organization data
        """
        desired_keys = [
            "organization_name",
            "annual_revenue",
            "no_of_employees",
            "currency",
        ]

        name_to_organization = {}
        try:
            for organization in data.get("data", []):
                organization_name = organization.get("organization_name")
                filtered_organization = {
                    key: organization[key]
                    for key in desired_keys
                    if key in organization
                }
                if organization_name:
                    name_to_organization[organization_name] = filtered_organization
            return name_to_organization
        except Exception as e:
            print(f"Error preprocessing organization data: {str(e)}")
            return {}

    def _preprocess_organization_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess organization data to map names to organization IDs

        Args:
            data: The raw organization data from the API

        Returns:
            A dictionary mapping first names to organization IDs
        """
        name_to_organization_id = {}
        try:
            for organization in data.get("data", []):
                organization_name = organization.get(
                    "organization_name", "noorganization_name"
                )
                organization_id = organization.get("organization_id")
                if organization_name and organization_id:
                    name_to_organization_id[organization_name] = organization_id
            return name_to_organization_id
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

    def find_organization_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a organization by first name

        Args:
            name: The first name to search for

        Returns:
            The organization data if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_organization.get(name)

    def get_organization_id_by_name(self, name: str) -> Optional[str]:
        """
        Get a organization ID by first name

        Args:
            name: The first name to search for

        Returns:
            The organization ID if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_organization_id.get(name)

    def analyze_organization_details(self, organization_data: Any) -> str:
        """
        Analyze organization details using an LLM

        Args:
            organization_data: The organization data to analyze

        Returns:
            A human-readable analysis of the organization data
        """
        if not organization_data:
            return "No organization details found."

        if isinstance(organization_data, dict):
            organization_data = json.dumps(organization_data, indent=4)

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
            result = chain.invoke({"details": organization_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing organization details: {str(e)}"

    def get_organization(self, name: str) -> bool:
        """
        get a organization by first name

        Args:
            name: The first name of the organization to get

        Returns:
            True if the organization was get successfully, False otherwise
        """
        if not name:
            print("No name provided for Search")
            return False

        organization_id = self.get_organization_id_by_name(name)
        if not organization_id:
            print(f"No organization ID found for name: {name}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/get_organization_by_id?mdl={self.config.module_id}&&id={organization_id}"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after Search
            self._refresh_organization_data()

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error Search organization: {str(e)}")
            return False

    def run_interactive_session(self) -> None:
        """Run an interactive session allowing users to search and get organizations"""
        print("Starting organization Manager Interactive Session")

        if not self.name_to_organization_id:
            print("No organizations found. Please check your API connection.")
            return

        while True:
            try:
                user_input = input(
                    "\nEnter a name to get (or type 'exit' to quit): "
                ).strip()

                if user_input.lower() in ["exit", "cancel", "quit"]:
                    print("Exiting the program.")
                    break

                if not user_input:
                    print("Name cannot be empty.")
                    continue

                related_names = self.find_related_names(
                    user_input, list(self.name_to_organization_id.keys())
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
                        organization_id = self.get_organization_id_by_name(
                            selected_name
                        )

                        if not organization_id:
                            print(
                                f"Error: Could not find organization ID for {selected_name}"
                            )
                            continue

                        print(f"\n=== organization to get ===")
                        print(f"Selected name: {selected_name}")

                        organization_data = self.find_organization_by_name(
                            selected_name
                        )
                        if organization_data:
                            analysis = self.analyze_organization_details(
                                organization_data
                            )
                            print(analysis)

                            confirmation = input(
                                "\nDo you want to get this organization? (y/n): "
                            ).lower()
                            if confirmation == "y":
                                success = self.get_organization(selected_name)
                                if success:
                                    print("organization successfully fetched.")
                                    # Ask if user wants to get another organization
                                    continue_choice = input(
                                        "get another organization? (y/n): "
                                    )
                                    if continue_choice.lower() != "y":
                                        break
                                else:
                                    print("Failed to get organization.")
                            else:
                                print("Operation cancelled.")
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
        manager = ReadOrganizationManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
