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
    """Configuration for the CRM opportunity Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    module_id: str


class DeleteOpportunityManager:
    """A class to manage CRM opportunitys with API integration and deletion capability"""

    def __init__(self):
        """Initialize the opportunity Manager with configuration from environment variables"""
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
                "OPPORTUNITY_MODEL_ID", "5bab99cc-ee9b-49b4-bfe7-32239b1bc28c"
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
        self.name_to_opportunity: Dict[str, Dict] = {}
        self.name_to_opportunity_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_opportunity_data()

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

    def _refresh_opportunity_data(self) -> None:
        """Refresh the cached opportunity data"""
        opportunity_data = self.get_opportunitys()
        if opportunity_data:
            self.name_to_opportunity = self._preprocess_opportunitys(opportunity_data)
            self.name_to_opportunity_id = self._preprocess_opportunity_ids(
                opportunity_data
            )

    def get_opportunitys(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all opportunitys from the API

        Returns:
            A dictionary containing opportunity data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_opportunity?mdl={self.config.module_id}&&role=undefined"

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

    def _preprocess_opportunitys(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess opportunity data for efficient name-based lookup

        Args:
            data: The raw opportunity data from the API

        Returns:
            A dictionary mapping first names to opportunity data
        """
        desired_keys = [
            "opportunity_name",
            "opportunity_value",
            "probability",
            "status",
            "recurring",
        ]

        name_to_opportunity = {}
        try:
            for opportunity in data.get("data", []):
                opportunity_name = opportunity.get("opportunity_name")
                filtered_opportunity = {
                    key: opportunity[key] for key in desired_keys if key in opportunity
                }
                if opportunity_name:
                    name_to_opportunity[opportunity_name] = filtered_opportunity
            return name_to_opportunity
        except Exception as e:
            print(f"Error preprocessing opportunity data: {str(e)}")
            return {}

    def _preprocess_opportunity_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess opportunity data to map names to opportunity IDs

        Args:
            data: The raw opportunity data from the API

        Returns:
            A dictionary mapping first names to opportunity IDs
        """
        name_to_opportunity_id = {}
        try:
            for opportunity in data.get("data", []):
                opportunity_name = opportunity.get(
                    "opportunity_name", "noopportunity_name"
                )
                opportunity_id = opportunity.get("opportunity_id")
                if opportunity_name and opportunity_id:
                    name_to_opportunity_id[opportunity_name] = opportunity_id
            return name_to_opportunity_id
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

    def find_opportunity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a opportunity by first name

        Args:
            name: The first name to search for

        Returns:
            The opportunity data if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_opportunity.get(name)

    def get_opportunity_id_by_name(self, name: str) -> Optional[str]:
        """
        Get a opportunity ID by first name

        Args:
            name: The first name to search for

        Returns:
            The opportunity ID if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_opportunity_id.get(name)

    def analyze_opportunity_details(self, opportunity_data: Any) -> str:
        """
        Analyze opportunity details using an LLM

        Args:
            opportunity_data: The opportunity data to analyze

        Returns:
            A human-readable analysis of the opportunity data
        """
        if not opportunity_data:
            return "No opportunity details found."

        if isinstance(opportunity_data, dict):
            opportunity_data = json.dumps(opportunity_data, indent=4)

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
            result = chain.invoke({"details": opportunity_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing opportunity details: {str(e)}"

    def delete_opportunity(self, name: str) -> bool:
        """
        Delete a opportunity by first name

        Args:
            name: The first name of the opportunity to delete

        Returns:
            True if the opportunity was deleted successfully, False otherwise
        """
        if not name:
            print("No name provided for deletion")
            return False

        opportunity_id = self.get_opportunity_id_by_name(name)
        if not opportunity_id:
            print(f"No opportunity ID found for name: {name}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/delete_opportunity?mdl={self.config.module_id}&&id={opportunity_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after deletion
            self._refresh_opportunity_data()

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error deleting opportunity: {str(e)}")
            return False

    def run_interactive_session(self) -> None:
        """Run an interactive session allowing users to search and delete opportunitys"""
        print("Starting opportunity Manager Interactive Session")

        if not self.name_to_opportunity_id:
            print("No opportunitys found. Please check your API connection.")
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
                    user_input, list(self.name_to_opportunity_id.keys())
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
                        opportunity_id = self.get_opportunity_id_by_name(selected_name)

                        if not opportunity_id:
                            print(
                                f"Error: Could not find opportunity ID for {selected_name}"
                            )
                            continue

                        print(f"\n=== opportunity to Delete ===")
                        print(f"Selected name: {selected_name}")

                        opportunity_data = self.find_opportunity_by_name(selected_name)
                        if opportunity_data:
                            analysis = self.analyze_opportunity_details(
                                opportunity_data
                            )
                            print(analysis)

                            confirmation = input(
                                "\nDo you want to Delete this opportunity? (y/n): "
                            ).lower()
                            if confirmation == "y":
                                success = self.delete_opportunity(selected_name)
                                if success:
                                    print("opportunity successfully deleted.")
                                    # Ask if user wants to delete another opportunity
                                    continue_choice = input(
                                        "Delete another opportunity? (y/n): "
                                    )
                                    if continue_choice.lower() != "y":
                                        break
                                else:
                                    print("Failed to delete opportunity.")
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
        manager = DeleteOpportunityManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
