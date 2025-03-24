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
    """Configuration for the CRM Note Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    module_id: str


class DeleteNoteManager:
    """A class to manage CRM Notes with API integration and deletion capability"""

    def __init__(self):
        """Initialize the Note Manager with configuration from environment variables"""
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
                "NOTE_MODEL_ID", "8caca9a9-7c48-4081-9eeb-170009963a5f"
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
        self.name_to_Note: Dict[str, Dict] = {}
        self.name_to_Note_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_Note_data()

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

    def _refresh_Note_data(self) -> None:
        """Refresh the cached Note data"""
        Note_data = self.get_Notes()
        if Note_data:
            self.name_to_Note = self._preprocess_Notes(Note_data)
            self.name_to_Note_id = self._preprocess_Note_ids(Note_data)

    def get_Notes(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all Notes from the API

        Returns:
            A dictionary containing Note data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_all_basic_notes?mdl={self.config.module_id}&&lead_id=null&&opportunity_id=null"

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

    def _preprocess_Notes(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess Note data for efficient name-based lookup

        Args:
            data: The raw Note data from the API

        Returns:
            A dictionary mapping first names to Note data
        """
        desired_keys = ["title", "content", "status"]

        name_to_Note = {}
        try:
            for Note in data.get("data", []):
                title = Note.get("title")
                filtered_Note = {key: Note[key] for key in desired_keys if key in Note}
                if title:
                    name_to_Note[title] = filtered_Note
            return name_to_Note
        except Exception as e:
            print(f"Error preprocessing Note data: {str(e)}")
            return {}

    def _preprocess_Note_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess Note data to map names to Note IDs

        Args:
            data: The raw Note data from the API

        Returns:
            A dictionary mapping first names to Note IDs
        """
        name_to_Note_id = {}
        try:
            for Note in data.get("data", []):
                title = Note.get("title", "notitle")
                Note_id = Note.get("note_id")
                if title and Note_id:
                    name_to_Note_id[title] = Note_id
            return name_to_Note_id
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

    def find_Note_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a Note by first name

        Args:
            name: The first name to search for

        Returns:
            The Note data if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_Note.get(name)

    def get_Note_id_by_name(self, name: str) -> Optional[str]:
        """
        Get a Note ID by first name

        Args:
            name: The first name to search for

        Returns:
            The Note ID if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_Note_id.get(name)

    def analyze_Note_details(self, Note_data: Any) -> str:
        """
        Analyze Note details using an LLM

        Args:
            Note_data: The Note data to analyze

        Returns:
            A human-readable analysis of the Note data
        """
        if not Note_data:
            return "No Note details found."

        if isinstance(Note_data, dict):
            Note_data = json.dumps(Note_data, indent=4)

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
            result = chain.invoke({"details": Note_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing Note details: {str(e)}"

    def delete_Note(self, name: str) -> bool:
        """
        Delete a Note by first name

        Args:
            name: The first name of the Note to delete

        Returns:
            True if the Note was deleted successfully, False otherwise
        """
        if not name:
            print("No name provided for deletion")
            return False

        Note_id = self.get_Note_id_by_name(name)
        if not Note_id:
            print(f"No Note ID found for name: {name}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/delete_basic_notes?mdl={self.config.module_id}&&id={Note_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after deletion
            self._refresh_Note_data()

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error deleting Note: {str(e)}")
            return False

    def run_interactive_session(self) -> None:
        """Run an interactive session allowing users to search and delete Notes"""
        print("Starting Note Manager Interactive Session")

        if not self.name_to_Note_id:
            print("No Notes found. Please check your API connection.")
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
                    user_input, list(self.name_to_Note_id.keys())
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
                        Note_id = self.get_Note_id_by_name(selected_name)

                        if not Note_id:
                            print(f"Error: Could not find Note ID for {selected_name}")
                            continue

                        print(f"\n=== Note to Delete ===")
                        print(f"Selected name: {selected_name}")

                        Note_data = self.find_Note_by_name(selected_name)
                        if Note_data:
                            analysis = self.analyze_Note_details(Note_data)
                            print(analysis)

                            confirmation = input(
                                "\nDo you want to Delete this Note? (y/n): "
                            ).lower()
                            if confirmation == "y":
                                success = self.delete_Note(selected_name)
                                if success:
                                    print("Note successfully deleted.")
                                    # Ask if user wants to delete another Note
                                    continue_choice = input(
                                        "Delete another Note? (y/n): "
                                    )
                                    if continue_choice.lower() != "y":
                                        break
                                else:
                                    print("Failed to delete Note.")
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
        manager = DeleteNoteManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
