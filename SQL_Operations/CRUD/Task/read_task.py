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
    """Configuration for the CRM task Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    module_id: str


class ReadTaskManager:
    """A class to manage CRM tasks with API integration and Search capability"""

    def __init__(self):
        """Initialize the task Manager with configuration from environment variables"""
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
                "TASK_MODEL_ID", "8caca9a9-7c48-4081-9eeb-170009963a5f"
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
        self.name_to_task: Dict[str, Dict] = {}
        self.name_to_task_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_task_data()

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

    def _refresh_task_data(self) -> None:
        """Refresh the cached task data"""
        task_data = self.get_tasks()
        if task_data:
            self.name_to_task = self._preprocess_tasks(task_data)
            self.name_to_task_id = self._preprocess_task_ids(task_data)

    def get_tasks(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all tasks from the API

        Returns:
            A dictionary containing task data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_all_basic_task?mdl={self.config.module_id}&&lead_id=null&&opportunity_id=null"

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

    def _preprocess_tasks(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess task data for efficient name-based lookup

        Args:
            data: The raw task data from the API

        Returns:
            A dictionary mapping first names to task data
        """
        desired_keys = [
            "title",
            "description",
            "type",
            "task_status",
            "big_status",
            "note_text",
        ]

        name_to_task = {}
        try:
            for task in data.get("data", []):
                title = task.get("title")
                filtered_task = {key: task[key] for key in desired_keys if key in task}
                if title:
                    name_to_task[title] = filtered_task
            return name_to_task
        except Exception as e:
            print(f"Error preprocessing task data: {str(e)}")
            return {}

    def _preprocess_task_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess task data to map names to task IDs

        Args:
            data: The raw task data from the API

        Returns:
            A dictionary mapping first names to task IDs
        """
        name_to_task_id = {}
        try:
            for task in data.get("data", []):
                title = task.get("title", "notitle")
                task_id = task.get("task_id")
                if title and task_id:
                    name_to_task_id[title] = task_id
            return name_to_task_id
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

    def find_task_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a task by first name

        Args:
            name: The first name to search for

        Returns:
            The task data if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_task.get(name)

    def get_task_id_by_name(self, name: str) -> Optional[str]:
        """
        Get a task ID by first name

        Args:
            name: The first name to search for

        Returns:
            The task ID if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_task_id.get(name)

    def analyze_task_details(self, task_data: Any) -> str:
        """
        Analyze task details using an LLM

        Args:
            task_data: The task data to analyze

        Returns:
            A human-readable analysis of the task data
        """
        if not task_data:
            return "No task details found."

        if isinstance(task_data, dict):
            task_data = json.dumps(task_data, indent=4)

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
            result = chain.invoke({"details": task_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing task details: {str(e)}"

    def get_task(self, name: str) -> bool:
        """
        get a task by first name

        Args:
            name: The first name of the task to get

        Returns:
            True if the task was get successfully, False otherwise
        """
        if not name:
            print("No name provided for Search")
            return False

        task_id = self.get_task_id_by_name(name)
        if not task_id:
            print(f"No task ID found for name: {name}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/get_basic_task?mdl={self.config.module_id}&&id={task_id}"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after Search
            self._refresh_task_data()

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error Searching task: {str(e)}")
            return False

    def run_interactive_session(self) -> None:
        """Run an interactive session allowing users to search and get tasks"""
        print("Starting task Manager Interactive Session")

        if not self.name_to_task_id:
            print("No tasks found. Please check your API connection.")
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
                    user_input, list(self.name_to_task_id.keys())
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
                        task_id = self.get_task_id_by_name(selected_name)

                        if not task_id:
                            print(f"Error: Could not find task ID for {selected_name}")
                            continue

                        print(f"\n=== task to get ===")
                        print(f"Selected name: {selected_name}")

                        task_data = self.find_task_by_name(selected_name)
                        if task_data:
                            analysis = self.analyze_task_details(task_data)
                            print(analysis)

                            # Ask if user wants to get another task
                            continue_choice = input("get another task? (y/n): ")
                            if continue_choice.lower() != "y":
                                break

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
        manager = ReadTaskManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
