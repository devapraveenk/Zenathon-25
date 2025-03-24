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
    """Configuration for the CRM product Manager"""

    api_url_base: str
    api_token: str
    openai_api_key: str
    groq_api_key: str
    openai_model: str
    groq_model: str
    module_id: str


class DeleteProductManager:
    """A class to manage CRM products with API integration and deletion capability"""

    def __init__(self):
        """Initialize the product Manager with configuration from environment variables"""
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
                "PRODUCT_MODEL_ID", "28d2bccb-f6d1-4bb7-8984-e641608f4e58"
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
        self.name_to_product: Dict[str, Dict] = {}
        self.name_to_product_id: Dict[str, str] = {}

        # Load initial data
        self._refresh_product_data()

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

    def _refresh_product_data(self) -> None:
        """Refresh the cached product data"""
        product_data = self.get_products()
        if product_data:
            self.name_to_product = self._preprocess_products(product_data)
            self.name_to_product_id = self._preprocess_product_ids(product_data)

    def get_products(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all products from the API

        Returns:
            A dictionary containing product data or None if the request failed
        """
        url = f"{self.config.api_url_base}/api/v1/crm/get_all_product?mdl={self.config.module_id}&&role=undefined"

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

    def _preprocess_products(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Preprocess product data for efficient name-based lookup

        Args:
            data: The raw product data from the API

        Returns:
            A dictionary mapping first names to product data
        """
        desired_keys = [
            "product_name",
            "product_type",
            "description",
            "currency",
            "status",
            "unit_price",
        ]

        name_to_product = {}
        try:
            for product in data.get("data", []):
                product_name = product.get("product_name")
                filtered_product = {
                    key: product[key] for key in desired_keys if key in product
                }
                if product_name:
                    name_to_product[product_name] = filtered_product
            return name_to_product
        except Exception as e:
            print(f"Error preprocessing product data: {str(e)}")
            return {}

    def _preprocess_product_ids(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess product data to map names to product IDs

        Args:
            data: The raw product data from the API

        Returns:
            A dictionary mapping first names to product IDs
        """
        name_to_product_id = {}
        try:
            for product in data.get("data", []):
                product_name = product.get("product_name", "noproduct_name")
                product_id = product.get("product_id")
                if product_name and product_id:
                    name_to_product_id[product_name] = product_id
            return name_to_product_id
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

    def find_product_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a product by first name

        Args:
            name: The first name to search for

        Returns:
            The product data if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_product.get(name)

    def get_product_id_by_name(self, name: str) -> Optional[str]:
        """
        Get a product ID by first name

        Args:
            name: The first name to search for

        Returns:
            The product ID if found, None otherwise
        """
        if not name:
            return None

        return self.name_to_product_id.get(name)

    def analyze_product_details(self, product_data: Any) -> str:
        """
        Analyze product details using an LLM

        Args:
            product_data: The product data to analyze

        Returns:
            A human-readable analysis of the product data
        """
        if not product_data:
            return "No product details found."

        if isinstance(product_data, dict):
            product_data = json.dumps(product_data, indent=4)

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
            result = chain.invoke({"details": product_data})
            return result.content
        except Exception as e:
            return f"An error occurred while analyzing product details: {str(e)}"

    def delete_product(self, name: str) -> bool:
        """
        Delete a product by first name

        Args:
            name: The first name of the product to delete

        Returns:
            True if the product was deleted successfully, False otherwise
        """
        if not name:
            print("No name provided for deletion")
            return False

        product_id = self.get_product_id_by_name(name)
        if not product_id:
            print(f"No product ID found for name: {name}")
            return False

        url = f"{self.config.api_url_base}/api/v1/crm/delete_product?mdl={self.config.module_id}&&id={product_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()

            # Refresh cached data after deletion
            self._refresh_product_data()

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error deleting product: {str(e)}")
            return False

    def run_interactive_session(self) -> None:
        """Run an interactive session allowing users to search and delete products"""
        print("Starting product Manager Interactive Session")

        if not self.name_to_product_id:
            print("No products found. Please check your API connection.")
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
                    user_input, list(self.name_to_product_id.keys())
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
                        product_id = self.get_product_id_by_name(selected_name)

                        if not product_id:
                            print(
                                f"Error: Could not find product ID for {selected_name}"
                            )
                            continue

                        print(f"\n=== product to Delete ===")
                        print(f"Selected name: {selected_name}")

                        product_data = self.find_product_by_name(selected_name)
                        if product_data:
                            analysis = self.analyze_product_details(product_data)
                            print(analysis)

                            confirmation = input(
                                "\nDo you want to Delete this product? (y/n): "
                            ).lower()
                            if confirmation == "y":
                                success = self.delete_product(selected_name)
                                if success:
                                    print("product successfully deleted.")
                                    # Ask if user wants to delete another product
                                    continue_choice = input(
                                        "Delete another product? (y/n): "
                                    )
                                    if continue_choice.lower() != "y":
                                        break
                                else:
                                    print("Failed to delete product.")
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
        manager = DeleteProductManager()
        manager.run_interactive_session()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
