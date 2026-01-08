"""Web search tool using Google Vertex AI with Google Search."""

import os
import re
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger

try:
    import google.genai as genai
    from google.genai import types
    from google.oauth2 import service_account
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False


class WebSearchTool:
    """Tool for performing web searches using Google Vertex AI with Google Search."""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for real-time information"
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self) -> bool:
        """Initialize the Vertex AI client."""
        if not VERTEXAI_AVAILABLE:
            logger.error("Google GenAI package not available")
            return False
            
        try:
            # Get configuration from environment or defaults
            service_account_file = os.getenv("VERTEXAI_SERVICE_ACCOUNT_FILE", "vertexai-client.json")
            project = os.getenv("VERTEXAI_PROJECT", "glados-483715")
            location = os.getenv("VERTEXAI_LOCATION", "us-central1")
            
            # Check if service account file exists
            if not os.path.exists(service_account_file):
                logger.error(f"Service account file not found: {service_account_file}")
                return False
            
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials
            )
            
            logger.success(f"Vertex AI client initialized for project: {project}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client: {e}")
            return False
    
    def search_web(self, query: str, max_words: int = 100) -> Dict[str, Any]:
        """
        Perform web search using Vertex AI with Google Search.
        
        Args:
            query: Search query string
            max_words: Maximum words in response (default: 100)
            
        Returns:
            Dictionary with search results or error
        """
        if not self.client:
            return {"success": False, "error": "Vertex AI client not initialized"}
        
        try:
            # Create search tool
            search_tool = types.Tool(google_search=types.GoogleSearch())
            
            # Create optimized prompt for concise, accurate responses
            system_prompt = f"""
            You are GLaDOS providing web search results. Be concise, factual, and accurate.
            Maximum {max_words} words. Focus on most relevant, CURRENT information.
            First always reply with valid information, after that you can use your characteristic dry tone for a final remark.
            Refrain yourself from replying with symbols. Per example "°C" should be "degrees Celsius". January 9th should be "January nineth".
            Default to European date formats and metrics.
            """
            
            # Generate content with search
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\n{query}",
                config=types.GenerateContentConfig(
                    tools=[search_tool],
                    temperature=0.1  # Lower temperature for factual responses
                )
            )
            
            if response.text:
                logger.success(f"Web search completed for query: '{query}'")
                return {
                    "success": True,
                    "query": query,
                    "response": response.text.strip(),
                    "sources": getattr(response, 'candidates', [])
                }
            else:
                return {"success": False, "error": "No response received"}
                
        except Exception as e:
            error_msg = f"Web search failed: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def parse_search_command(self, command: str) -> Dict[str, Any]:
        """
        Parse search command and extract query.
        
        Examples:
        - "search the web for weather in tokyo japan tomorrow"
        - "search web for nba scores mavericks vs lakers"
        - "search the web for hot topics in stock market"
        """
        command_lower = command.lower()
        
        # Pattern to extract query after search phrases
        search_patterns = [
            r"search\s+the\s+web\s+for\s+(.+)",
            r"search\s+web\s+for\s+(.+)",
            r"search\s+for\s+(.+)",
            r"web\s+search\s+(.+)",
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, command_lower)
            if match:
                query = match.group(1).strip()
                return {"success": True, "query": query}
        
        return {"success": False, "error": "Could not extract search query from command"}
    
    def execute(self, command: str) -> Dict[str, Any]:
        """Execute web search command."""
        logger.info(f"Web search executing command: {command}")
        
        # Parse the command to extract query
        parsed = self.parse_search_command(command)
        if not parsed["success"]:
            return parsed
        
        query = parsed["query"]
        
        # Perform the search
        result = self.search_web(query)
        
        if result.get("success"):
            logger.success(f"Web search successful: {query}")
        else:
            logger.error(f"Web search failed: {result.get('error')}")
        
        return result


# Test the tool
if __name__ == "__main__":
    tool = WebSearchTool()
    
    test_commands = [
        "search the web for weather in tokyo japan tomorrow",
        "search web for nba scores mavericks vs lakers",
        "search the web for hot topics in stock market"
    ]
    
    for cmd in test_commands:
        print(f"Testing: {cmd}")
        result = tool.execute(cmd)
        print(f"Result: {result.get('success', False)}")
        if result.get("success"):
            print(f"Response: {result.get('response', 'No response')}")
        print()
