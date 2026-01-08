import google.genai as genai
from google.genai import types
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = "vertexai-client.json"  # JSON from above

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]  # important!
)

client = genai.Client(
    vertexai=True,
    project="glados-483715",
    location="us-central1",
    credentials=credentials
)

search_tool = types.Tool(
    google_search=types.GoogleSearch()
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the weather in Tokyo tomorrow?",
    config=types.GenerateContentConfig(
        tools=[search_tool]
    )
)

print(response.text)

