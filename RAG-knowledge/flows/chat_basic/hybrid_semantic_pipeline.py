from promptflow import tool
import re
import os
import json

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.models import (
    HybridCountAndFacetMode,
    HybridSearch,
    SearchScoreThreshold,
    VectorizableTextQuery,
    VectorizableImageBinaryQuery,
    VectorizableImageUrlQuery,
    VectorSimilarityThreshold,
)
from openai import AzureOpenAI

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
INDEX_NAME = "rag-search-index-push-03"
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME")
PROJECT_NAME = os.getenv("PROJECT_NAME")

# User-specified parameter
USE_AAD_FOR_SEARCH = False  # Set this to False to use API key for authentication

def authenticate_azure_search(api_key=None, use_aad_for_search=False):
    if use_aad_for_search:
        print("Using AAD for authentication.")
        credential = DefaultAzureCredential()
    else:
        print("Using API keys for authentication.")
        if api_key is None:
            raise ValueError("API key must be provided if not using AAD for authentication.")
        credential = AzureKeyCredential(api_key)
    return credential


def generate_answer(query, context):
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01"
        )
    system_message = f"""
    system:
	You are an AI assistant that helps users answer questions given a specific context. You will be given a context and asked a question based on that context. Your answer should be as precise as possible and should only come from the context.
	You must generate a response in markdown format. You must include the image url for showing the image in the response, if the context corresponding to the answer contains "image_url".
	Please add citation after each sentence when possible in a form "(Source: citation)".
	context: {context}
	user: 
	"""
    message_text = [
		{"role":"system","content": system_message},
		{"role":"user","content": query}
	]
    completion = client.chat.completions.create(
		model="gpt-4o", # model = "deployment_name"
		messages = message_text,
		# response_format={"type": "json_object"},
		temperature=0,
		)
    return completion.choices[0].message.content

def generate_rephrase_query(text):
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01"
        )

    system_message = """
    # Your Task
    - Given the following conversation history and the users next question,rephrase the question to be a stand alone question.
    - You must output json format.

    # Json format example:
    {
        "questions": [
            "rephrase question content ....",
        ]
    }
    """

    message_text = [
		{"role":"system","content": system_message},
		{"role":"user","content": text}
	]
    completion = client.chat.completions.create(
		model="gpt-4o-mini", # model = "deployment_name"
		messages = message_text,
		response_format={"type": "json_object"},
		temperature=0,
		)
    return completion.choices[0].message.content

# Define Hybrid search + Semantic Ranker pipeline
def hybrid_semantic_pipeline(search_client, search_text):
    # Rephrase the search text
    search_text_rephrased = json.loads(generate_rephrase_query(search_text))["questions"][0]
    vector_query = VectorizableTextQuery(
        text=search_text_rephrased,
        k_nearest_neighbors=50,
        fields="vector",
    )
    
    # Perform the search
    results = search_client.search(
        query_type='semantic',
        query_language='ja',
        semantic_configuration_name='my-semantic-config',
        search_text=search_text_rephrased,
        vector_queries=[vector_query],
        top=5,
        select="content, title, image_url",
        search_fields=["content", "title", "key_phrases"],
    )
    
    # Collecting search results
    context_text = ""
    retrieved_results = []
    for result in results:
        context_text += result["content"] + " "
        retrieved_results.append({
            "title": result.get("title"),
            "content": result.get("content"),
            "image_url": result.get("image_url")
        })
    
    # Generate the final answer
    final_answer = generate_answer(search_text_rephrased, context_text)
    
    # Return a dictionary with prompt, retrieved results, and the final answer
    return {
        "prompt": search_text,
        "rephrased_prompt": search_text_rephrased,
        "retrieved_results": retrieved_results,
        "final_answer": final_answer
    }



# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(question: str) -> str:
    azure_search_credential = authenticate_azure_search(api_key=AZURE_SEARCH_ADMIN_KEY, use_aad_for_search=USE_AAD_FOR_SEARCH)
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=azure_search_credential)
    result = hybrid_semantic_pipeline(search_client, question)

    print(f"result: {result}")

    return result