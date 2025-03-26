from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search_client = SearchClient(
    endpoint="https://<your-search-endpoint>.search.windows.net",
    index_name="<your-index-name>",
    credential=AzureKeyCredential("<your-api-key>")
)

def search_kb(issue_text):
    results = search_client.search(search_text=issue_text, top=1)
    for result in results:
        return result['kb_article']
    return None