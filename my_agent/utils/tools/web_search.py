from langchain.tools import tool
import requests
from requests.exceptions import RequestException, Timeout, HTTPError, ConnectionError
import os
from bs4 import BeautifulSoup
from typing import Dict, Any

from my_agent.utils.config import get_config


# load config
default_web_config: Dict[str, Any] = {
    'ca_bundle': '/etc/ssl/certs/ca-certificates.crt',
    'timeout': 10,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

config = get_config()
web_config: Dict[str, Any] = config.get_func_tools_config().get('web_search', default_web_config)


if not os.path.exists(web_config['ca_bundle']):
    raise FileNotFoundError(
        f"System CA bundle not found at {web_config['ca_bundle']}")


def html2text(response_text: str) -> str:
    """Convert HTML content to plain text."""
    soup = BeautifulSoup(response_text, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    cleaned_lines = (line for line in lines if line)
    text = '\n'.join(' '.join(line.split()) for line in cleaned_lines)

    return text


@tool
def web_search(url: str) -> str:
    """Perform a web search and return the content of the page.

    Args:
        url: The URL of the web page to fetch.
    """

    try:
        headers = {
            "User-Agent": web_config.get('user_agent')
        }
        response = requests.get(url, headers=headers,
                                timeout=web_config.get('timeout'), verify=web_config.get('ca_bundle'))
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        text_content = html2text(response.text)
        return f"Response Status: {response.status_code}\nContent Length: {len(text_content)} characters\nContent: {text_content}"

    except Timeout:
        return "The request timed out. Please try again later."
    except ConnectionError:
        return "Failed to establish a connection. Please check your network or the URL."
    except HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except RequestException as err:
        return f"An error occurred while fetching the URL: {err}"
