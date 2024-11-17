# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'





from flask import Flask, request, jsonify
from googlesearch import search
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Create the LLM definition
def create_llm_definition():
    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

# Process the link and extract the text from the web page
def processing_link_to_text(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    page_text = soup.get_text(separator='\n', strip=True)
    return page_text

# Search for device-related links
def get_links(device_name):
    links = []
    params = {
        "q": device_name,
        "api_key": os.environ.get("SERP_API_KEY"),
        "gl": "in",
        "hl": "en" 
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    for result in results.get("organic_results", []):
        links.append(result.get("link"))
    
    return links

# Process the text to extract information
def processing_text_to_info(input, device_name):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "extract four things only related to {device_name} from input text description. first device name, second device price, third any available offers, fourth website name"),
            ("human", "{input}"),
        ]
    )
    llm = create_llm_definition()
    chain = prompt | llm
    result = chain.invoke(
        {
            "device_name": device_name,
            "input": input,
        }
    )
    return result.content

# Flask app
app = Flask(__name__)

# Endpoint to process the device name
@app.route('/process_device', methods=['POST'])
def process_device():
    data = request.json
    device_name = data.get('device_name', '')

    if not device_name:
        return jsonify({'error': 'Device name is required'}), 400

    # Step 1: Get links related to the device name
    links = get_links(device_name)

    if not links:
        return jsonify({'error': 'No links found for the device'}), 404

    # Step 2: Process the first link and extract text
    link = links[0]
    text = processing_link_to_text(link)

    # Step 3: Extract relevant information from the text
    result = processing_text_to_info(text[:1200], device_name)

    # Return the result as a JSON response
    return jsonify({'device_info': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
