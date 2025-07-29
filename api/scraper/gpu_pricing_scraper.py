import requests
import json
from openai import OpenAI
import os
from datetime import datetime, UTC
from dotenv import load_dotenv
from benchmark.mongo_client import Mongo

mongo_client = Mongo(os.getenv("MONGODB_URL"))

load_dotenv()

def scrape_gpu_prices():
    list_of_scraped_urls = [
        {"url":"https://www.runpod.io/pricing","platform":"runpod"},
        {"url":"https://www.hyperbolic.ai/marketplace","platform":"hyperbolic"}, 
        {"url":"https://lambda.ai/pricing","platform":"lambda"},
        {"url":"https://www.primeintellect.ai/","platform":"primeintellect"},
        {"url":"https://cloud.google.com/compute/gpus-pricing","platform":"google"},
        {"url":"https://vast.ai/pricing","platform":"vast.ai"},
        {"url":"https://akash.network/pricing/gpus/","platform":"akash"},
        {"url":"https://www.scaleway.com/en/gpu-instances/","platform":"scaleway"},
        {"url":"https://nebius.com/prices","platform":"nebius"},
    ]

    # Get Jina API key from environment variable
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY environment variable is required")
    
    headers = {
        'Authorization': f'Bearer {jina_api_key}'
    }

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for url_item in list_of_scraped_urls:
        try:
            print(f"Scraping: {url_item}")
            url = url_item["url"]
            platform_name = url_item["platform"]
            
            # Get content from Jina Reader API
            jina_response = requests.get('https://r.jina.ai/' + url, headers=headers)
            
            if jina_response.status_code != 200:
                print(f"Failed to fetch {url}: {jina_response.status_code}")
                continue

            # Use OpenAI to extract GPU pricing data
            prompt = f"""
            Extract GPU pricing information from the following webpage content. 
            Return ONLY a valid JSON array of objects with these exact keys:
            - name: GPU name (clean, without extra text like "active" or "available")
            - price: price per hour as FLOAT number only (e.g., 0.69, not "$0.69 /hr")
            - url: full URL to the GPU page
            - platform: "{platform_name}"
            
            Webpage content:
            {jina_response.text[:8000]}  # Limit content to avoid token limits
            
            Return only the JSON array, nothing else.
            """

            openai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a data extraction expert. Return only valid JSON arrays with GPU pricing data."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1
            )

            # Parse the response content
            response_text = openai_response.choices[0].message.content.strip()
            
            # Clean up the response - remove any markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            try:
                # Parse JSON
                gpu_data = json.loads(response_text)
                
                if not isinstance(gpu_data, list):
                    print(f"Expected list but got: {type(gpu_data)}")
                    continue
                
                print(f"Extracted {len(gpu_data)} GPU records from {platform_name}")
                
                # Insert each GPU record into MongoDB
                for item in gpu_data:
                    # Add metadata
                    item['scraped_at'] = datetime.now(UTC)
                    item['source_url'] = url
                    
                    # Clean up the data
                    if 'name' in item and item['name']:
                        item['name'] = item['name'].strip()
                    if 'price' in item and item['price']:
                        # Ensure price is a float
                        try:
                            if isinstance(item['price'], str):
                                # Remove currency symbols and text, keep only numbers
                                price_str = item['price'].replace('$', '').replace('/hr', '').replace('/hour', '').strip()
                                item['price'] = float(price_str)
                            elif isinstance(item['price'], (int, float)):
                                item['price'] = float(item['price'])
                            else:
                                item['price'] = 0.0
                        except (ValueError, TypeError):
                            item['price'] = 0.0
                    
                    # Insert into MongoDB
                    try:
                        mongo_client.insert_one("gpu_prices", item)
                        print(f"Inserted: {item.get('name', 'Unknown')} - {item.get('price', 'N/A')}")
                    except Exception as e:
                        print(f"Failed to insert {item}: {e}")
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from {platform_name}: {e}")
                print(f"Response was: {response_text[:200]}...")
                continue
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    print("GPU scraping completed!")

def get_gpu_prices_from_mongo():
    """Retrieve GPU prices from MongoDB"""
    try:
        # Get all GPU prices
        gpu_prices = list(mongo_client.find_many("gpu_prices", {}))
        
        print(f"Found {len(gpu_prices)} GPU price records:")
        for price in gpu_prices:
            print(f"- {price.get('name', 'Unknown')}: {price.get('price', 'N/A')} on {price.get('platform', 'Unknown')}")
        
        return gpu_prices
        
    except Exception as e:
        print(f"Error retrieving GPU prices: {e}")
        return []

if __name__ == "__main__":
    scrape_gpu_prices()
    # Uncomment to test retrieval
    # get_gpu_prices_from_mongo()