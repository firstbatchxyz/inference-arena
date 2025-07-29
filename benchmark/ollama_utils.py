import requests
import time
import json

def verify_model_availability(pod_url, llm_id,):
    print("Verifying model availability...")
    max_retries = 1000
    retry_delay = 5
    
    for retry in range(max_retries):
        try:
            response = requests.get(pod_url+"/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                print(f"DEBUG: Models response: {models_data}")  # Debug output
                
                models = models_data.get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                
                # Check for the model (handle both with and without tag)
                model_found = False
                for model_name in model_names:
                    if llm_id in model_name or model_name in llm_id:
                        model_found = True
                        print(f"Model {llm_id} is available and ready (found as {model_name})")
                        return True
                
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return False
            else:
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return False
                    
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False


def pull_model(pod_url, llm_id):
    print(f"Starting to pull model {llm_id}...")
    
    # Use streaming to monitor download progress
    download_completed = False
    try:
        with requests.post(pod_url+"/api/pull", json={"model": llm_id}, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            status = data.get('status', '')
                            
                            # Print download progress
                            if 'total' in data and 'completed' in data:
                                total = data['total']
                                completed = data['completed']
                                percent = (completed / total * 100) if total > 0 else 0
                            else:
                                pass
                            
                            # Check if download is complete
                            if status == "success" or "success" in str(data):
                                download_completed = True
                                break
                                
                        except json.JSONDecodeError as e:
                            continue
            else:
                print(f"Failed to pull model {llm_id}: Status code {response.status_code}")
                ## Retry the pull
                pull_model(pod_url, llm_id)
                return
                
    except Exception as e:
        print(f"Error pulling model: {e}")
        return
    
    if not download_completed:
        print("WARNING: Download may not have completed properly")
    
    # Wait a bit for the model to be fully loaded
    print("Waiting for model to be fully loaded...")
    time.sleep(1)
