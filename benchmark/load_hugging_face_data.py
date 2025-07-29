# from datasets import load_dataset
# from mongo_client import Mongo
# import os
# import tiktoken  # For OpenAI models
# # Alternative: from transformers import AutoTokenizer  # For Hugging Face models

# def load_hugging_face_data():
#     mongo_client=Mongo(os.getenv("MONGODB_URL"))
#     dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")
#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or "gpt-4"
    
#     for row in dataset:
#         tokens = encoding.encode(row["prompt"])
#         token_size = len(tokens)
        
#         mongo_client.insert_one("benchmark_test_data",{
#             "prompt":row["prompt"],
#             "token_size":token_size
#         })
        
        
        
        
        

# load_hugging_face_data()