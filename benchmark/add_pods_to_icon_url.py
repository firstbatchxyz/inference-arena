from mongo_client import Mongo
import os

mongo = Mongo(os.getenv("MONGODB_URL"))


pods = mongo.get_collection("pod_benchmarks")
pods = pods.find({"icon_url": {"$exists": False}})

for pod in pods:
    if "qwen" in pod["llm_common_name"].lower():
        pod["icon_url"] = "https://res.cloudinary.com/dr1oufadv/image/upload/v1753126574/qwen_logo_b7jdiv.png"
    elif "llama" in pod["llm_common_name"].lower():
        pod["icon_url"] = "https://res.cloudinary.com/dr1oufadv/image/upload/v1753126411/pngwing.com_tdnvun.png"
    elif "deepseek" in pod["llm_common_name"].lower():
        pass
    elif "gemma" in pod["llm_common_name"].lower():
        pod["icon_url"] = "https://res.cloudinary.com/dr1oufadv/image/upload/v1753126758/google_m7u9zv.png"
    elif "mistral" in pod["llm_common_name"].lower():
        pod["icon_url"] = "https://res.cloudinary.com/dr1oufadv/image/upload/v1753126885/mistral_otrhoz.png"

    mongo.update_one("pod_benchmarks", {"_id": pod["_id"]}, {"$set": {"icon_url": pod["icon_url"]}})
    print(pod["_id"])
    
    

