import schedule
import time
import json
import hashlib
from datetime import datetime, timedelta, UTC
from openai import OpenAI
import os
from dotenv import load_dotenv
from benchmark.mongo_client import Mongo
from .gpu_pricing_scraper import scrape_gpu_prices, get_gpu_prices_from_mongo

load_dotenv()

class GPUNormalizer:
    """AI-powered GPU name normalization"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.mongo_client = Mongo(os.getenv("MONGODB_URL"))
        
        # Standard GPU mappings - comprehensive list for cross-platform matching
        self.gpu_mappings = {
            # NVIDIA RTX Series (Consumer)
            "rtx 4090": "RTX 4090",
            "rtx 4090 ti": "RTX 4090 Ti", 
            "rtx 4080": "RTX 4080",
            "rtx 4080 ti": "RTX 4080 Ti",
            "rtx 4080 super": "RTX 4080 Super",
            "rtx 4070": "RTX 4070",
            "rtx 4070 ti": "RTX 4070 Ti",
            "rtx 4070 super": "RTX 4070 Super",
            "rtx 4060": "RTX 4060",
            "rtx 4060 ti": "RTX 4060 Ti",
            "rtx 3090": "RTX 3090",
            "rtx 3090 ti": "RTX 3090 Ti",
            "rtx 3080": "RTX 3080",
            "rtx 3080 ti": "RTX 3080 Ti",
            "rtx 3070": "RTX 3070",
            "rtx 3070 ti": "RTX 3070 Ti",
            "rtx 3060": "RTX 3060",
            "rtx 3060 ti": "RTX 3060 Ti",
            "rtx 3050": "RTX 3050",
            "rtx 5090": "RTX 5090",
            
            # NVIDIA Data Center GPUs
            "a100": "NVIDIA A100",
            "a100 80gb": "NVIDIA A100 80GB",
            "a100 40gb": "NVIDIA A100 40GB",
            "a100 80 gb": "NVIDIA A100 80GB",
            "a100 40 gb": "NVIDIA A100 40GB",
            "h100": "NVIDIA H100",
            "h100 80gb": "NVIDIA H100 80GB",
            "h100 40gb": "NVIDIA H100 40GB",
            "h100 80 gb": "NVIDIA H100 80GB",
            "h100 40 gb": "NVIDIA H100 40GB",
            "h100 sxm": "NVIDIA H100 SXM",
            "h100 pcie": "NVIDIA H100 PCIe",
            "h200": "NVIDIA H200",
            "h200 sxm": "NVIDIA H200 SXM",
            "h200 pcie": "NVIDIA H200 PCIe",
            "b200": "NVIDIA B200",
            "v100": "NVIDIA V100",
            "v100 32gb": "NVIDIA V100 32GB",
            "v100 16gb": "NVIDIA V100 16GB",
            "v100 32 gb": "NVIDIA V100 32GB",
            "v100 16 gb": "NVIDIA V100 16GB",
            "t4": "NVIDIA T4",
            "l4": "NVIDIA L4",
            "l40": "NVIDIA L40",
            "l40s": "NVIDIA L40S",
            "a40": "NVIDIA A40",
            "a30": "NVIDIA A30",
            "a10": "NVIDIA A10",
            "a10g": "NVIDIA A10G",
            "rtx a6000": "NVIDIA RTX A6000",
            "rtx a5000": "NVIDIA RTX A5000",
            "rtx a4000": "NVIDIA RTX A4000",
            "rtx a3000": "NVIDIA RTX A3000",
            "rtx 6000 ada": "NVIDIA RTX 6000 Ada",
            "rtx 5000 ada": "NVIDIA RTX 5000 Ada",
            "rtx 4000 ada": "NVIDIA RTX 4000 Ada",
            "rtx 3000 ada": "NVIDIA RTX 3000 Ada",
            
            # AMD GPUs
            "rx 7900 xtx": "AMD RX 7900 XTX",
            "rx 7900 xt": "AMD RX 7900 XT",
            "rx 7800 xt": "AMD RX 7800 XT",
            "rx 7700 xt": "AMD RX 7700 XT",
            "rx 7600": "AMD RX 7600",
            "rx 7600 xt": "AMD RX 7600 XT",
            "rx 6950 xt": "AMD RX 6950 XT",
            "rx 6900 xt": "AMD RX 6900 XT",
            "rx 6800 xt": "AMD RX 6800 XT",
            "rx 6800": "AMD RX 6800",
            "rx 6750 xt": "AMD RX 6750 XT",
            "rx 6700 xt": "AMD RX 6700 XT",
            "rx 6650 xt": "AMD RX 6650 XT",
            "rx 6600 xt": "AMD RX 6600 XT",
            "rx 6600": "AMD RX 6600",
            
            # AMD Data Center GPUs
            "mi300x": "AMD MI300X",
            "mi300a": "AMD MI300A",
            "mi250x": "AMD MI250X",
            "mi250": "AMD MI250",
            "mi210": "AMD MI210",
            "mi100": "AMD MI100",
            "mi60": "AMD MI60",
            "mi50": "AMD MI50",
            "mi25": "AMD MI25",
            
            # Intel GPUs
            "arc a770": "Intel Arc A770",
            "arc a750": "Intel Arc A750",
            "arc a580": "Intel Arc A580",
            "arc a570": "Intel Arc A570",
            "arc a560": "Intel Arc A560",
            "arc a550": "Intel Arc A550",
            "arc a540": "Intel Arc A540",
            "arc a530": "Intel Arc A530",
            "arc a380": "Intel Arc A380",
            "arc a310": "Intel Arc A310",
            "pvc": "Intel PVC",
            "pvc 128": "Intel PVC 128",
            "pvc 64": "Intel PVC 64"
        }
        
        # Keywords that indicate non-GPU items to exclude
        self.exclude_keywords = [
            "flex", "package", "bundle", "spot", "reserved", "dedicated", "shared",
            "cpu", "memory", "storage", "network", "bandwidth", "instance", "machine",
            "server", "compute", "node", "cluster", "pod", "container", "vm", "virtual",
            "hour", "day", "month", "year", "minute", "second", "time", "duration",
            "active", "available", "inactive", "unavailable", "status", "state",
            "region", "zone", "datacenter", "location", "price", "cost", "billing",
            "subscription", "plan", "tier", "level", "basic", "premium", "pro", "enterprise"
        ]
    
    def is_gpu_item(self, gpu_name: str) -> bool:
        """Check if the item is actually a GPU (not a package, instance, etc.)"""
        if not gpu_name:
            return False
        
        cleaned_name = gpu_name.lower().strip()
        
        # Check if it contains exclude keywords
        for keyword in self.exclude_keywords:
            if keyword in cleaned_name:
                return False
        
        # Check if it contains GPU-related keywords
        gpu_keywords = ["gpu", "rtx", "gtx", "a100", "h100", "v100", "t4", "l4", "a40", "rx", "mi", "arc", "pvc"]
        has_gpu_keyword = any(keyword in cleaned_name for keyword in gpu_keywords)
        
        # If it has GPU keywords, it's likely a GPU
        if has_gpu_keyword:
            return True
        
        # Check if it matches any of our known GPU patterns
        for key in self.gpu_mappings.keys():
            if key in cleaned_name:
                return True
        
        return False
    
    def normalize_gpu_name(self, gpu_name: str) -> str:
        """Normalize GPU name using predefined mappings and AI"""
        if not gpu_name:
            return None
        
        # First check if this is actually a GPU item
        if not self.is_gpu_item(gpu_name):
            return None
        
        # Clean the input
        cleaned_name = gpu_name.lower().strip()
        
        # Try exact mapping first
        if cleaned_name in self.gpu_mappings:
            return self.gpu_mappings[cleaned_name]
        
        # Try partial matching with priority for longer matches
        best_match = None
        best_match_length = 0
        
        for key, value in self.gpu_mappings.items():
            if key in cleaned_name:
                if len(key) > best_match_length:
                    best_match = value
                    best_match_length = len(key)
            elif cleaned_name in key:
                if len(cleaned_name) > best_match_length:
                    best_match = value
                    best_match_length = len(cleaned_name)
        
        if best_match:
            return best_match
        
        # Use AI for complex matching only if it looks like a GPU
        try:
            ai_result = self._ai_normalize_gpu_name(gpu_name)
            # Only return AI result if it's not "Unknown GPU"
            if ai_result and ai_result != "Unknown GPU":
                return ai_result
        except Exception as e:
            print(f"AI normalization failed for '{gpu_name}': {e}")
        
        # If we can't normalize it but it looks like a GPU, return the original
        return gpu_name if self.is_gpu_item(gpu_name) else None
    
    def _ai_normalize_gpu_name(self, gpu_name: str) -> str:
        """Use AI to normalize GPU name"""
        prompt = f"""
        You are a GPU expert. Normalize the following GPU name to a standard format.
        
        Input GPU name: "{gpu_name}"
        
        Please return ONLY the normalized GPU name in a standard format like:
        - RTX 4090, RTX 4080, RTX 4070, etc.
        - NVIDIA A100, NVIDIA H100, NVIDIA V100, etc.
        - AMD RX 7900 XTX, AMD RX 7900 XT, etc.
        - AMD MI300X, AMD MI250X, etc.
        - Intel Arc A770, Intel PVC, etc.
        
        IMPORTANT: Only return GPU names. If the input is a package, instance, or non-GPU item, return "Unknown GPU".
        
        Examples of what to exclude:
        - "Flex Package", "Spot Instance", "Dedicated Server" → "Unknown GPU"
        - "RTX 4090 Spot", "A100 Reserved" → "RTX 4090", "NVIDIA A100"
        
        Normalized name:"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a GPU expert that normalizes GPU names to standard formats."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        normalized = response.choices[0].message.content.strip()
        
        # Clean up the response
        if ":" in normalized:
            normalized = normalized.split(":")[-1].strip()
        
        return normalized if normalized and normalized != "Unknown GPU" else gpu_name
    
    def generate_gpu_id(self, gpu_name: str, platform: str = None) -> str:
        """Generate unique ID for GPU based on normalized name only (for cross-platform comparison)"""
        # Use only the normalized GPU name to allow cross-platform price comparison
        hash_input = gpu_name.lower().strip()
        return hashlib.md5(hash_input.encode()).hexdigest()

def process_and_normalize_gpu_data():
    """Process and normalize existing GPU data in MongoDB"""
    print("Processing and normalizing GPU data...")
    
    normalizer = GPUNormalizer()
    
    # Get all raw GPU data
    raw_gpu_data = list(normalizer.mongo_client.find_many("gpu_prices", {}))
    
    processed_count = 0
    skipped_count = 0
    for gpu_record in raw_gpu_data:
        try:
            original_name = gpu_record.get('name', '')
            platform = gpu_record.get('platform', '')
            
            if not original_name or not platform:
                continue
            
            # Normalize GPU name
            normalized_name = normalizer.normalize_gpu_name(original_name)
            
            # Skip if normalization returned None (not a GPU item)
            if normalized_name is None:
                print(f"Skipped non-GPU item: {original_name}")
                skipped_count += 1
                continue
            
            # Generate unique ID
            gpu_id = normalizer.generate_gpu_id(normalized_name, platform)
            
            # Create processed record
            processed_record = {
                "gpu_id": gpu_id,
                "original_name": original_name,
                "normalized_name": normalized_name,
                "platform": platform,
                "price": gpu_record.get('price', ''),
                "url": gpu_record.get('url', ''),
                "source_url": gpu_record.get('source_url', ''),
                "scraped_at": gpu_record.get('scraped_at', datetime.now(UTC)),
                "processed_at": datetime.now(UTC),
                "raw_data": gpu_record
            }
            
            # Check if processed record already exists
            existing = normalizer.mongo_client.find_one(
                "gpu_pricing_processed",
                {
                    "gpu_id": gpu_id,
                    "platform": platform,
                    "processed_at": {
                        "$gte": datetime.now(UTC) - timedelta(hours=1)
                    }
                }
            )
            
            if existing:
                # Update existing record
                normalizer.mongo_client.update_one(
                    "gpu_pricing_processed",
                    {"_id": existing["_id"]},
                    {"$set": processed_record}
                )
                print(f"Updated: {original_name} -> {normalized_name}")
            else:
                # Insert new record
                normalizer.mongo_client.insert_one("gpu_pricing_processed", processed_record)
                print(f"Inserted: {original_name} -> {normalized_name}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {gpu_record.get('name', 'Unknown')}: {e}")
    
    print(f"Processed {processed_count} GPU records, skipped {skipped_count} non-GPU items")
    return processed_count

def update_hourly_pricing():
    """Update pricing data every hour"""
    print(f"Running hourly pricing update at {datetime.now()}")
    
    # Get the latest processed GPU data
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Find the most recent pricing data for each GPU (across all platforms)
    pipeline = [
        {
            "$sort": {"processed_at": -1}
        },
        {
            "$group": {
                "_id": {"gpu_id": "$gpu_id"},
                "latest_records": {"$push": "$$ROOT"},
                "platforms": {"$addToSet": "$platform"},
                "price_range": {
                    "$push": "$price"
                }
            }
        }
    ]
    
    latest_pricing = list(mongo_client.db["gpu_pricing_processed"].aggregate(pipeline))
    
    # Create hourly summary
    all_prices = []
    all_platforms = set()
    gpu_count_by_platform = {}
    
    for record in latest_pricing:
        platforms = record.get("platforms", [])
        prices = record.get("price_range", [])
        
        all_platforms.update(platforms)
        
        # Count GPUs by platform
        for platform in platforms:
            if platform not in gpu_count_by_platform:
                gpu_count_by_platform[platform] = 0
            gpu_count_by_platform[platform] += 1
        
        # Collect all valid prices
        for price in prices:
            if price is not None and price != "":
                try:
                    all_prices.append(float(price))
                except (ValueError, TypeError):
                    continue
    
    hourly_summary = {
        "timestamp": datetime.now(UTC),
        "total_unique_gpus": len(latest_pricing),
        "platforms": list(all_platforms),
        "price_range": {
            "min": min(all_prices) if all_prices else 0,
            "max": max(all_prices) if all_prices else 0,
        },
        "gpu_count_by_platform": gpu_count_by_platform
    }
    
    # Save hourly summary
    mongo_client.insert_one("gpu_pricing_hourly_summary", hourly_summary)
    
    print(f"Updated hourly summary: {hourly_summary['total_unique_gpus']} unique GPUs across {len(hourly_summary['platforms'])} platforms")
    print(f"Price range: ${hourly_summary['price_range']['min']:.2f} - ${hourly_summary['price_range']['max']:.2f}/hour")
    for platform, count in hourly_summary['gpu_count_by_platform'].items():
        print(f"  {platform}: {count} GPUs")

def run_full_scraping_job():
    """Main scraping job that runs every 12 hours"""
    print(f"Starting full GPU scraping job at {datetime.now()}")
    
    try:
        # Step 1: Scrape new data
        scrape_gpu_prices()
        
        # Step 2: Process and normalize the data
        process_and_normalize_gpu_data()
        
        # Step 3: Update hourly pricing
        update_hourly_pricing()
        
        print("Full GPU scraping job completed successfully!")
        
    except Exception as e:
        print(f"Error in full scraping job: {e}")

def check_scraper_status():
    """Check if the scraper is running and return status information"""
    try:
        mongo_client = Mongo(os.getenv("MONGODB_URL"))
        
        # Check for recent data
        recent_data = mongo_client.find_one(
            "gpu_pricing_processed",
            {"processed_at": {"$gte": datetime.now(UTC) - timedelta(hours=1)}}
        )
        
        # Check for recent hourly summaries
        recent_summary = mongo_client.find_one(
            "gpu_pricing_hourly_summary",
            {"timestamp": {"$gte": datetime.now(UTC) - timedelta(hours=1)}}
        )
        
        return {
            "has_recent_data": bool(recent_data),
            "has_recent_summary": bool(recent_summary),
            "last_data_time": recent_data.get("processed_at") if recent_data else None,
            "last_summary_time": recent_summary.get("timestamp") if recent_summary else None
        }
    except Exception as e:
        return {"error": str(e)}

def schedule_jobs():
    """Schedule the scraping jobs"""
    print("Setting up GPU scraping schedule...")
    
    # Run full scraping every 12 hours
    schedule.every(12).hours.do(run_full_scraping_job)
    
    # Run hourly updates
    schedule.every().hour.do(update_hourly_pricing)
    
    # Run immediately on startup
    print("Running initial scraping job...")
    try:
        run_full_scraping_job()
    except Exception as e:
        print(f"Error in initial scraping job: {e}")
    
    print("Scheduled jobs:")
    print("- Full scraping: Every 12 hours")
    print("- Hourly updates: Every hour")
    
    # Keep the script running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("GPU scraper stopped by user")
            break
        except Exception as e:
            print(f"Error in GPU scraper loop: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    schedule_jobs() 