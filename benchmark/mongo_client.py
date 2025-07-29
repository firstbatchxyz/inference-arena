from pymongo import MongoClient, InsertOne, UpdateOne, DeleteOne
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Mongo:
    def __init__(self, url):
        # Configure MongoDB client with connection pooling and optimization
        self.client = MongoClient(
            url,
            maxPoolSize=50,        # Maximum connections in pool
            minPoolSize=10,        # Minimum connections to maintain
            maxIdleTimeMS=30000,   # Close idle connections after 30s
            waitQueueTimeoutMS=5000,  # Wait 5 seconds for connection from pool
            serverSelectionTimeoutMS=5000,  # Wait 5 seconds for server selection
            retryWrites=True,      # Enable automatic retry for writes
            w='majority',          # Ensure write acknowledgment from majority
            readPreference='primary',  # Read from primary for consistency
            connectTimeoutMS=10000,  # Connection timeout
            socketTimeoutMS=20000,   # Socket timeout
        )
        self.db = self.client["dria_benchmark"]
        
        # Test connection on initialization
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB connection established successfully with pooling")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def get_collection(self, collection_name: str):
        return self.db[collection_name]
    
    def insert_one(self, collection_name: str, data: dict):
        collection = self.get_collection(collection_name)
        collection.insert_one(data)

    def insert_many(self, collection_name: str, data: list[dict]):
        collection = self.get_collection(collection_name)
        collection.insert_many(data)

    def find_one(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.find_one(query)

    def find_many(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.find(query)
    
    def count_documents(self, collection_name: str, query: dict = None):
        """Count documents in a collection that match the query"""
        collection = self.get_collection(collection_name)
        if query is None:
            query = {}
        return collection.count_documents(query)
    
    def update_one(self, collection_name: str, query: dict, data: dict):
        collection = self.get_collection(collection_name)
        collection.update_one(query, data)

    def update_many(self, collection_name: str, query: dict, data: dict):
        collection = self.get_collection(collection_name)
        collection.update_many(query, data)

    def delete_one(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        collection.delete_one(query)

    def delete_many(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        collection.delete_many(query)

    def close(self):
        self.client.close()
    
    def bulk_insert(self, collection_name: str, data: List[Dict[str, Any]], ordered: bool = False):
        """
        Bulk insert multiple documents for better performance
        
        Args:
            collection_name: Name of the collection
            data: List of documents to insert
            ordered: Whether to execute operations in order (default: False for better performance)
        """
        if not data:
            return
        
        collection = self.get_collection(collection_name)
        operations = [InsertOne(doc) for doc in data]
        
        try:
            result = collection.bulk_write(operations, ordered=ordered)
            logger.info(f"Bulk inserted {result.inserted_count} documents into {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Bulk insert failed for {collection_name}: {e}")
            raise
    
    def bulk_update(self, collection_name: str, updates: List[Dict[str, Any]], ordered: bool = False):
        """
        Bulk update multiple documents
        
        Args:
            collection_name: Name of the collection
            updates: List of update operations, each containing 'filter' and 'update' keys
            ordered: Whether to execute operations in order
        """
        if not updates:
            return
        
        collection = self.get_collection(collection_name)
        operations = [
            UpdateOne(update['filter'], update['update'], upsert=update.get('upsert', False))
            for update in updates
        ]
        
        try:
            result = collection.bulk_write(operations, ordered=ordered)
            logger.info(f"Bulk updated {result.modified_count} documents in {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Bulk update failed for {collection_name}: {e}")
            raise

    def _distribution_summary_to_dict(self, dist_summary):
        """
        Convert a DistributionSummary object to a dictionary that MongoDB can serialize.
        
        Args:
            dist_summary: DistributionSummary object from guidellm
            
        Returns:
            Dictionary representation of the distribution summary
        """
        if dist_summary is None:
            return None
            
        # Check if it's actually a DistributionSummary by checking if it has the expected attributes
        if not hasattr(dist_summary, 'mean'):
            return dist_summary
            
        result = {
            "mean": dist_summary.mean,
            "median": dist_summary.median,
            "mode": dist_summary.mode,
            "variance": dist_summary.variance,
            "std_dev": dist_summary.std_dev,
            "min": dist_summary.min,
            "max": dist_summary.max,
            "count": dist_summary.count,
            "total_sum": dist_summary.total_sum,
        }
        
        # Handle percentiles if they exist
        if hasattr(dist_summary, 'percentiles') and dist_summary.percentiles:
            result["percentiles"] = {
                "p001": dist_summary.percentiles.p001,
                "p01": dist_summary.percentiles.p01,
                "p05": dist_summary.percentiles.p05,
                "p10": dist_summary.percentiles.p10,
                "p25": dist_summary.percentiles.p25,
                "p75": dist_summary.percentiles.p75,
                "p90": dist_summary.percentiles.p90,
                "p95": dist_summary.percentiles.p95,
                "p99": dist_summary.percentiles.p99,
                "p999": dist_summary.percentiles.p999,
            }
        
        return result

    def safe_get_metric(self, obj, attr_path: str, default=None):
        """
        Safely get a nested attribute from an object, returning default if any part of the path doesn't exist.
        
        Args:
            obj: The object to extract from
            attr_path: Dot-separated attribute path (e.g., "metrics.requests_per_second.total" or "streams[0]")
            default: Default value to return if attribute doesn't exist
            
        Returns:
            The attribute value or default if not found
        """
        try:
            attrs = attr_path.split('.')
            current = obj
            for attr in attrs:
                # Handle array access like "streams[0]"
                if '[' in attr and ']' in attr:
                    attr_name = attr.split('[')[0]
                    index = int(attr.split('[')[1].split(']')[0])
                    current = getattr(current, attr_name)[index]
                else:
                    current = getattr(current, attr)
            return current
        except (AttributeError, TypeError, IndexError, ValueError):
            return default

    def safe_get_metric_dict(self, obj, metric_name: str, default=None):
        """
        Safely get a metric dictionary with total, successful, and errored values.
        
        Args:
            obj: The object containing metrics
            metric_name: The name of the metric (e.g., "requests_per_second")
            default: Default value to return if metric doesn't exist
            
        Returns:
            Dictionary with total, successful, and errored values, or default if not found
        """
        try:
            metric_obj = getattr(obj, metric_name)
            
            # Get the attributes and convert DistributionSummary objects to dicts
            total = getattr(metric_obj, "total", default)
            successful = getattr(metric_obj, "successful", default)
            errored = getattr(metric_obj, "errored", default)
            
            return {
                "total": self._distribution_summary_to_dict(total) if total is not None else default,
                "successful": self._distribution_summary_to_dict(successful) if successful is not None else default,
                "errored": self._distribution_summary_to_dict(errored) if errored is not None else default
            }
        except (AttributeError, TypeError):
            return {
                "total": default,
                "successful": default,
                "errored": default
            }