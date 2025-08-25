import logging

from pymongo import MongoClient

logger = logging.getLogger(__name__)


class Mongo:
    def __init__(self, url):
        self.client = MongoClient(
            url,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
            w="majority",
            readPreference="primary",
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
        )
        self.db = self.client["dria_benchmark_test"]

        # Test connection on initialization
        try:
            self.client.admin.command("ping")
            logger.info("MongoDB connection established successfully with pooling")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def insert_one(self, collection_name: str, data: dict):
        collection = self.get_collection(collection_name)
        collection.insert_one(data)

    def find_one(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.find_one(query)

    def find_many(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.find(query)

    def update_one(self, collection_name: str, query: dict, data: dict):
        collection = self.get_collection(collection_name)
        collection.update_one(query, data)

    def close(self):
        self.client.close()

    def get_metric(self, obj, attr_path: str, default=None):
        """
        Get a nested attribute from an object, returning default if any part of the path doesn't exist for saving metrics to mongo
        """
        try:
            attrs = attr_path.split(".")
            current = obj
            for attr in attrs:
                # Handle array access like "streams[0]"
                if "[" in attr and "]" in attr:
                    attr_name = attr.split("[")[0]
                    index = int(attr.split("[")[1].split("]")[0])
                    current = getattr(current, attr_name)[index]
                else:
                    current = getattr(current, attr)
            return current
        except (AttributeError, TypeError, IndexError, ValueError):
            return default
