from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]

def sessions():
    return _db["sessions"]

def pages():
    return _db["pages"]
