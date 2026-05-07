from google.cloud import storage
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sa-key.json"
client = storage.Client()
buckets = client.list_buckets()
print("Danh sách các bucket của bạn:")
for b in buckets:
    print(b.name)
