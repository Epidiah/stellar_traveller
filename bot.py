from requests_oauthlib import OAuth1
from space_vista import *
import json
import os
import requests
import sys
import time
# from dotenv import load_dotenv

# load_dotenv()

MEDIA_ENDPOINT_URL = "https://upload.twitter.com/1.1/media/upload.json"
METADATA_ENDPOINT_URL = "https://upload.twitter.com/1.1/media/metadata/create.json"
POST_TWEET_URL = "https://api.twitter.com/1.1/statuses/update.json"

CONSUMER_KEY = os.getenv("C_KEY")
CONSUMER_SECRET = os.getenv("C_SECRET")
ACCESS_TOKEN = os.getenv("VS_A_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("VS_A_TOKEN_SECRET")
# ACCESS_TOKEN = os.getenv("A_TOKEN")
# ACCESS_TOKEN_SECRET = os.getenv("A_TOKEN_SECRET")

GIF_FILENAME = "starry.gif"


oauth = OAuth1(
    CONSUMER_KEY,
    client_secret=CONSUMER_SECRET,
    resource_owner_key=ACCESS_TOKEN,
    resource_owner_secret=ACCESS_TOKEN_SECRET,
)


class GifTweet(object):
    def __init__(self, file_name):
        """
        Defines gif tweet properties
        """
        self.gif_file_name = file_name
        self.total_bytes = os.path.getsize(self.gif_file_name)
        self.media_id = None
        self.processing_info = None

    def upload_init(self):
        """
        Initializes Upload
        """
        print("INIT")

        request_data = {
            "command": "INIT",
            "media_type": "image/gif",
            "total_bytes": self.total_bytes,
            "media_category": "tweet_gif",
        }

        req = requests.post(url=MEDIA_ENDPOINT_URL, data=request_data, auth=oauth)
        media_id = req.json()["media_id"]

        self.media_id = media_id

        print("Media ID: %s" % str(media_id))

    def upload_append(self):
        """
        Uploads media in chunks and appends to chunks uploaded
        """
        segment_id = 0
        bytes_sent = 0
        file = open(self.gif_file_name, "rb")

        while bytes_sent < self.total_bytes:
            chunk = file.read(4 * 1024 * 1024)

            print("APPEND")

            request_data = {
                "command": "APPEND",
                "media_id": self.media_id,
                "segment_index": segment_id,
            }

            files = {"media": chunk}

            req = requests.post(
                url=MEDIA_ENDPOINT_URL, data=request_data, files=files, auth=oauth
            )

            if req.status_code < 200 or req.status_code > 299:
                print(req.status_code)
                print(req.text)
                sys.exit(0)

            segment_id = segment_id + 1
            bytes_sent = file.tell()

            print("%s of %s bytes uploaded" % (str(bytes_sent), str(self.total_bytes)))

        print("Upload chunks complete.")

    def upload_finalize(self):
        """
        Finalizes uploads and starts processing
        """
        print("FINALIZE")

        request_data = {"command": "FINALIZE", "media_id": self.media_id}

        req = requests.post(url=MEDIA_ENDPOINT_URL, data=request_data, auth=oauth)
        print(req.json())

        self.processing_info = req.json().get("processing_info", None)
        self.check_status()

    def check_status(self):
        """
        Checks processing status
        """
        if self.processing_info is None:
            return

        state = self.processing_info["state"]

        print("Media processing status is %s " % state)

        if state == "succeeded":
            return

        if state == "failed":
            err = state["error"]
            print(f"Error Code: {err['code']}\nError: {err['name']} - {err['message']}")
            sys.exit(0)

        check_after_secs = self.processing_info["check_after_secs"]

        print("Checking after %s seconds" % str(check_after_secs))
        time.sleep(check_after_secs)

        print("STATUS")

        request_params = {"command": "STATUS", "media_id": self.media_id}

        req = requests.get(url=MEDIA_ENDPOINT_URL, params=request_params, auth=oauth)

        self.processing_info = req.json().get("processing_info", None)
        self.check_status()

    def set_metadata(self):
        request_data = {
            "media_id": self.media_id,
            "alt_text": "Veiw of space from observation windows aboard a starship as some planets gently roll by.",
        }
        req = requests.post(url=METADATA_ENDPOINT_URL, data=request_data, auth=oauth)

    def tweet(self, status):
        """
        Publishes Tweet with attached gif
        """
        request_data = {
            "status": status,
            "media_ids": self.media_id,
        }

        req = requests.post(url=POST_TWEET_URL, data=request_data, auth=oauth)
        print(req.json())


if __name__ == "__main__":
    status_dict = random_spacescape()
    minute = 60
    hour = 60 ** 2
    # time.sleep(67*minute)
    # print("\a")
    spacescape = GifTweet(GIF_FILENAME)
    spacescape.upload_init()
    spacescape.upload_append()
    spacescape.upload_finalize()
    spacescape.set_metadata()
    status = '\n'.join([v for v in status_dict.values()])
    spacescape.tweet(status)
