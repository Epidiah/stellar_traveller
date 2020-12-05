import os
import tweepy
from dotenv import load_dotenv

load_dotenv()
C_KEY = os.getenv('C_KEY')
C_SECRET = os.getenv('C_SECRET')
A_TOKEN = os.getenv('VS_A_TOKEN')
A_TOKEN_SECRET = os.getenv('VS_A_TOKEN_SECRET')

auth = tweepy.OAuthHandler(C_KEY, C_SECRET)
auth.set_access_token(A_TOKEN, A_TOKEN_SECRET)
api = tweepy.API(auth)


starry = api.media_upload('../starry-000001.gif')
# Something here like...
# api.create_media_metadata(starry.media_id, "A spacescape as seen from a starship in gentle orbit.")
api.update_status('Engaging thrusters...', media_ids=[starry.media_id])
