import praw
from psaw import PushshiftAPI
import pandas as pd
import re

# Uses the PRAW api to authenticate Reddit user, then uses
# the PSAW api to fetch the latest comments from a subreddit
# and saves those comments to the 'data.txt' file

file = open('src\data.txt', 'wb')

client_id = ''
secret_key = ''

r = praw.Reddit(client_id=client_id, client_secret=secret_key, username='', password='', user_agent='test')
api = PushshiftAPI(r)

#fetch latest comments from r/sweden
gen = api.search_comments(subreddit='sweden', limit=1000)

for comment in gen:
    if comment.body == '[removed]' or comment.body == '[deleted]':
        pass
    else:
        #remove all links from the comment and add it to file
        comment_string = re.sub(r'http\S+', '', comment.body)
        file.write(comment_string.encode(encoding='utf-8') + '\n'.encode(encoding='utf-8'))

file.close()

print('Done fetching comments')