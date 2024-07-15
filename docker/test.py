import requests
from bs4 import BeautifulSoup

# URL of the website you want to scrape
url = 'https://www.reddit.com/r/iphone15/comments/170rmdc/should_i_buy_the_iphone_15_plus/'

# Fetch the HTML content
response = requests.get(url)
html_content = response.text

# Parse the HTML content using BeautifulSoup with html5lib parser
soup = BeautifulSoup(html_content, 'html5lib')

# Extract data from the soup object as needed
# For example, to get all the text from the webpage:
text = soup.get_text()

print(text)