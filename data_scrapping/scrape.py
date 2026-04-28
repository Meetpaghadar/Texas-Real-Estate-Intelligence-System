import requests
from bs4 import BeautifulSoup
import pandas as pd

# 1. Define the target URL
url = "http://quotes.toscrape.com/"

print(f"Fetching data from {https://www.zillow.com/houston-tx/}...\n")

# 2. Send a GET request to the website
response = requests.get(url)

# Check if the request was successful (Status code 200)
if response.status_code == 200:
    # 3. Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 4. Find all the "quote" blocks on the page
    # Inspecting the website shows each quote is in a <div class="quote">
    quote_blocks = soup.find_all('div', class_='quote')
    
    # Create an empty list to store our extracted data
    data = []
    
    # 5. Loop through each block and extract the text and author
    for block in quote_blocks:
        text = block.find('span', class_='text').text
        author = block.find('small', class_='author').text
        
        # Append as a dictionary to our list
        data.append({
            'Author': author,
            'Quote': text
        })
        
    # 6. Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # 7. Display the structured data
    print("Scraping Successful! Here is the data:\n")
    print(df)
    
    # Optional: Save it to a CSV file for your records
    df.to_csv('scraped_quotes.csv', index=False)
    print("\nData saved to 'scraped_quotes.csv'")

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")