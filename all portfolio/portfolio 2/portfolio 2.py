import requests
from bs4 import BeautifulSoup
import webbrowser

# Function to fetch news headlines
def get_news():
    url = "https://www.bbc.com/news"  # BBC News website
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        headlines = []
        for item in soup.find_all("a", class_="gs-c-promo-heading", limit=5):  # Get top 5 headlines
            title = item.get_text(strip=True)
            link = "https://www.bbc.com" + item["href"]
            headlines.append((title, link))
        
        return headlines
    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {e}"

# Function to display news and open links
def news_bot():
    print("Fetching latest news...")
    news = get_news()
    
    if isinstance(news, str):
        print(news)
    else:
        for i, (title, link) in enumerate(news):
            print(f"{i+1}. {title}")
        
        choice = input("\nEnter the number of the article to open (or press Enter to exit): ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(news):
                webbrowser.open(news[choice-1][1])
                print("Opening article in browser...")
            else:
                print("Invalid choice.")

# Run the bot
if __name__ == "__main__":
    news_bot()
