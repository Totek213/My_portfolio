import nltk
nltk.download('punkt')
nltk.download('wordnet') # Added for WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer # Not used in current example, but good for future expansion
from sklearn.metrics.pairwise import linear_kernel # Not used in current example, but good for future expansion

import pandas as pd
import numpy as np
import re
import os

# --- Configuration ---
LANGUAGE = "english" # Changed to English
NUM_SENTENCES_SUMMARY = 3 # Number of sentences in the summary
NUM_TOPICS = 3 # Number of topics to detect
NUM_WORDS_TOPIC = 5 # Number of words per topic

# --- 1. Simulated Data Sources (Instead of actual Web Scraping/API) ---
articles_data = [
    {
        "id": "a001",
        "title": "The Future of Artificial Intelligence in Medicine",
        "text": """Artificial Intelligence (AI) is revolutionizing medicine, offering new tools for diagnosis, treatment, and research.
        Machine learning algorithms are used to analyze medical images, such as MRIs and CT scans,
        helping detect cancers and other diseases at an early stage. AI systems can also assist in drug discovery,
        significantly reducing the time needed to develop new therapies. The use of AI in personalized medicine
        allows tailoring treatment to individual patient characteristics, which increases therapeutic effectiveness and minimizes side effects.
        Challenges include ethical issues, data privacy, and the need for collaboration between doctors and AI engineers.
        The future looks promising but requires further research and regulation.
        """
    },
    {
        "id": "a002",
        "title": "Latest Trends in Blockchain Technology",
        "text": """Blockchain technology, primarily known for cryptocurrencies, is finding increasingly widespread application across various economic sectors.
        Beyond finance, blockchain is used in supply chain management to enhance transparency and product traceability.
        Smart contracts, self-executing agreements recorded on the blockchain, automate business processes and reduce the need for intermediaries.
        DeFi (Decentralized Finance) is another rapidly growing area, offering decentralized financial services without banks.
        Challenges include scalability, legal regulations, and user education. Blockchain has the potential to change the way we trust each other.
        """
    },
    {
        "id": "a003",
        "title": "Impact of Climate Change on Agriculture",
        "text": """Climate change poses a serious challenge to global agriculture. Rising temperatures,
        extreme weather events such as droughts and floods, and changes in rainfall patterns affect crop yields and food security.
        Farmers must adapt to new conditions by introducing innovative cultivation techniques, such as precision agriculture,
        which uses data to optimize resources. The development of drought-resistant crop varieties and more efficient irrigation systems
        are crucial for ensuring the stability of food production. Governments and international organizations are working on policies supporting
        sustainable agriculture and reducing greenhouse gas emissions in the agricultural sector.
        """
    },
    {
        "id": "a004",
        "title": "New Discoveries in Robotics",
        "text": """Robotics is developing rapidly, and new discoveries are paving the way for more advanced and autonomous systems.
        Increasingly sophisticated machine learning algorithms enable robots to learn complex tasks and adapt to changing environments.
        The introduction of collaborative robots (cobots) into industry increases production efficiency and worker safety.
        Mobile robots are used in logistics, space exploration, and healthcare. The development of artificial muscles and new materials
        allows for the creation of more flexible and agile robots. Ethical aspects of integrating robots into society
        are becoming increasingly important as they become more widespread.
        """
    },
    {
        "id": "a005",
        "title": "The Evolution of Cybersecurity in the Age of AI",
        "text": """With the advancement of artificial intelligence, the cybersecurity landscape is also changing.
        AI is used by defenders to detect complex threats and automate incident response,
        as well as by attackers to create more sophisticated attacks, such as AI-generated phishing or automated vulnerability scanning.
        Companies are investing in AI-powered security systems that can analyze vast amounts of data in real time,
        identifying anomalies and patterns indicative of an attack. Continuous employee training and software updates
        are crucial to meet new challenges. Digital hygiene education is key to data protection.
        """
    }
]

# --- 2. Text Processing Module (NLP) ---

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
# English stop words from Gensim
english_stopwords = set(STOPWORDS)


def preprocess_text(text):
    """
    Cleans and processes text: tokenization, lowercase, remove numbers/punctuation, lemmatization, remove stop words.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Remove digits
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = word_tokenize(text, language=LANGUAGE)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in english_stopwords and len(word) > 2]
    return tokens

# --- 3. Machine Learning Module (ML) ---

class NewsSenseML:
    def __init__(self):
        self.summarizer = LsaSummarizer(Stemmer(LANGUAGE))
        self.summarizer.stop_words = get_stop_words(LANGUAGE)
        self.lda_model = None
        self.dictionary = None
        # self.tfidf_vectorizer = None # Not used in this basic example
        self.article_df = pd.DataFrame(columns=['id', 'title', 'text', 'summary', 'processed_tokens', 'topics'])

    def add_articles(self, articles):
        """Adds and processes new articles."""
        new_articles_list = []
        for article in articles:
            article_id = article['id']
            title = article['title']
            text = article['text']

            # Summarization
            parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
            summary_sentences = self.summarizer(parser.document, NUM_SENTENCES_SUMMARY)
            summary = " ".join([str(sentence) for sentence in summary_sentences])

            # Text processing for LDA and recommendations
            processed_tokens = preprocess_text(text)

            new_articles_list.append({
                'id': article_id,
                'title': title,
                'text': text,
                'summary': summary,
                'processed_tokens': processed_tokens,
                'topics': [] # Leave empty, LDA will populate
            })

        # Create DataFrame from new articles and concatenate with existing
        new_df = pd.DataFrame(new_articles_list)
        self.article_df = pd.concat([self.article_df, new_df], ignore_index=True)
        self.article_df.drop_duplicates(subset=['id'], inplace=True) # Prevent duplicates

    def train_topic_model(self):
        """Trains the LDA model for topic detection."""
        if self.article_df.empty:
            print("No articles to train the topic model.")
            return

        texts = self.article_df['processed_tokens'].tolist()
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        # Train LDA
        # Use LdaMulticore for better performance
        num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        self.lda_model = LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=self.dictionary, passes=10, workers=num_workers)

        # Assign topics to articles
        for index, row in self.article_df.iterrows():
            doc_bow = self.dictionary.doc2bow(row['processed_tokens'])
            topics = self.lda_model.get_document_topics(doc_bow, minimum_probability=0.01) # Minimum_probability filters less relevant topics
            # Sort topics by importance and take the most important ones
            sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
            self.article_df.at[index, 'topics'] = sorted_topics

        print(f"\nGenerated Topics (top {NUM_WORDS_TOPIC} words):")
        for idx, topic in self.lda_model.print_topics(num_words=NUM_WORDS_TOPIC):
            print(f"Topic {idx}: {topic}")

    def get_article_topics(self, article_id):
        """Returns topics for a specific article."""
        article = self.article_df[self.article_df['id'] == article_id]
        if not article.empty:
            return article['topics'].iloc[0]
        return []

    def get_topics_for_display(self, article_id):
        """Returns topics in a readable format for the user."""
        topics = self.get_article_topics(article_id)
        if not topics:
            return "No topics assigned."

        display_topics = []
        for topic_id, prob in topics:
            if self.lda_model and topic_id < self.lda_model.num_topics:
                topic_words = [word.split('*')[1].strip().replace('"', '') for word in self.lda_model.print_topic(topic_id, NUM_WORDS_TOPIC).split('+')]
                display_topics.append(f"Topic {topic_id} (weight: {prob:.2f}): {', '.join(topic_words)}")
            else:
                display_topics.append(f"Topic {topic_id} (weight: {prob:.2f})")
        return "\n".join(display_topics)


    def recommend_articles(self, user_preferred_topic_id=None, num_recommendations=3):
        """
        Simplified recommendation based on preferred topic or top articles.
        A real system would have a complex collaborative/content-based model here.
        """
        if self.article_df.empty or self.lda_model is None:
            print("No articles or LDA model has not been trained.")
            return pd.DataFrame()

        if user_preferred_topic_id is not None:
            # Recommend articles from the selected topic
            recommended_articles = []
            for index, row in self.article_df.iterrows():
                for topic_id, prob in row['topics']:
                    if topic_id == user_preferred_topic_id:
                        recommended_articles.append(row)
                        break # Only need to find one match
            
            recommended_df = pd.DataFrame(recommended_articles)
            # This is very simplified; a real system would have a more intelligent scoring mechanism
            if not recommended_df.empty:
                return recommended_df.head(num_recommendations)
            else:
                print(f"No articles found for topic {user_preferred_topic_id}. Recommending top general articles.")
                # If no articles, fall back to general top ones
                return self.article_df.sort_values(by='id', ascending=True).head(num_recommendations)
        else:
            # By default, recommend the latest/most popular (here - just top 3)
            return self.article_df.sort_values(by='id', ascending=True).head(num_recommendations)


# --- 4. Main Bot Logic ---

def run_news_sense_ai():
    print("Welcome to NewsSense AI - Your Smart News Curator!")
    ns_ml = NewsSenseML()
    ns_ml.add_articles(articles_data)
    ns_ml.train_topic_model()

    while True:
        print("\n--- Menu ---")
        print("1. Display all articles and summaries")
        print("2. Display topics for a specific article")
        print("3. Get recommendations (provide a preferred topic or 'none')")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("\n--- All Articles and Summaries ---")
            for index, row in ns_ml.article_df.iterrows():
                print(f"ID: {row['id']}")
                print(f"Title: {row['title']}")
                print(f"Summary: {row['summary']}\n")
        
        elif choice == '2':
            article_id = input("Enter article ID (e.g., a001): ")
            if article_id in ns_ml.article_df['id'].values:
                print(f"\n--- Topics for Article ID: {article_id} ---")
                print(ns_ml.get_topics_for_display(article_id))
            else:
                print("Article with the given ID not found.")

        elif choice == '3':
            print("\n--- Recommendations ---")
            print("Available topics (ID):")
            if ns_ml.lda_model:
                for idx, topic in ns_ml.lda_model.print_topics(num_words=NUM_WORDS_TOPIC):
                    print(f"  {idx}: {', '.join([word.split('*')[1].strip().replace('"', '') for word in topic.split('+')])}")
            
            topic_input = input("Enter preferred topic number (e.g., 0, 1, 2) or type 'none' for general recommendations: ")
            
            user_preferred_topic = None
            try:
                if topic_input.lower() != 'none':
                    user_preferred_topic = int(topic_input)
                    if user_preferred_topic not in range(NUM_TOPICS):
                        print("Invalid topic number. Please enter a number from the list.")
                        user_preferred_topic = None # Reset to avoid using invalid topic
            except ValueError:
                print("Invalid format. Please enter a number or 'none'.")

            recommended = ns_ml.recommend_articles(user_preferred_topic)
            if not recommended.empty:
                print("\nRecommended articles for you:")
                for index, row in recommended.iterrows():
                    print(f"ID: {row['id']}")
                    print(f"Title: {row['title']}")
                    print(f"Summary: {row['summary']}\n")
            else:
                print("No recommendations found.")

        elif choice == '4':
            print("Thank you for using NewsSense AI. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the bot
if __name__ == "__main__":
    run_news_sense_ai()