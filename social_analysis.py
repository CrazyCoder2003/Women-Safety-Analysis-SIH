import pandas as pd
import folium
import re
from newsapi import NewsApiClient
import praw

# API Configurations
NEWSAPI_KEY = '58426cf83a414950baec3d2f7c91e4ed'
REDDIT_CLIENT_ID = 'bCo3gwYUquQuswp_PXzHPA'
REDDIT_CLIENT_SECRET = 'nvqvuRSJ_0WBgsMrkNWcEVVFucLjFw'


def generate_social_heatmap():
    # Load geographical data
    states_df = pd.read_csv('India States-UTs.csv')
    cities_df = pd.read_csv('IndianCities.csv', encoding='latin1')  # Handle encoding

    # Clean city data
    cities_df = cities_df.rename(columns={
        'Lat': 'Latitude',
        'Long': 'Longitude',
        'State': 'State'  # Explicitly keep State as is
    })[['City', 'State', 'Latitude', 'Longitude']].dropna()

    # Convert coordinates to numeric
    cities_df['Latitude'] = pd.to_numeric(cities_df['Latitude'], errors='coerce')
    cities_df['Longitude'] = pd.to_numeric(cities_df['Longitude'], errors='coerce')
    cities_df = cities_df.dropna()

    cities_df = cities_df[cities_df['State'].notna()]

    # Initialize APIs
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent='india-crime-map')

    # Get Indian news sources
    indian_sources = newsapi.get_sources(country='in', language='en')['sources']
    source_ids = [s['id'] for s in indian_sources]

    # Fetch news data
    news_articles = newsapi.get_everything(
        q='crime OR assault OR robbery',
        sources=','.join(source_ids),
        language='en',
        page_size=100
    )

    # Fetch Reddit data from Indian subreddits
    indian_subreddits = ['india', 'delhi', 'mumbai', 'bangalore', 'chennai',
                         'kolkata', 'hyderabad', 'pune', 'kerala', 'ahmedabad']
    reddit_posts = []
    for sub in indian_subreddits:
        subreddit = reddit.subreddit(sub)
        reddit_posts += subreddit.search('crime OR assault OR robbery', limit=50)

    # Location extraction setup
    state_names = states_df['State/UT'].str.lower().tolist()
    city_names = cities_df['City'].str.lower().tolist()
    state_city_map = cities_df.set_index('City')['State'].to_dict()

    def extract_location(text):
        text = str(text).lower()
        locations = []

        # Find cities
        for city in city_names:
            if re.search(r'\b' + re.escape(city) + r'\b', text):
                locations.append(('city', city.title()))

        # Find states
        for state in state_names:
            if re.search(r'\b' + re.escape(state) + r'\b', text):
                locations.append(('state', state.title()))

        return locations

    # Process data
    locations = []
    for article in news_articles['articles']:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        for loc_type, name in extract_location(text):
            locations.append({'name': name, 'type': loc_type})

    for post in reddit_posts:
        text = f"{post.title} {post.selftext}"
        for loc_type, name in extract_location(text):
            locations.append({'name': name, 'type': loc_type})

    # Create dataframe
    df = pd.DataFrame(locations)

    # Merge coordinates
    def get_coordinates(row):
        if row['type'] == 'city':
            city_data = cities_df[cities_df['City'].str.lower() == row['name'].lower()]
            if not city_data.empty:
                return city_data.iloc[0]['Latitude'], city_data.iloc[0]['Longitude']
        else:
            state_data = states_df[states_df['State/UT'].str.lower() == row['name'].lower()]
            if not state_data.empty:
                return state_data.iloc[0]['Latitude'], state_data.iloc[0]['Longitude']
        return (None, None)

    df[['Latitude', 'Longitude']] = df.apply(get_coordinates, axis=1, result_type='expand')
    df = df.dropna()

    # Calculate severity
    counts = df.groupby(['name', 'Latitude', 'Longitude']).size().reset_index(name='count')
    counts['Severity_Score'] = 10 * (counts['count'] - counts['count'].min()) / \
                               (counts['count'].max() - counts['count'].min())

    # Create map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB dark_matter')

    def get_color(score):
        return ['#2dc937', '#e7b416', '#dba11c', '#cc4a1e', '#a60707'][
            min(int(score // 2), 4)
        ]

    for _, row in counts.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8 + row['count'] * 0.8,
            color=get_color(row['Severity_Score']),
            fill=True,
            fill_opacity=0.7,
            popup=f"<strong>{row['name']}</strong><br>"
                  f"Mentions: {row['count']}<br>"
                  f"Severity: {row['Severity_Score']:.1f}/10",
            tooltip=row['name']
        ).add_to(m)

    return m._repr_html_()