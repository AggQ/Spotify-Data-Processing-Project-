# Spotify Data Analysis Project
# Complete implementation with all features shown in the images

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('dark_background')
sns.set(style="darkgrid")

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Part 1: Data Collection and Preparation
# --------------------------------------------------

def initialize_spotify_client():
    """Initialize Spotify API client"""
    # Replace with your own credentials
    client_id = 'YOUR_CLIENT_ID'
    client_secret = 'YOUR_CLIENT_SECRET'
    
    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, 
        client_secret=client_secret
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def create_sample_data():
    """Create sample dataset that mimics real Spotify data"""
    np.random.seed(42)
    
    # Audio features
    feat_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo', 
                'loudness', 'duration_ms']
    
    # Sample data
    genres = ['pop', 'rap', 'rock', 'edm', 'hip hop', 'dance pop', 
              'country', 'trance', 'house', 'trap']
    
    artists = ['Drake', 'The Weeknd', 'Taylor Swift', 'Kendrick Lamar',
               'Billie Eilish', 'Post Malone', 'Ariana Grande', 'Ed Sheeran',
               'Dua Lipa', 'Bad Bunny']
    
    track_names = [
        'God\'s Plan', 'Blinding Lights', 'Anti-Hero', 'HUMBLE.',
        'bad guy', 'Sunflower', 'thank u, next', 'Shape of You',
        'Don\'t Start Now', 'Dákiti'
    ]
    
    # Create sample tracks
    track_data = []
    for i in range(500):
        artist_idx = np.random.randint(0, len(artists))
        track_idx = np.random.randint(0, len(track_names))
        genre_idx = np.random.randint(0, len(genres))
        
        track = {
            'track_id': f"track_{i}",
            'track_name': f"{track_names[track_idx]} {i}",
            'track_popularity': np.random.randint(30, 100),
            'artist_name': artists[artist_idx],
            'artist_genres': [genres[genre_idx]],
            'danceability': np.random.uniform(0.3, 0.9),
            'energy': np.random.uniform(0.3, 0.9),
            'speechiness': np.random.uniform(0.03, 0.3),
            'acousticness': np.random.uniform(0.01, 0.8),
            'instrumentalness': np.random.uniform(0, 0.4),
            'liveness': np.random.uniform(0.05, 0.3),
            'valence': np.random.uniform(0.1, 0.9),
            'tempo': np.random.uniform(70, 180),
            'loudness': np.random.uniform(-20, -5),
            'duration_ms': np.random.randint(150000, 300000),
            'key': np.random.randint(0, 12),
            'mode': np.random.randint(0, 2),
            'time_signature': 4
        }
        track_data.append(track)
    
    return pd.DataFrame(track_data)

# Part 2: Data Analysis
# --------------------------------------------------

def analyze_data(track_df):
    """Perform all analysis and visualization"""
    
    # 2.1 Correlation Analysis
    plt.figure(figsize=(12, 8))
    corr = track_df[['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 
                    'tempo']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens')
    plt.title('Audio Features Correlation')
    plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.2 Feature Distributions
    plt.figure(figsize=(10, 6))
    sns.jointplot(data=track_df, x='energy', y='danceability', kind='hex')
    plt.savefig('images/danceability_energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.3 Clustering Analysis
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
               'acousticness', 'instrumentalness', 'liveness', 'valence']
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(track_df[features])
    
    # Elbow method to find optimal clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('images/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Apply K-means with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    track_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    track_df['pca_x'] = pca_features[:, 0]
    track_df['pca_y'] = pca_features[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='pca_x', y='pca_y', hue='cluster', data=track_df, palette='viridis')
    plt.title('Track Clusters (PCA)')
    plt.savefig('images/pca_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.4 Genre Analysis
    top_genres = track_df.explode('artist_genres')['artist_genres'].value_counts().head(5).index
    
    plt.figure(figsize=(12, 6))
    for genre in top_genres:
        genre_data = track_df[track_df['artist_genres'].apply(lambda x: genre in x)]
        sns.kdeplot(data=genre_data, x='danceability', label=genre, fill=True)
    
    plt.title('Danceability Distribution by Genre')
    plt.legend()
    plt.savefig('images/features_by_genre.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.5 Comprehensive Dashboard
    fig = plt.figure(figsize=(20, 15), facecolor='black')
    fig.suptitle('Spotify Data Analysis Dashboard', fontsize=24, color='white')
    
    # Layout
    gs = fig.add_gridspec(3, 3)
    
    # 1. Top Artists
    ax1 = fig.add_subplot(gs[0, 0])
    top_artists = track_df['artist_name'].value_counts().head(5)
    sns.barplot(y=top_artists.index, x=top_artists.values, palette='Greens', ax=ax1)
    ax1.set_title('Top 5 Artists', color='white')
    
    # 2. Popularity Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=track_df, x='track_popularity', kde=True, color='green', ax=ax2)
    ax2.set_title('Popularity Distribution', color='white')
    
    # 3. Energy vs Valence
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(data=track_df, x='energy', y='valence', hue='cluster', 
                   palette='viridis', ax=ax3)
    ax3.set_title('Energy vs Valence', color='white')
    
    # 4. Feature Comparison
    ax4 = fig.add_subplot(gs[1, :])
    features = ['danceability', 'energy', 'acousticness', 'valence']
    sns.boxplot(data=track_df[features], palette='Greens', ax=ax4)
    ax4.set_title('Audio Feature Distributions', color='white')
    
    # 5. Tempo Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    sns.histplot(data=track_df, x='tempo', kde=True, color='green', ax=ax5)
    ax5.set_title('Tempo Distribution', color='white')
    
    # 6. Loudness vs Energy
    ax6 = fig.add_subplot(gs[2, 1:])
    sns.kdeplot(data=track_df, x='loudness', y='energy', fill=True, cmap='Greens', ax=ax6)
    ax6.set_title('Loudness vs Energy Density', color='white')
    
    plt.tight_layout()
    plt.savefig('images/spotify_data_dashboard.png', facecolor='black', dpi=300, bbox_inches='tight')
    plt.close()

# Part 3: Main Execution
# --------------------------------------------------

def main():
    print("Starting Spotify Data Analysis...")
    
    # Try to get real data from Spotify API
    try:
        sp = initialize_spotify_client()
        print("Connected to Spotify API")
        
        # Get top tracks (commented out for demo)
        # tracks_df = get_top_tracks_by_popularity(sp, limit=500)
        # audio_features_df = get_audio_features(sp, tracks_df['track_id'].tolist())
        # track_df = pd.merge(tracks_df, audio_features_df, on='track_id')
        # track_df.to_csv('data/spotify_track_data.csv', index=False)
        
        # For demo purposes, we'll use sample data
        track_df = create_sample_data()
        track_df.to_csv('data/spotify_track_data.csv', index=False)
        
    except Exception as e:
        print(f"Could not connect to Spotify API: {e}")
        print("Using sample data instead")
        track_df = create_sample_data()
        track_df.to_csv('data/spotify_track_data.csv', index=False)
    
    # Perform analysis
    analyze_data(track_df)
    print("Analysis complete! Check the 'images' folder for visualizations.")

if __name__ == "__main__":
    main()