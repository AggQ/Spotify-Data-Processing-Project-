# Spotify-Data-Processing-Project-
This project creates a Spotify data analysis project which analyses Spotify track data to discover patterns in audio features, popularity, and genres. Note that results will be saved in images folder. This is basically for _audiophiles_.

• _audio·phile_: a person with love for, affinity towards or obsession with high-quality playback of sound and music.

## Prerequisites

1. Python 3.8+
2. Spotify Account 
3. Required Python packages:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - spotipy
   - scikit-learn

1. Connects to the Spotify API using the Spotipy library
2. Extracts track data with audio features
3. Analyzes the data with various metrics
4. Creates visualisations similar to those in your images

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/spotify-analysis-project.git
   cd spotify-analysis-project

2. Install required packages:
    ```
    pip install pandas numpy matplotlib seaborn
    ```
    ```
    pip install spotipy
    ```
    ```
    pip install scikit-learn
    ```
    ```
    pip install python-dotenv
    ```

3. Run the script:
```python spotify_analysis.py```
    The script will generate all visualisations.

## Output files
Data visualizations:
1. ```correlation_heatmap.png``` - Correlation between audio features
    Louder tracks (Strong Positive ~0.8): Louder tracks tend to be more energetic
    Acoustic tracks (Strong Negative ~-0.7): Acoustic tracks typically have lower energy
    Danceable tracks (Moderate Positive ~0.5): Happier songs (higher valence) are often more danceable
2. ```danceability_energy_distribution.png``` - Relationship between danceability and energy
    Identifies tracks that balance both qualities suitable for parties.

3. ```elbow_method.png``` - Optimal number of clusters for K-means
        The "elbow" at 4-5 clusters suggests this is the ideal groupings. Adding more clusters will not improve the model by much.

4. ```pca_clustering.png``` - Track clusters visualized with PCA
    Each color represents a cluster of similar tracks.
    Spread along axes shows how distinct the groups are.
    Overlapping clusters have similar audio characteristics (they sound the same and work well together in playlists).

5. ```features_by_genre.png``` - Feature distributions by genre
    This shows how genres differ:
    EDM/Dance genres peak in danceability (right side)
    Rock shows wider variation in energy values
    Acoustic genres score high in acousticness (naturally)
    You can predict a song's genre accurately just from its audio features.

6. ```spotify_data_dashboard.png``` - Comprehensive dashboard of key metrics
    Top Artists/Genres: Shows which artists/genres you like the most
    Popularity Distribution: Most tracks cluster around 50-80 (out of 100)
    Energy-Valence Relationship: Confirms happy songs are more energetic
    Tempo Analysis: Most songs fall between 100-140 BPM (pop/rock music range)
    Loudness-Energy: The densest area shows popular tracks are loud and energetic
