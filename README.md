# Spotify-Data-Processing-Project-
This project creates a Spotify data analysis project based on the images you've shared. It analyses Spotify track data to discover patterns in audio features, popularity, and genres.

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
The script generates several visualizations:
1. ```correlation_heatmap.png``` - Correlation between audio features
2. ```danceability_energy_distribution.png``` - Relationship between danceability and energy
3. ```elbow_method.png``` - Optimal number of clusters for K-means
4. ```pca_clustering.png``` - Track clusters visualized with PCA
5. ```features_by_genre.png``` - Feature distributions by genre
6. ```spotify_data_dashboard.png``` - Comprehensive dashboard of key metrics
