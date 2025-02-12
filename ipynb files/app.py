import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv('/Users/crazeformarvel/Desktop/Recommendation/movies3.csv')
ratings = pd.read_csv('/Users/crazeformarvel/Desktop/Recommendation/ratings3.csv')
movie_ratings = pd.merge(movies, ratings, on='movieId')

# Preprocess data for Correlation Coefficient
movie_matrix = movie_ratings.pivot_table(index='title', columns='userId', values='rating')
movie_stats = movie_ratings.groupby('title').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['mean_rating', 'num_ratings']

# Preprocess data for Cosine Similarity
df_movies = movies[['movieId', 'title', 'genres']].copy()
df_movies['indexcol'] = df_movies.index
df_movies['genres'] = df_movies['genres'].apply(lambda x: x.lower().replace('|', ' '))
tfidf = TfidfVectorizer(stop_words='english')
count_matrix = tfidf.fit_transform(df_movies['genres'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Preprocess data for KNN
rating_matrix = movie_ratings.pivot_table(index='title', columns='userId', values='rating').fillna(0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(rating_matrix.values)

# Function for Correlation Coefficient recommendations
def get_correlation_recommendations(target_movie, top_n=10):
    if target_movie not in movie_matrix.index:
        return []
    target_movie_ratings = movie_matrix.loc[target_movie]
    similar_movies = movie_matrix.corrwith(target_movie_ratings).dropna().to_frame(name='correlation')
    similar_movies = similar_movies.join(movie_stats['num_ratings'])
    similar_movies = similar_movies[similar_movies['num_ratings'] > 10]
    return similar_movies.sort_values('correlation', ascending=False).head(top_n).index.tolist()

# Function for Cosine Similarity recommendations
def get_cosine_similar_movies(movie_title, top_n=10):
    if movie_title not in df_movies['title'].values:
        return []
    idx = df_movies[df_movies['title'] == movie_title]['indexcol'].values[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return df_movies['title'].iloc[[i[0] for i in sim_scores]].tolist()

# Function for KNN recommendations
def get_knn_recommendations(movie_title, top_n=10):
    if movie_title not in rating_matrix.index:
        return []
    query_index = rating_matrix.index.get_loc(movie_title)
    distances, indices = model_knn.kneighbors(rating_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=top_n+1)
    return rating_matrix.index[indices.flatten()[1:]].tolist()

# Combined recommendation system
def get_combined_recommendations(movie_title, top_n=10):
    recommendations = {}
    for movie in get_correlation_recommendations(movie_title, top_n):
        recommendations[movie] = recommendations.get(movie, 0) + 2
    for movie in get_cosine_similar_movies(movie_title, top_n):
        recommendations[movie] = recommendations.get(movie, 0) + 1
    for movie in get_knn_recommendations(movie_title, top_n):
        recommendations[movie] = recommendations.get(movie, 0) + 3
    return [movie for movie, _ in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)][:top_n]

# Streamlit UI
st.title("Movie Recommendation System")
user_input = st.text_input("Enter a movie title:")
if st.button("Get Recommendations"):
    if user_input:
        recommendations = get_combined_recommendations(user_input, top_n=10)
        if recommendations:
            st.subheader(f"Recommendations for '{user_input}':")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.error("Movie not found or insufficient data.")
