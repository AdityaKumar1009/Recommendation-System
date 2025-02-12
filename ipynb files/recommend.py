import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv('/Users/crazeformarvel/Desktop/Recommendation/movies3.csv')  # Movie information, including genres
ratings = pd.read_csv('/Users/crazeformarvel/Desktop/Recommendation/ratings3.csv')  # Ratings by users
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
rating_matrix = movie_ratings.pivot_table(index='title', columns='userId', values='rating')
rating_matrix = rating_matrix.fillna(0)
rating_matrix_new = rating_matrix.values
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(rating_matrix_new)

# Function for Correlation Coefficient recommendations
def get_correlation_recommendations(target_movie, top_n=10):
    if target_movie not in movie_matrix.index:
        return []
    target_movie_ratings = movie_matrix.loc[target_movie]
    similar_movies = movie_matrix.corrwith(target_movie_ratings)
    similar_movies = similar_movies.dropna().to_frame(name='correlation')
    similar_movies = similar_movies.join(movie_stats['num_ratings'])
    similar_movies = similar_movies[similar_movies['num_ratings'] > 10]
    similar_movies = similar_movies.sort_values('correlation', ascending=False).head(top_n)
    return similar_movies.index.tolist()

# Function for Cosine Similarity recommendations
def get_cosine_similar_movies(movie_title, top_n=10):
    if movie_title not in df_movies['title'].values:
        return []
    idx = df_movies[df_movies['title'] == movie_title]['indexcol'].values[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['title'].iloc[movie_indices].tolist()

# Function for KNN recommendations
def get_knn_recommendations(movie_title, top_n=10):
    if movie_title not in rating_matrix.index:
        return []
    query_index = rating_matrix.index.get_loc(movie_title)
    distances, indices = model_knn.kneighbors(
        rating_matrix.iloc[query_index, :].values.reshape(1, -1),
        n_neighbors=top_n + 1
    )
    recommendations = rating_matrix.index[indices.flatten()[1:]].tolist()
    return recommendations

# Combined recommendation system
def get_combined_recommendations(movie_title, top_n=10):
    corr_recommendations = get_correlation_recommendations(movie_title, top_n)
    cosine_recommendations = get_cosine_similar_movies(movie_title, top_n)
    knn_recommendations = get_knn_recommendations(movie_title, top_n)

    # Combine recommendations with a weighted score
    recommendations = {}
    for movie in corr_recommendations:
        recommendations[movie] = recommendations.get(movie, 0) + 2  # Higher weight for correlation
    for movie in cosine_recommendations:
        recommendations[movie] = recommendations.get(movie, 0) + 1  # Medium weight for cosine similarity
    for movie in knn_recommendations:
        recommendations[movie] = recommendations.get(movie, 0) + 3  # Lower weight for KNN

    # Sort movies by combined scores
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    final_recommendations = [movie for movie, score in sorted_recommendations][:top_n]

    return final_recommendations

# Example usage
target_movie = "The Batman (2022)"
recommendations = get_combined_recommendations(target_movie, top_n=10)
print(f"Recommendations for '{target_movie}':")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")
