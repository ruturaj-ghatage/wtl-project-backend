import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

class HybridRecommender:
    def __init__(self, content_based_weight=0.5, collaborative_weight=0.5, k=10):
        self.content_based_weight = content_based_weight
        self.collaborative_weight = collaborative_weight
        self.k = k

        # Load the dataset
        self.df = pd.read_csv('https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv')

        self.df = self.df.drop(['index', 'budget', 'homepage', 'keywords', 'original_language', 'original_title', 'popularity', 'production_companies',
                           'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'vote_average', 'vote_count'], axis=1)

        # Preprocess the data for content-based filtering
        self.df['overview'] = self.df['overview'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df['overview'])

        # Preprocess the data for collaborative filtering
        self.df['genres'] = self.df['genres'].fillna('')
        self.df['genres_list'] = self.df['genres'].apply(lambda x: [s.strip() for s in x.split()])
        genre_labels = sorted(set([genre for genres in self.df['genres_list'] for genre in genres]))
        genre_vectors = []
        for genres in self.df['genres_list']:
            genre_vector = [1 if genre in genres else 0 for genre in genre_labels]
            genre_vectors.append(genre_vector)
        self.cf_data = pd.DataFrame(genre_vectors, columns=genre_labels)

        # Build the collaborative filtering model
        self.cf_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.cf_model.fit(self.cf_data)

    def recommend(self, movie_id):
        # Get the index of the movie in the dataset
        idx = self.df[self.df['id'] == movie_id].index[0]

        # Get the content-based recommendations
        content_scores = list(enumerate(linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()))
        content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:self.k+1]
        content_indices = [i[0] for i in content_scores]

        # Get the collaborative filtering recommendations
        movie_data = self.cf_data.iloc[idx].values.reshape(1, -1)
        distances, indices = self.cf_model.kneighbors(movie_data, n_neighbors=self.k)
        cf_indices = indices.flatten()[1:]

        # Combine the recommendations
        indices = set(content_indices).union(set(cf_indices))
        scores = {}
        for i in indices:
            if i in content_indices:
                scores[i] = scores.get(i, 0) + self.content_based_weight * content_scores[content_indices.index(i)][1]
            if i in cf_indices:
                scores[i] = scores.get(i, 0) + self.collaborative_weight * (1 - distances[0][list(cf_indices).index(i)])
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get the recommended movies
        recommended_movies = []
        for i in scores:
            recommended_movies.append({'id': self.df.iloc[i[0]]['id'], 'title': self.df.iloc[i[0]]['title'], 'score': i[1]})

        return recommended_movies