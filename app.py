from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
movies_df = pd.read_csv(
    'https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv', encoding='ISO-8859-1')

# Remove NaN values
movies_df = movies_df.dropna()

# Drop unnecessary columns
movies_df = movies_df.drop(['index', 'budget', 'homepage', 'id', 'keywords', 'original_language', 'original_title', 'overview', 'popularity', 'production_companies',
                           'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'vote_average', 'vote_count'], axis=1)

# Replace any spaces in the movie titles with underscores
movies_df['title'] = movies_df['title'].str.replace(' ', '_')

# Create a matrix of user ratings
cv = CountVectorizer()
genres_matrix = cv.fit_transform(movies_df['genres'])

# Calculate similarity between movies
cosine_sim = cosine_similarity(genres_matrix)

app = Flask(__name__)

cors = CORS(app, resources={
            r"/*": {"origins": "http://localhost:3001"}}, supports_credentials=True)
# CORS(app)


# Function to get the top 5 most similar movies
def get_recommendations(title, cosine_sim):

    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies_df['title'].iloc[movie_indices].values.tolist()


@app.route('/')
def index():
    return {
        'message': 'Hello world!'
    }


@app.route('/recommendations', methods=['POST', 'OPTIONS'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'], supports_credentials=True)
def recommendations():
    title = request.json['title']
    recommendations = get_recommendations(title, cosine_sim)
    response = make_response(jsonify(recommendations))
    return response


if __name__ == '__main__':
    app.run(port=5000)
