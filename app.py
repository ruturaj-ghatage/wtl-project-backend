from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from hybrid_recommender import HybridRecommender

app = Flask(__name__)

cors = CORS(app, resources={
            r"/*": {"origins": "*"}}, supports_credentials=True)
# CORS(app)

hybrid_recommender = HybridRecommender()


@app.route('/')
def index():
    return {
        'message': 'Hello world!'
    }

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    # Load the request data
    data = request.json

    # Validate the request data
    if 'movie_name' not in data:
        return jsonify({'error': 'Missing "movie_id" parameter'}), 400

    # Initialize the recommender system

    # Get the recommendations for the specified movie
    movie_name = data['movie_name']
    recommendations = hybrid_recommender.recommend(movie_name)

    if not recommendations:
        return jsonify({'message': f"No recommendations found for '{movie_name}'."}), 404

    for movie in recommendations:
        movie['id'] = str(movie['id'])
        movie['score'] = str(movie['score'])

    # Return the recommendations
    return make_response(jsonify({'recommendations': recommendations}))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
