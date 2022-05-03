import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# calling the data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# calling the model
model = joblib.load("../models/classifier.pkl")


# The page shows a detail of the results and receives new data
@app.route('/')
@app.route('/index')
def index():
    
    # extract data

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create the data



    categories =  df[df.columns[4:]]
    cate_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)
    
    #Distributions of the by genre

    mp_counts = df.groupby('genre').count()['related']
    mp_names = list(mp_counts.index)
    
    # Top 7 categories
    remove_col = ['id', 'message', 'original', 'genre']
    y = df.loc[:, ~df.columns.isin(remove_col)]
    category_counts = y.sum().sort_values().tail(7)
    category_names = list(category_counts.index)
    
   
    # Graphs
   
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        #---+++i
        #category plotting (visualization#2)
        {
            'data': [
                Bar(
                    x=cate_names,
                    y=cate_counts
                )
            ],
            'layout': {
                'title': 'Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=mp_names,
                    y=mp_counts
                )
            ],

            'layout': {
                'title': 'Genres by Relate Categorie',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },
        {
            'data': [
                Bar(
                    y=category_names,
                    x=category_counts,
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Top 7 Categories',
                'xaxis': {
                    'title': "Count"
                }
            }
        }
    ]
 
    #---+++f          
    # Graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
