__version__ = "0.9.0"
__author__ = "Enrico Altavilla"

import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from nltk.corpus import stopwords

# Global random seed for reproducibility
RANDOM_SEED = 42


# Function to load Excel file and prompt user for input columns
def load_data(file_name):
    """
    Loads search query data and calculates the change in position from an Excel file.
    
    Parameters:
    - file_name (str): The name of the Excel file containing the data.
    
    Returns:
    - queries (list of str): List of search queries.
    - position_changes (list of float): List of calculated changes in positions.
    """
    df = pd.read_excel(file_name, sheet_name=0)

    query_col_index = 0
    new_position_col_index = 1
    old_position_col_index = 2

    num_columns = len(df.columns)

    if num_columns != 3:
        print("Columns available in the Excel sheet:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")

        query_col_index = int(input("Enter the number of the column containing the search queries: "))
        new_position_col_index = int(input("Enter the number of the column containing the more recent positions (NEW positions): "))
        old_position_col_index = int(input("Enter the number of the column containing the less recent positions (PREVIOUS positions): "))

        if any(index >= num_columns for index in [query_col_index, new_position_col_index, old_position_col_index]):
            print("Error: The specified column numbers are not valid.")
            return None, None

    df_filtered = df.iloc[:, [query_col_index, new_position_col_index, old_position_col_index]]
    df_filtered.columns = ['queries', 'new_positions', 'old_positions']

    return df_filtered

# Clean the input data
def clean_data(df_filtered):
    """
    Clean the input data by removing duplicates and rows with missing values.

    Parameters:
    - df_filtered (DataFrame): Filtered DataFrame containing search queries and positions.

    Returns:
    - df_filtered (DataFrame): Cleaned DataFrame.
    """
    # Remove any duplicate queries
    df_filtered = df_filtered.drop_duplicates(subset=['queries'])

    # Remove any rows with missing values in the 'new_positions' or 'old_positions' columns
    df_filtered = df_filtered.dropna(subset=['new_positions', 'old_positions'])

    # Substitute the value 0 with 101 in the columns 'new_positions' and 'old_positions'
    df_filtered['new_positions'] = df_filtered['new_positions'].replace(0, 101)
    df_filtered['old_positions'] = df_filtered['old_positions'].replace(0, 101)

    # Remove any rows where the 'new_positions' or 'old_positions' are 0
    # df_filtered = df_filtered[(df_filtered['new_positions'] != 0) & (df_filtered['old_positions'] != 0)]

    # Clip the 'new_positions' values to the 99th percentile
    # df_filtered['new_positions'] = df_filtered['new_positions'].clip(upper=df_filtered['new_positions'].quantile(0.99))

    return df_filtered



# Calculate the change in position for each query
def calculate_position_changes(df_filtered, focus_on_top=False):
    """
    Calculate the change in position for each search query.

    Parameters:
    - df_filtered (DataFrame): Filtered DataFrame containing search queries and positions.
    - focus_on_top (bool): If True, apply exponential decay to give more weight to changes in top positions.

    Returns:
    - queries (list of str): List of search queries.
    - position_changes (list of float): List of calculated changes in positions.
    """

    # Calculate the change in position for each query
    df_filtered['change'] = df_filtered['old_positions'] - df_filtered['new_positions']

    # This applies an exponential decay to give more weight to changes in the top positions
    if focus_on_top:
        k = 20  # Decay constant
        
        df_filtered['change'] = df_filtered['change'] * np.exp(-np.minimum(df_filtered['old_positions'], df_filtered['new_positions']) / k)

    queries = df_filtered['queries'].tolist()
    position_changes = df_filtered['change'].tolist()

    return queries, position_changes




# Function to analyze topics using Non-negative Matrix Factorization (NMF)
# DEPRECATED: Use find_optimal_topics instead
def analyze_topics(queries, position_changes, stopwords_language=None, n_components=5):
    """
    Analyzes search queries using NMF to identify latent topics and how they relate to ranking changes.
    
    Parameters:
    - queries (list of str): List of search queries.
    - position_changes (list of float): List of calculated changes in positions.
    - stopwords_language (str): Language for stopwords (e.g., 'english', 'italian').
    - n_components (int): Number of topics to identify.

    Returns:
    - topic_coefficients (array): Coefficients of topics from linear regression.
    - components (array): Components of the NMF model.
    - terms (list of str): List of terms from the vectorizer.
    """
    stop_words = stopwords.words(stopwords_language) if stopwords_language else []
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=5, ngram_range=(1, 3))
    X = vectorizer.fit_transform(queries)

    nmf = NMF(n_components=n_components, random_state=RANDOM_SEED)
    X_topics = nmf.fit_transform(X)

    regressor = LinearRegression()
    regressor.fit(X_topics, position_changes)

    topic_coefficients = regressor.coef_
    terms = vectorizer.get_feature_names_out()

    return topic_coefficients, nmf.components_, terms


# Function to compute outliers in topic coefficients
# Return the number of coefficients that are more than 2.5 standard deviations away from the mean
def get_outliers(X_topics, position_changes):
    """
    Compute the number of outliers in the topic coefficients based on linear regression.
    
    Parameters:
    - X_topics (array): Document-topic matrix from NMF.
    - position_changes (list of float): List of calculated changes in positions.

    Returns:
    - Number of coefficients that are more than 2.5 standard deviations away from the mean.
    """
    regressor = LinearRegression()
    regressor.fit(X_topics, position_changes)
    topic_coefficients = regressor.coef_
    coef_std_dev = np.std(topic_coefficients)
    return np.sum(np.abs(topic_coefficients) > 2.5 * coef_std_dev)


# Function to find the optimal number of topics using NMF
def find_optimal_topics(X, position_changes, min_topics=40, max_topics=80):
    """
    Find the optimal number of topics using NMF with custom initialization to speed up the process.
    
    Parameters:
    - X (array): Document-term matrix.
    - position_changes (list of float): List of calculated changes in positions.
    - min_topics (int): Minimum number of topics to consider.
    - max_topics (int): Maximum number of topics to consider.

    Returns:
    - best_nmf (NMF): NMF model with the optimal number of topics.
    """

    most_outliers = 0

    # How every iterations to run NMF with the (slower) nndsvda initialization
    how_often = 1

    # This guarantees that the W and H matrices are initialized in the first iteration
    cycle = how_often
    np.random.seed(RANDOM_SEED)

    # Get the time
    start_time = time.time()

    for n in range(min_topics, max_topics + 1):

        print(f"Cycle: {n}/{max_topics}", end='\r')

        # Run NMF with nndsvda initialization only every how_often iterations
        if cycle % how_often == 0:
            nmf = NMF(n_components=n, init='nndsvda', tol=1e-3, random_state=RANDOM_SEED)
            W = nmf.fit_transform(X)
            H = nmf.components_
        
        # Reuse the matrices from the previous run and extend them with random values
        else:
            W_init = np.hstack([W, np.random.rand(W.shape[0], 1)])
            H_init = np.vstack([H, np.random.rand(1, H.shape[1])])

            nmf = NMF(n_components=n, init='custom', tol=1e-3, random_state=RANDOM_SEED)
            W = nmf.fit_transform(X, W=W_init, H=H_init)
            H = nmf.components_


        outliers = get_outliers(W, position_changes)
        #print(f"Topics: {n} - Outliers: {outliers}")

        if outliers >= most_outliers:
            most_outliers = outliers
            best_nmf = nmf

        cycle += 1


    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    return best_nmf


# Main function
def main():

    # Get the filename from the command line parameters
    params = sys.argv[1:]
    if len(params) == 0:
        print("Usage: python visibility-topics.py <file_name>")
        return
    
    file_name = params[0]


    # Load the data
    df = load_data(file_name)

    if df is None:
        return

    # Clean the data
    df = clean_data(df)

    if df is None:
        return

    # Calculate the position changes
    queries, position_changes = calculate_position_changes(df, focus_on_top=False)

    # Scale the position changes
    scaler = StandardScaler()
    position_changes = scaler.fit_transform(np.array(position_changes).reshape(-1, 1)).flatten()

    if queries is None or position_changes is None:
        return  # Exit if loading data failed

    # Use stopwords for Italian
    stopwords_language = 'italian'
    stop_words = stopwords.words(stopwords_language)

    # Vectorize the search queries
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=5, ngram_range=(1, 3))
    X = vectorizer.fit_transform(queries)

    # Find the optimal NMF object
    best_nmf = find_optimal_topics(X, position_changes)

    # Access the final W (document-topic matrix) from the NMF object
    X_topics = best_nmf.transform(X)
    
    # Perform linear regression on the topics and position_changes
    regressor = LinearRegression()
    regressor.fit(X_topics, position_changes)

    # Access the topic coefficients from the regression
    topic_coefficients = regressor.coef_

    # Access the H (topic-term matrix) from the NMF object
    components = best_nmf.components_
    terms = vectorizer.get_feature_names_out()

    # Find the top term and top 5 terms for each topic
    top_terms = []
    top_terms_for_topics = []
    for i in range(best_nmf.n_components):
        topic_term_weights = components[i]

        sorted_terms = sorted(zip(terms, topic_term_weights), key=lambda x: x[1], reverse=True)[:10]

        top_terms.append(sorted_terms[0][0])

        # Get the top terms that have a weight of at least 0.5, formatted for the tooltip
        top_terms_for_topic = '<br>'.join([f"{term} ({round(weight, 1)})" for term, weight in sorted_terms if weight >= 0.5])
        top_terms_for_topics.append(top_terms_for_topic)

    # Combine the top terms and coefficients into a list of tuples and sort by coefficient
    sorted_topics = sorted(zip(top_terms, topic_coefficients, top_terms_for_topics), key=lambda x: x[1], reverse=True)
    sorted_top_terms, sorted_coefficients, sorted_top_5_terms = zip(*sorted_topics)

    # Create Plotly bar chart
    fig = go.Figure(data=[go.Bar(
        x=sorted_top_terms, 
        y=sorted_coefficients, 
        marker_color=sorted_coefficients, 
        text=[f'{round(coef, 1)}' for coef in sorted_coefficients],
        textposition='outside',
        hovertext=sorted_top_5_terms,
        hoverinfo='text'
    )])

    fig.update_layout(
        title=dict(text="How topics were affected by the ranking changes",
                    font=dict(size=24,
                        family="Trebuchet MS",
                        color='white',
                    )),
        xaxis_title="Top Topics",
        yaxis_title="Coefficient",
        plot_bgcolor='rgb(54,69,79)',
        paper_bgcolor='rgb(54,69,79)',
        font=dict(color='white'),
    )

    fig.show()





if __name__ == "__main__":
    main()
