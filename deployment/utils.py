import pandas as pd 
from pandas import json_normalize
import seaborn as sns
import numpy as np
from dotenv import dotenv_values
import requests
from bs4 import BeautifulSoup
import re
import json
import codecs
import time  
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from keras.models import load_model
from keras.utils import to_categorical


def get_league_matches(api_key, season_id):
    """
    Retrieve league matches data for a specific season.

    Parameters:
    - api_key (str): API key for accessing the Football Data API.
    - season_id (int): ID of the specific season.

    Returns:
    - response: HTTP response object containing the retrieved data.
    """

    # Construct the URL for accessing league matches data for the specified season
    url = f"https://api.football-data-api.com/league-matches?key={api_key}&season_id={season_id}"

    # Make an HTTP GET request to the defined URL using the requests library
    response = requests.get(url)

    # Return the response object
    return response

def get_api_match_data(api_key, season_ids = [9, 10, 11, 12, 161, 246, 1625, 2012, 3119, 
                                               3121, 3125, 3131, 3137, 4759, 6135, 7704, 9660]):
    """
    Create a Pandas DataFrame by fetching and concatenating league matches data for multiple seasons.

    Parameters:
    - api_key (str): API key for accessing the Football Data API.
    - season_ids (list): List of season IDs for which data will be fetched.

    Returns:
    - concatenated_df: Pandas DataFrame containing concatenated league matches data for the specified seasons.
    """
    # Initialize an empty list to store individual DataFrames for each season
    list_of_dfs = []

    # Iterate through each season ID in the provided list
    for season_id in season_ids:
        try:
            # Fetch league matches data for the current season
            response = get_league_matches(api_key, season_id)
            data = response.json()
            
            # Create a DataFrame from the fetched data
            df = pd.DataFrame(data["data"])
            
            # Append the DataFrame to the list
            list_of_dfs.append(df)
        except:
            # Handle errors and exit the function if an error occurs
            print("There was an error.")
            exit()

    if len(list_of_dfs) > 1:
        # Concatenate the DataFrames in the list to create a single DataFrame
        concatenated_df = pd.concat(list_of_dfs, ignore_index=True)
    else:
        # Concatenate the DataFrames in the list to create a single DataFrame
        concatenated_df = list_of_dfs[0]
    
    # Return the concatenated DataFrame
    return concatenated_df


def filter_match_data_by_date(match_data, date):
    # Convert the 'date_unix' column to datetime
    match_data['date_unix'] = pd.to_datetime(match_data['date_unix'], unit='s')

    match_data['date'] = match_data['date_unix'].dt.date

    relevant_columns = ['id', 'game_week',  'home_name', 'away_name', 
                        'odds_ft_1', 'odds_ft_x', 'odds_ft_2', 'date' ]

    filtered_data = match_data[relevant_columns]

    filtered_data = filtered_data[filtered_data['date'] == date]

    return filtered_data

def scrape_match_data(seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]):
 
    def get_season_html(season):
        # Construct the URL based on the league (EPL) and season
        url = f"https://understat.com/league/EPL/{season}"

        # Send an HTTP GET request to the constructed URL
        response = requests.get(url)

        # Get the content of the response, which typically contains the HTML content of the web page
        html_content = response.content

        # Return the HTML content
        return html_content

    def parse_html_content(html_content):
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all script tags in the HTML
        scripts = soup.find_all('script')

        # Access the script tag at index 2 (change index if needed)
        target_script = scripts[2]

        # Convert the script content to a string
        target_string = str(target_script.contents[0])

        # Decode the string using unicode_escape
        cleaned_string = codecs.decode(target_string, 'unicode_escape')

        # Extract the relevant JSON data from the decoded string
        # (Note: The specific indices [30:-4] may need adjustment based on the data structure)
        teams_data = json.loads(cleaned_string[30:-4])

        # Return the extracted teams_data
        return teams_data
    
    def normalized_dataframe(teams_data):
        # Create an empty list to store individual team DataFrames
        teams_normalized_dfs = []

        # Iterate through each team's data
        for team_id, team_data in teams_data.items():
            # Create a DataFrame from the team's data
            team_df = pd.DataFrame(team_data)

            # Normalize the 'history' column using json_normalize and concatenate it with the original DataFrame
            team_normalized_df = pd.concat([team_df.drop(['history'], axis=1), 
                                            json_normalize(team_df['history'])], axis=1)

            # Append the normalized DataFrame to the list
            teams_normalized_dfs.append(team_normalized_df)

        # Return the final DataFrame
        return teams_normalized_dfs
    
    # Create an empty list to store normalized DataFrames
    normalized_dfs = []

    # Iterate through each season
    for season in seasons:
        # Fetch HTML content for the current season
        season_html_content = get_season_html(season)

        # Parse HTML content to obtain data
        season_parsed_data = parse_html_content(season_html_content)

        # Create normalized DataFrame for the current season
        season_normalized_df = normalized_dataframe(season_parsed_data)

        # Extend the list with the normalized DataFrames for the current season
        normalized_dfs.extend(season_normalized_df)

        # Add a 5-second delay before fetching data for the next season
        time.sleep(5)

    # The 'normalized_dfs' list now contains all the normalized DataFrames for each season

    # Create a single DataFrame by concatenating all individual team DataFrames
    final_df = pd.concat(normalized_dfs, ignore_index=True)
    
    return final_df 

def aggregate_api_scraped_data(api_key):
    api_data = get_api_match_data(api_key)  
    scraped_data = scrape_match_data()
    def winningTeam(row):
        # Extracting the goal counts for the home and away teams from the row
        homeGoalCount = row['homeGoalCount']
        awayGoalCount = row['awayGoalCount']
        
        # Checking which team won based on the goal counts
        if homeGoalCount > awayGoalCount:
            return 1  # 1 means home team won
        elif awayGoalCount > homeGoalCount:
            return 2  # 2 means away team won
        else:
            return 0  # 0 indicates a draw
    
    # Apply the winningTeam function to each row of the dataframe
    api_data['1x2'] = api_data.apply(lambda row: winningTeam(row), axis=1)
    
    
    def convert_dates(api_data, scraped_data):
        # Convert the 'date_unix' column to datetime
        api_data['date_unix'] = pd.to_datetime(api_data['date_unix'], unit='s')

        api_data['date'] = api_data['date_unix'].dt.date

        # Convert 'date' column to datetime format
        scraped_data['date'] = pd.to_datetime(scraped_data['date'])

        # Extract the date part from the 'date' column
        scraped_data['date'] = scraped_data['date'].dt.date

    convert_dates(api_data, scraped_data)

    def harmonize_names(api_data, scraped_data):
        # Define a dictionary that maps original team names to their corresponding new names in the scraped_data 
        names = {'Leicester': 'Leicester City', 'West Ham': 'West Ham United', 
                'Tottenham': 'Tottenham Hotspur', 'Swansea':'Swansea City', 
                'Stoke': 'Stoke City', 'Hull': 'Hull City', 'Bournemouth': 'AFC Bournemouth',
                'Norwich':'Norwich City', 'Huddersfield':'Huddersfield Town', 
                'Brighton':'Brighton & Hove Albion', 'Cardiff':'Cardiff City', 'Leeds': 'Leeds United', 
                'Luton':'Luton Town'}
        
        # Replace names in the 'title' column
        scraped_data['title'].replace(names, inplace=True)
    
    harmonize_names(api_data, scraped_data)

    def harmonize_teamID(api_data, scraped_data):
        # Group the data by 'homeID' and get unique team names for each group
        grouped_data = api_data.groupby('homeID')['home_name'].unique()

        # Create a dictionary using zip
        # - Explode the nested lists in 'home_name' to individual elements
        # - Use zip to pair up each team name with its corresponding 'homeID'
        # - Convert the pairs into a dictionary
        team_id_mapping = dict(zip(grouped_data.explode(), grouped_data.index))

        # Map team names to their corresponding teamID using the team_id_mapping dictionary
        scraped_data['teamID'] = scraped_data['title'].map(team_id_mapping)

    harmonize_teamID(api_data, scraped_data)

    def aggregation(api_data, scraped_data):
        # Define a list of features from the 'scraped_data' dataframe to include in the merged dataframe
        scraped_data_feats = ['xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 
                            'xpts', 'npxGD', 'ppda.att', 'ppda.def', 'ppda_allowed.att', 'ppda_allowed.def']
        # sort scraped_data
        scraped_data = scraped_data.sort_values(by='date', ascending=False)

        # Create empty columns for home and away features
        api_data = api_data.assign(**{f"{feat}_home": None for feat in scraped_data_feats})
        api_data = api_data.assign(**{f"{feat}_away": None for feat in scraped_data_feats})
        
        # Iterate through each row in api_data
        for index, row in api_data.iterrows():
            home_id = row['homeID']
            away_id = row['awayID']
            match_date = row['date']
            
            # Retrieve home stats from scraped_data dataframe
            home_result = scraped_data[(scraped_data['teamID'] == home_id) & (scraped_data['date'] < match_date)]
            # get mean for each statistics for the last  15 games before current match
            home_result_mean = home_result.head(15).describe().loc['mean']
            # Check if there are matching records
            if not home_result.empty:
                # Take the first matching record
                home_filtered_result_dict = home_result_mean.to_dict()
                
                # Filter out irrelevant columns from result
                home_filtered_result_dict = {key: value for key, value in home_filtered_result_dict.items() if key in scraped_data_feats}
                
                # Update values in the dataframe
                for key, value in home_filtered_result_dict.items():
                    api_data.at[index, f'{key}_home'] = value

            # Retrieve away stats from scraped_data dataframe
            away_result = scraped_data[(scraped_data['teamID'] == away_id) & (scraped_data['date'] < match_date)]
            # get mean for each statistics for the last  15 games before current match
            away_result_mean = away_result.head(15).describe().loc['mean']
            # Check if there are matching records
            if not away_result.empty:
                # Take the first matching record
                away_filtered_result_dict = away_result_mean.to_dict()
                
                # Filter out irrelevant columns from result
                away_filtered_result_dict = {key: value for key, value in away_filtered_result_dict.items() if key in scraped_data_feats}
                
                # Update values in the dataframe
                for key, value in away_filtered_result_dict.items():
                    api_data.at[index, f'{key}_away'] = value
        return api_data
    return aggregation(api_data, scraped_data)


def model_prediction(aggregated_data, match_date, model_features_targets):
    # select useful columns from aggregated_data
    relevant_columns = ['id', 'home_name', 'away_name', 'date'] + model_features_targets
    relevant_data = aggregated_data[relevant_columns]
    # Convert 'date' column to datetime format
    relevant_data['date'] = pd.to_datetime(relevant_data['date'])
    # Extract the date part from the 'date' column
    relevant_data['date'] = relevant_data['date'].dt.date

    date_matches = relevant_data[relevant_data['date'] == match_date] 

    try:
    
        def data_preprocessing(date_matches, model_features_targets):
            date_matches.dropna(inplace=True)
            # Separate features (X) and target variable (y)
            X = date_matches[model_features_targets].drop(columns=['1x2']) # Features
            y = date_matches['1x2']  # Target variable
            # Standardize the features using StandardScaler
            scaler = StandardScaler()

            # Fit and transform the training set to standardize its features
            X_scaled = scaler.fit_transform(X)

            # Use LabelEncoder to convert the target variable to numeric values
            label_encoder = LabelEncoder()

            # Fit and transform the training set labels to numeric values
            y_encoded = label_encoder.fit_transform(y)

            # Convert labels to one-hot encoding
            y_encoded_categorical = to_categorical(y_encoded)


            return X_scaled, y_encoded_categorical
        def make_predictions(date_matches, model_features_targets):
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "deployment_models", "base_model")
            X_encoded, y_encoded_categorical = data_preprocessing(date_matches, model_features_targets)
            model = load_model(model_path)
            predictions = model.predict(X_encoded)

            return predictions

        def create_predictions_table(date_matches, model_features_targets):
            predictions = make_predictions(date_matches, model_features_targets)
            predictions_df = pd.DataFrame(predictions * 100, columns=['draw prob (%)', 'homeWin prob (%)', 'awayWin prob (%)'])

            # Round the values to 2 decimals
            predictions_df = predictions_df.round(2)

            # Reset the index of date_matches before concatenating
            date_matches_reset = date_matches.reset_index(drop=True)

            # Concatenate predictions_df and date_matches_reset along columns (axis=1)
            result_df = pd.concat([date_matches_reset[['home_name', 'away_name']], predictions_df], axis=1)

            return result_df

        
        return create_predictions_table(date_matches, model_features_targets)
    except ValueError:
        # List of column names
        columns = ['home_name', 'away_name', 'draw prob (%)', 'homeWin prob (%)', 'awayWin prob (%)']

        # Create an empty DataFrame
        empty_df = pd.DataFrame(columns=columns)

        return empty_df
        
    







    



