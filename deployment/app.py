import streamlit as st
from utils import aggregate_api_scraped_data, model_prediction
from dotenv import dotenv_values


# Initialize session state
if 'last_date' not in st.session_state:
    st.session_state.last_date = None

# Function to retrieve match data from Footystats API
@st.cache_data(ttl=86400)  # Cache for 24 hours (86400 seconds)
def aggregate_data(api_key):
    return aggregate_api_scraped_data(api_key=api_key)


api_key = st.secrets['API_KEY']

st.markdown("""
<style>
  .main > div {
    padding-left: 0rem;
    padding-right: 0rem;
  }
</style>
""", unsafe_allow_html=True)

# set title with white text
st.title('EPL Precision Predictor')

# set header with white text
st.header('Please match date to get prediction')

# Render the calendar and get the selected date
selected_date = st.date_input("Select a date")

# Use the selected date in other parts of your logic
st.write("Selected date:", selected_date)
# Use the selected_date in your logic...

# Retrieve match data only if the date has changed
if st.session_state.last_date != selected_date:
    st.session_state.last_date = selected_date
    matches_df = aggregate_data(api_key)
    st.session_state.matches_df = matches_df
else:
    matches_df = st.session_state.matches_df

# define feature columns
features_targets =  [
    'odds_ft_1', 'odds_ft_x', 'odds_ft_2',
    'home_ppg', 'away_ppg',
    'xG_home', 'xG_away', 'xGA_home', 'xGA_away',
    'npxG_home', 'npxG_away', 'npxGA_home', 'npxGA_away',
    'deep_home', 'deep_away', 'deep_allowed_home', 'deep_allowed_away',
    'scored_home', 'scored_away', 'missed_home', 'missed_away',
    'xpts_home', 'xpts_away',
    'npxGD_home', 'npxGD_away',
    'ppda.att_home', 'ppda.att_away', 'ppda.def_home', 'ppda.def_away',
    'ppda_allowed.att_home', 'ppda_allowed.att_away',
    'ppda_allowed.def_home', 'ppda_allowed.def_away',
    '1x2'
]
# Filter matches data to include matches for the selected date only
predictions_df = model_prediction(matches_df, selected_date, features_targets) 

# render DataFrame 
st.dataframe(predictions_df)
