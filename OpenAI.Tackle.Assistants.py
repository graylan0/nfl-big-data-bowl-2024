import openai
import json
import os
from collections import Counter

# Ensure you have the openai package installed: pip install openai

# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Make sure the API key is available
if not openai.api_key:
    raise ValueError("The OpenAI API key has not been set in the environment variables.")

# Function to load the JSON data
def load_data(file_path):
    with open(file_path, 'r') as file:
        games_data = json.load(file)
    return games_data

# Function to analyze the data for insights
def analyze_data(games_data):
    # Extract basic information
    total_games = len(games_data)
    season_counts = Counter(game['season'] for game in games_data)
    weeks_counts = Counter(game['week'] for game in games_data)
    
    # Calculate score-related statistics
    home_scores = [game['homeFinalScore'] for game in games_data]
    visitor_scores = [game['visitorFinalScore'] for game in games_data]
    average_home_score = sum(home_scores) / total_games
    average_visitor_score = sum(visitor_scores) / total_games
    total_scores = [sum(score) for score in zip(home_scores, visitor_scores)]
    average_total_score = sum(total_scores) / total_games
    
    # Most common home and visitor teams
    home_teams = Counter(game['homeTeamAbbr'] for game in games_data)
    visitor_teams = Counter(game['visitorTeamAbbr'] for game in games_data)
    
    # Report summarization
    summary_report = {
        'total_games': total_games,
        'seasons': season_counts,
        'weeks': weeks_counts,
        'average_home_score': average_home_score,
        'average_visitor_score': average_visitor_score,
        'average_total_score': average_total_score,
        'most_common_home_teams': home_teams.most_common(5),
        'most_common_visitor_teams': visitor_teams.most_common(5)
    }
    
    return summary_report

# Function to create an assistant
def create_assistant():
    assistant = openai.Assistant.create(
        instructions="Multiversal AI Designed to give coaching insights to NFL Teams.",
        model="gpt-4-1106-preview",
        tools=[{"type": "code_interpreter"}]
    )
    return assistant

# Function to generate insights using the assistant
def generate_insights(assistant, summary_report):
    # Use the assistant to generate insights based on the summary report
    # This is a placeholder for the OpenAI API call
    insights = "Generated insights based on the summary report"
    return insights

# Main function to run the analysis
def main():
    file_path = '/mnt/data/games.json'  # Replace with your actual file path
    games_data = load_data(file_path)
    summary_report = analyze_data(games_data)
    assistant = create_assistant()
    insights = generate_insights(assistant, summary_report)
    print(insights)

if __name__ == "__main__":
    main()
