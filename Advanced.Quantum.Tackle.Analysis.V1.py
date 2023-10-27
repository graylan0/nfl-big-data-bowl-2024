import openai
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from skopt import gp_minimize
import datetime
import json

# Load API key from config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
        openai.api_key = config["openai_api_key"]
except Exception as e:
    print(f"An error occurred while loading the API key: {e}")

# Function to save results to a Markdown file
def save_to_markdown_file(agent_prompts):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"Results_{timestamp}.md"
        with open(filename, "w") as f:
            f.write("# GPT-3.5 Turbo Responses\n\n")
            for i, prompt in enumerate(agent_prompts):
                f.write(f"## Learning Round {i+1}\n")
                f.write(f"{prompt}\n\n")
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving to Markdown file: {e}")

# Load and preprocess data
try:
    df = pd.read_csv('tracking_week_1.csv')
    team_df = df[df['club'] == 'LA']
except Exception as e:
    print(f"Error reading CSV file: {e}")

# Function to summarize tackle data
def summarize_tackle_data(tackle_data_frame):
    summary = tackle_data_frame.groupby(['gameId', 'nflId']).agg({
        's': 'sum',
        'a': 'sum'
    }).reset_index()
    summary_dict = summary.to_dict(orient='records')
    return summary_dict

# Function to calculate advanced performance metric
# Function to calculate advanced performance metric
def advanced_performance_metric(params):
    try:
        quantum_data = qnode(params)
        tackle_data_summary = summarize_tackle_data(team_df)
        speed_values = [x['s'] for x in tackle_data_summary]
        
        mse = np.mean((np.array(quantum_data) - np.array(speed_values)) ** 2)
        
        # Convert PennyLane tensor to Python float
        mse_scalar = mse.item()
        
        return mse_scalar
    except Exception as e:
        print(f"An error occurred in advanced_performance_metric: {e}")
        raise


# Define a quantum circuit with PennyLane
def quantum_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Initialize a quantum device
dev = qml.device("default.qubit", wires=2)

# Create a QNode
qnode = qml.QNode(quantum_circuit, dev)

# Initialize learning rounds
learning_rounds = 5

# Initialize agent prompts and intercommunication data
agent_prompts = []

# Initialize POOLINSERT and HOLD cache
pool_insert_cache = {}

# System rules
system_rules = """System Rules:
1. Agents must analyze quantum tackle data for advanced insights.
2. Agents should employ advanced strategies for performance improvement.
3. Agents will intercommunicate using a POOLINSERT and HOLD cache mechanism.
4. This will be done over a series of learning rounds to contribute to collective understanding."""

# Main loop for learning rounds
for round in range(learning_rounds):
    try:
        # Optimize the parameters using Bayesian optimization
        params = np.array([0.5, 0.1], requires_grad=True)
        result = gp_minimize(lambda p: advanced_performance_metric(p), [(-3.14, 3.14), (-3.14, 3.14)], n_calls=10, random_state=0)
        params = result.x

        # Execute the quantum circuit to get the quantum_data
        quantum_data = qnode(params)

        # Update POOLINSERT cache
        pool_insert_cache[f'Round_{round+1}'] = quantum_data.tolist()

        # Summarize the tackle data for this round
        tackle_data_summary = summarize_tackle_data(team_df)

        # Generate a prompt for GPT-4 Turbo based on the quantum_data and tackle_data_summary
        messages = [
            {"role": "system", "content": system_rules},
            {"role": "system", "content": "You are a helpful assistant specialized in advanced quantum and data analysis."},
            
            {"role": "user", "content": f"Agent 1, provide an in-depth analysis of the following advanced quantum tackle data: {quantum_data}. Summarized tackle data for this round is: {tackle_data_summary}. Also, suggest any data or insights that should be added to the POOLINSERT cache."},
            
            {"role": "user", "content": f"Agent 2, based on Agent 1's analysis, elaborate on advanced strategies for performance improvement. Refer to POOLINSERT data: {pool_insert_cache}."},
            
            {"role": "user", "content": f"Agent 3, offer a second opinion on the data analysis and strategies suggested by Agents 1 and 2. Cross-reference with POOLINSERT data: {pool_insert_cache}."},
            
            {"role": "user", "content": f"Agent 4, considering the inputs from Agents 1, 2, and 3, provide a risk assessment based on the quantum and tackle data. Also, suggest any preventive measures."},
            
            {"role": "user", "content": f"Agent 5, evaluate the efficiency of the current strategies based on the data and the inputs from all previous agents. Suggest improvements."},
            
            {"role": "user", "content": f"Agent 6, analyze the data for patterns or trends that could be useful for future games, integrating the insights from all previous agents."},
        ]

        # Make the GPT-3.5 Turbo API call
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=messages
        )

        # Store the GPT-3.5 Turbo response
        agent_prompts.append(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"An error occurred during the learning round {round+1}: {e}")

# Output the GPT-3.5 Turbo responses
try:
    for i, prompt in enumerate(agent_prompts):
        print(f"GPT-4 Turbo Response for Learning Round {i+1}: {prompt}")
except Exception as e:
    print(f"Error printing GPT-4 Turbo responses: {e}")

# Save the results to a Markdown file
try:
    save_to_markdown_file(agent_prompts)
except Exception as e:
    print(f"Error saving to Markdown file: {e}")
