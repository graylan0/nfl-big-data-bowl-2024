import openai
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import datetime
import json
import os
import matplotlib.pyplot as plt
import pyswarms as ps
import seaborn as sns
from llama_cpp import Llama 

llm = Llama(
  model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
  n_gpu_layers=-1,
  n_ctx=3900,
)
# Number of rounds to select
num_selected_rounds = 5  # You can change this to 10 or any other number
# Define the quantum device
dev = qml.device("default.qubit", wires=4)

def analyze_with_llama(agent_prompts):
    llama_responses = []
    for prompt in agent_prompts:
        try:
            # Use Llama to analyze the GPT-4 response
            llama_response = llm(prompt)
            llama_responses.append(llama_response)
        except Exception as e:
            print(f"An error occurred while using Llama to analyze the prompt: {e}")
    return llama_responses

# Define the QNN
@qml.qnode(dev)
def quantum_circuit(params, feature_vector):
    qml.templates.BasicEntanglerLayers(params, wires=range(4))
    qml.templates.MottonenStatePreparation(feature_vector, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class GPT4:
    @staticmethod
    def interpret_quantum_score(quantum_score, llm):
        # Normalize the quantum score to a range between 0 and 1
        normalized_score = (quantum_score + 1) / 2.0  # Assuming quantum_score is between -1 and 1

        # Map the normalized score to a color using a colormap (here we use a red-to-green colormap)
        rgba_color = cm.RdYlGn(normalized_score)
        hex_color = '#%02x%02x%02x' % (int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))

        # Use the model to interpret the color code (assuming llm is your Llama model instance)
        try:
            detailed_prompt = (
                "Agent, you are tasked with interpreting the following color code "
                "which represents a quantum score. Please provide a detailed interpretation.\n"
                f"Color Code: {hex_color}"
            )
            llama_response = llm(detailed_prompt, max_tokens=100)
            interpretation = llama_response['choices'][0]['text']
        except Exception as e:
            interpretation = f"An error occurred while using Llama to interpret the color code: {e}"

        return interpretation, hex_color


# Define the cost function for ORCASWARM
def orca_cost_function(params):
    feature_vector = np.array([0.1, 0.2, 0.3, 0.4])
    quantum_score = quantum_circuit(params.reshape((1, 4)), feature_vector)
    interpretation, color = GPT4.interpret_quantum_score(quantum_score)
    return -np.mean(quantum_score), interpretation, color  # Negate because PSO minimizes

# Initialize ORCASWARM
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=4, options=options)

# Perform optimization using ORCASWARM
best_cost, best_params = optimizer.optimize(orca_cost_function, iters=100)

# Use the best_params to get the optimized quantum score
optimized_quantum_score, interpretation, color = orca_cost_function(best_params)

# Generate heatmap
heatmap_data = np.random.rand(4, 4)  # Replace with actual data
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# Convert heatmap to HTML color codes
hex_colors = sns.color_palette("coolwarm", as_cmap=True)(heatmap_data).as_hex()

# Function to interface with real Llama2
async def real_llama2_agent(heatmap_colors, historical_data):
    try:
        detailed_prompt = (
            "Agent, you are tasked with analyzing the following heatmap colors. "
            "Please adhere to the following rules and guidelines:\n"
            "1. Provide a detailed breakdown of the color distribution.\n"
            "2. Identify any patterns or anomalies.\n"
            "3. Suggest potential areas for further investigation.\n"
            "4. Cross-reference with the following historical data: {}\n"
            "5. Summarize your findings in a concise manner.\n"
            "Heatmap Colors: {}"
        ).format(historical_data, heatmap_colors)
        
        llama2_response = await llm(detailed_prompt, max_tokens=500)
        analyzed_text = "Real Llama2 analyzed heatmap colors based on the rules: {}".format(llama2_response['choices'][0]['text'])
        return analyzed_text
    except Exception as e:
        error_message = "An error occurred while using Llama2: {}".format(e)
        return error_message

llama2_result = real_llama2_agent(hex_colors)
print(f"Llama2 Agent Result: {llama2_result}")

# Save the results for data interoperability
result_data = {
    'best_params': best_params.tolist(),
    'optimized_quantum_score': optimized_quantum_score,
    'interpretation': interpretation,
    'color': color,
    'heatmap_colors': hex_colors.tolist(),
    'llama2_result': llama2_result
}

with open('orca_results.json', 'w') as f:
    json.dump(result_data, f)
# Function to load past learning rounds from Markdown files
def load_past_learning_rounds(directory):
    past_rounds = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                with open(os.path.join(directory, filename), "r") as f:
                    past_rounds.append(f.read())
        print("Past learning rounds loaded successfully.")
        return past_rounds
    except Exception as e:
        print(f"An error occurred while loading past learning rounds: {e}")
        return None

# Define a custom quantum function using PennyLane
def custom_quantum_function(params):
    qml.RX(params[0], wires=0)
    qml.RZ(params[2], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    qml.RZ(params[4], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Initialize a device
dev = qml.device("default.qubit", wires=2)

# QNode
circuit = qml.QNode(custom_quantum_function, dev)

# Function to load NFL data
def load_nfl_data(directory):
    try:
        nfl_data = {}
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                nfl_data[filename] = pd.read_csv(os.path.join(directory, filename))
        print("NFL data loaded successfully.")
        return nfl_data
    except Exception as e:
        print(f"An error occurred while loading NFL data: {e}")
        return None

# Function to save results to a Markdown file
def save_to_markdown_file(agent_prompts):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"Results_{timestamp}.md"
        with open(filename, "w") as f:
            f.write("# GPT4 Responses\n\n")
            for i, prompt in enumerate(agent_prompts):
                f.write(f"## Learning Round {i+1}\n")
                f.write(f"{prompt}\n\n")
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving to Markdown file: {e}")

def evolution_round(past_rounds, agent_prompts):
    try:
        print("Analyzing data from past and current learning rounds...")
        
        # Combine insights from past and current rounds
        combined_insights = past_rounds + agent_prompts
        
        # Analyze the combined insights
        analyzed_data = "Analyzed Data: " + str(combined_insights.count("important_keyword"))
        
        # Update strategies based on the analyzed data
        updated_strategies = "Updated Strategies: Use more quantum computing" if "quantum" in analyzed_data else "No update"
        
        # Re-run the quantum circuit with new parameters if needed
        new_params = np.array([0.6, 0.7, 0.8, 0.9, 1.0])  # Example new parameters
        new_quantum_result = circuit(new_params)
        
        print(f"New Quantum Result: {new_quantum_result}")
        print(f"Updated Strategies: {updated_strategies}")
        
        # Save the evolution round results to a Markdown file
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"Evolution_Round_{timestamp}.md"
        with open(filename, "w") as f:
            f.write("# Evolution Round Results\n\n")
            f.write(f"## Analyzed Data\n{analyzed_data}\n\n")
            f.write(f"## Updated Strategies\n{updated_strategies}\n\n")
            f.write(f"## New Quantum Result\n{new_quantum_result}\n\n")
        
        print(f"Evolution round results saved to {filename}")
        
    except Exception as e:
        print(f"An error occurred during the evolution round: {e}")

# Main function
async def main():
    data_directory = "C:\\Users\\Shadow\\nfl\\data"
    nfl_data = load_nfl_data(data_directory)

    if nfl_data is not None:
        selected_file = 'players.csv'
        if selected_file in nfl_data:
            df = nfl_data[selected_file]
            mean_speed = df['Speed'].mean()
            mean_strength = df['Strength'].mean()
            var_speed = df['Speed'].var()
            var_strength = df['Strength'].var()
            skew_strength = df['Strength'].skew()

            # Create a parameter vector based on these statistics
            params = np.array([mean_speed, mean_strength, var_speed, var_strength, skew_strength])

            # Quantumize the NFL data
            quantum_result = circuit(params)
            print(f"Quantum result for selected NFL data: {quantum_result}")


    # Run the Evolution Round
    past_rounds = load_past_learning_rounds("C:\\Users\\Shadow\\nfl\\learn")
    evolution_round(past_rounds, agent_prompts)
    
    # Analyze GPT-4 responses using Llama
    llama_responses = analyze_with_llama(agent_prompts)
    
    # Output the Llama responses
    for i, response in enumerate(llama_responses):
        print(f"Llama Response for Learning Round {i + 1}: {response}")
    if past_rounds is not None:
        selected_past_rounds = past_rounds[:num_selected_rounds]
        for i, round_content in enumerate(selected_past_rounds):
            print(f"Content of Past Learning Round {i + 1}:\n{round_content}")

    # Generate heatmap
    heatmap_data = np.random.rand(4, 4)  # Replace with actual data
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
    
    # Save the heatmap as an image
    heatmap_image_path = "heatmap.png"
    plt.savefig(heatmap_image_path)
    plt.show()

    # Convert heatmap to HTML color codes
    hex_colors = sns.color_palette("coolwarm", as_cmap=True)(heatmap_data).as_hex()

    # Use real Llama2 for heatmap analysis
    llama2_result = await real_llama2_agent(hex_colors)
    print(llama2_result)

    # Save the results to a Markdown file
    save_to_markdown_file(agent_prompts, heatmap_image_path)

# Load API key from config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
        openai.api_key = config["openai_api_key"]
except Exception as e:
    print(f"An error occurred while loading the API key: {e}")

# Define agent_prompts before running the main function
agent_prompts = []

# Main loop for learning rounds
learning_rounds = 5  # You can change this to 10 or any other number

# System rules and TUI guide
system_rules_and_tui_guide = """## System Rules:
1. Agents must analyze quantum tackle data for advanced insights.
2. Agents should employ advanced strategies for performance improvement.
3. Agents will intercommunicate using a POOLINSERT and HOLD cache mechanism.
4. This will be done over a series of learning rounds to contribute to collective understanding.

## Text-based User Interface (TUI) Demo Guide for NFL Data Analysis

**Note: This is a TUI demo guide. Please interact with it as described below.**

You are now in the TUI mode. Below are the commands you can issue:

- `select [filename]`: To select a file for in-depth analysis.
- `go back`: To go back to the previous directory.
- `cd [directory_name]`: To change the current directory.
- `search [keyword]`: To search for files or directories containing the keyword.

### Examples:

1. **To inspect a file named player :**
    ```
    select players.csv
    ```
2. **To inspect for player data `:**
    ```
    select games.csv
    ```
3. **To inspect for tackle data:**
    ```
    select tackles.csv
    ```
"""

# Your existing code for the learning rounds loop
for round in range(learning_rounds):
    try:
        # Simulated data for demonstration
        quantum_data = f"Quantum Data for Round {round + 1}"
        tackle_data_summary = f"Tackle Data Summary for Round {round + 1}"
        POOLINSERT = f"Simulated POOLINSERT data for Round {round + 1}"

        # Generate a detailed prompt for GPT-4
        messages = [
            {"role": "system", "content": system_rules_and_tui_guide},
            {"role": "system", "content": "You are a specialized assistant in advanced quantum and data analysis."},
            {"role": "user", "content": f"Agent 1, refer to the TUI guide above. Provide an in-depth analysis of the quantum tackle data: {quantum_data}. Also, analyze the summarized tackle data: {tackle_data_summary}. Recommend any data for the POOLINSERT cache."},
            {"role": "user", "content": f"Agent 2, refer to the TUI guide above. Based on Agent 1's insights, propose advanced strategies for performance improvement. Cross-reference with POOLINSERT: {POOLINSERT}."},
            {"role": "user", "content": f"Agent 3, refer to the TUI guide above. Offer a second opinion on the data analysis and strategies suggested by Agents 1 and 2. Cross-reference with POOLINSERT: {POOLINSERT}."},
            {"role": "user", "content": "Agent 4, refer to the TUI guide above. Conduct a risk assessment focusing on both the quantum and tackle data. Recommend any preventive measures that should be taken."},
            {"role": "user", "content": "Agent 5, refer to the TUI guide above. Evaluate the efficiency of the current strategies based on the data and the inputs from all previous agents. Suggest improvements that could be made."},
            {"role": "user", "content": "Agent 6, refer to the TUI guide above. Scrutinize the data for patterns or trends that could be beneficial for future games. Integrate the insights from all previous agents in your analysis."},
            {"role": "user", "content": f"Agent 7, refer to the TUI guide above. Based on the available NFL data files, which file should we focus on for in-depth analysis?"}
        ]

        # Make the GPT-4 Turbo API call
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )

        # Store the GPT-4 Turbo response
        agent_prompts.append(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"An error occurred during the learning round {round + 1}: {e}")

# Output the GPT-4 responses
try:
    for i, prompt in enumerate(agent_prompts):
        print(f"GPT-4 Response for Learning Round {i + 1}: {prompt}")
except Exception as e:
    print(f"Error printing GPT-4 responses: {e}")

# Save the results to a Markdown file
try:
    save_to_markdown_file(agent_prompts)
except Exception as e:
    print(f"Error saving to Markdown file: {e}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())    
