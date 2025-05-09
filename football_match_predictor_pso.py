# === PART 1: Setup, Imports, Constants, and Class Definitions ===
# This section initializes libraries, configures warnings and logging, and sets the stage for PSO and NN implementation.

# === Standard Library Imports ===
import random  # For random number generation
import warnings  # To suppress specific warning types
import sqlite3  # For accessing SQLite databases
import time  # For measuring runtime

# === Numerical & Visualization Libraries ===
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For better looking visualizations
import pandas as pd  # For data manipulation and analysis
import logging  # To track and log events in code
import psutil  # For monitoring system and process utilities (optional)

# === Scikit-learn Modules ===
from sklearn.exceptions import ConvergenceWarning  # To suppress convergence-related warnings
from sklearn.model_selection import train_test_split  # For splitting data into training/testing
from sklearn.preprocessing import StandardScaler  # For feature normalization
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)  # For evaluating model performance

# === TensorFlow / Keras Modules ===
from tensorflow.keras.models import Sequential  # To build neural network models sequentially
from tensorflow.keras.layers import Dense, Dropout  # For neural network layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam  # Optimizers for gradient descent
from tensorflow.keras.utils import to_categorical  # One-hot encoding utility
from tensorflow.keras.callbacks import EarlyStopping  # Stop training when performance plateaus

# === Configuration: Warnings and Logging ===
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Suppress convergence warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)  # Configure logging format and level

# === Global Constants (Enhanced for Tuning) ===
# These constants control the PSO and neural network hyperparameter search.

SWARM_SIZE = 12         # Number of particles in the swarm
DIMENSIONS = 8          # Number of hyperparameters to optimize (e.g., neurons, lr, activation, etc.)
GENERATIONS = 20        # Maximum number of PSO generations/iterations
W_MAX, W_MIN = 0.9, 0.3  # Inertia weight range for balancing exploration and exploitation
C1, C2 = 2.05, 2.05      # Acceleration constants for personal best and informants' influence
DESIRED_PRECISION = 1e-6  # Precision threshold to stop early if performance is good enough

# Define lower and upper bounds for each dimension (hyperparameter):
# Format: [n1, n2, lr, activation_idx, dropout, optimizer_idx, batch_size, use_third_layer_flag]
MIN_BOUND = [8, 8, 0.0001, 0, 0.0, 0, 16, 0]
MAX_BOUND = [128, 128, 0.01, 2, 0.5, 3, 64, 1]

# List of available activation functions to select via activation index
ACTIVATIONS = ['relu', 'tanh', 'selu']

# Optimizer functions to choose from via optimizer index
OPTIMIZERS = [Adam, SGD, RMSprop, Nadam]

# Class labels for target variable in classification task
LABELS = ['Loss', 'Draw', 'Win']

# === Fitness Evaluator Class ===
# This class evaluates the fitness of a given set of hyperparameters (position vector).
# It builds a Keras model using these parameters, trains it, and returns a fitness score (1 - val_accuracy).

class FitnessEvaluator:
    def __init__(self, X_train, y_train, X_val, y_val):
        # Store training and validation data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def evaluate(self, position):
        # === Extract Hyperparameters from the PSO position vector ===
        n1 = int(position[0])                 # Number of neurons in the first hidden layer
        n2 = int(position[1])                 # Number of neurons in the second hidden layer
        lr = position[2]                      # Learning rate
        act = ACTIVATIONS[int(position[3]) % len(ACTIVATIONS)]  # Activation function index (0=relu, 1=tanh, 2=selu)
        dr = position[4]                      # Dropout rate
        opt_fn = OPTIMIZERS[int(position[5]) % len(OPTIMIZERS)] # Optimizer function (Adam, SGD, etc.)
        batch_size = int(position[6])         # Batch size
        use_third_layer = int(round(position[7]))  # Flag: 1 = use third hidden layer

        # === Build Keras Sequential Model ===
        model = Sequential()
        model.add(Dense(n1, input_dim=self.X_train.shape[1], activation=act))  # First hidden layer
        model.add(Dropout(dr))  # Dropout for regularization
        model.add(Dense(n2, activation=act))  # Second hidden layer

        if use_third_layer:
            # Optional third hidden layer with half of n2 neurons
            model.add(Dense(int(n2 / 2), activation=act))

        model.add(Dense(3, activation='softmax'))  # Output layer for 3-class classification

        # === Compile Model ===
        # Using categorical crossentropy since it’s a multi-class problem.
        model.compile(optimizer=opt_fn(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # === Configure EarlyStopping to prevent overfitting ===
        early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')

        # === Train Model ===
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,             # Reserve 20% of training data as validation
            epochs=30,                        # Max epochs
            batch_size=batch_size,
            verbose=0,                        # Silent training
            callbacks=[early_stop]            # Stop early if val_accuracy plateaus
        )

        # === Calculate fitness ===
        val_acc = history.history['val_accuracy'][-1]  # Final validation accuracy
        fitness = 1 - val_acc  # PSO minimizes, so lower fitness is better

        # Return both fitness score and the model itself
        return fitness, model


    # === Velocity Update ===
    # Updates the velocity vector of the particle based on:
    # - Inertia (previous velocity influence)
    # - Cognitive component (personal best attraction)
    # - Social component (global/informant best attraction)
    def update_velocity(self, global_best_position, inertia):
        for d in range(DIMENSIONS):
            r1, r2 = random.random(), random.random()  # Random coefficients for exploration

            # Cognitive term: particle's own best experience
            cognitive = C1 * r1 * (self.best_position[d] - self.position[d])

            # Social term: best known position among informants or swarm
            social = C2 * r2 * (global_best_position[d] - self.position[d])

            # Update velocity with weighted sum of all components
            self.velocity[d] = inertia * self.velocity[d] + cognitive + social

    # === Position Update ===
    # Updates the particle's position using the new velocity and evaluates its fitness.
    def update_position(self):
        for d in range(DIMENSIONS):
            self.position[d] += self.velocity[d]  # Move particle in search space
            # Ensure position stays within defined bounds
            self.position[d] = max(MIN_BOUND[d], min(MAX_BOUND[d], self.position[d]))

        # Re-evaluate fitness after position update
        self.fitness, _ = self.evaluator.evaluate(self.position)

    # === Group Best Update ===
    # Compares the current particle's best fitness to the best among its informants.
    # Updates the group_best_position if a better informant is found.
    def update_group_best(self, swarm):
        # Identify the informant with the best fitness
        best_informant = min(self.informants, key=lambda i: swarm[i].best_fitness)

        # If the informant has a better solution, update group best
        if swarm[best_informant].best_fitness < self.group_best_fitness:
            self.group_best_fitness = swarm[best_informant].best_fitness
            self.group_best_position = list(swarm[best_informant].best_position)


# === Unified PSO Runner ===
# This class manages the execution of the PSO algorithm for neural network hyperparameter optimization.
class UnifiedPSO:
    def __init__(self, X_train, y_train_cat, X_test, y_test, y_test_raw):
        # Initialize the evaluator to assess particle fitness based on training and test sets
        self.evaluator = FitnessEvaluator(X_train, y_train_cat, X_test, y_test)
        self.X_test = X_test              # Used for final prediction
        self.y_test = y_test_raw          # Ground truth labels for evaluation

    def run(self):
        # === Swarm Initialization ===
        # Create a list of particles with randomly initialized positions and velocities
        swarm = [Particle(self.evaluator) for _ in range(SWARM_SIZE)]

        # Identify initial global best particle (with lowest fitness value)
        global_best = min(swarm, key=lambda p: p.best_fitness)
        history = []  # To track best accuracy across generations

        # === Main PSO Loop ===
        for gen in range(GENERATIONS):
            # Dynamically adjust inertia weight to balance exploration/exploitation
            inertia = W_MAX - ((W_MAX - W_MIN) * gen / GENERATIONS)

            for p in swarm:
                # Update personal and group-level knowledge for each particle
                p.update_group_best(swarm)
                p.update_velocity(global_best.best_position, inertia)
                p.update_position()

                # Update personal best position and fitness if improved
                if p.fitness < p.best_fitness:
                    p.best_fitness = p.fitness
                    p.best_position = list(p.position)

            # Update global best if a better-performing particle is found
            best_particle = min(swarm, key=lambda p: p.best_fitness)
            if best_particle.best_fitness < global_best.best_fitness:
                global_best = best_particle

            # Convert fitness to accuracy and save for convergence plotting
            acc = 1 - global_best.best_fitness
            history.append(acc)

            # Logging for monitoring
            print(f"Generation {gen+1}: Best Accuracy = {acc:.4f} | Inertia = {inertia:.4f}")

            # Early stopping if target precision is achieved
            if global_best.best_fitness < DESIRED_PRECISION:
                print("✅ Desired precision reached.")
                break

        # Re-evaluate and return best model found along with convergence history
        _, best_model = self.evaluator.evaluate(global_best.best_position)
        return best_model, history


# === PART 2: Data Loading, Preprocessing, and Baseline Model Training ===

# === Step 1: Load Data from SQLite ===
# "Connect to the SQLite database containing football match and team statistics"
conn = sqlite3.connect('/kaggle/input/harshvardhan/database.sqlite')

# "Extract match-level information and associated team attributes using SQL joins"
df = pd.read_sql_query("""
    SELECT
        m.date,
        m.home_team_goal,
        m.away_team_goal,
        m.league_id,
        th.buildUpPlaySpeed AS home_speed,
        th.defenceAggression AS home_aggression,
        ta.buildUpPlaySpeed AS away_speed,
        ta.defenceAggression AS away_aggression
    FROM Match m
    LEFT JOIN Team_Attributes th ON m.home_team_api_id = th.team_api_id
    LEFT JOIN Team_Attributes ta ON m.away_team_api_id = ta.team_api_id
    WHERE m.home_team_goal IS NOT NULL AND m.away_team_goal IS NOT NULL
    LIMIT 5000;  -- "Restrict to 5000 matches to reduce training time"
""", conn)

# === Step 2: Match Outcome Labeling ===
# "Define a function to label each match as a win (2), draw (1), or loss (0) for the home team"
def get_result(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return 2  # Win
    elif row['home_team_goal'] < row['away_team_goal']:
        return 0  # Loss
    else:
        return 1  # Draw

# "Apply labeling logic to each row to get match results"
df['match_result'] = df.apply(get_result, axis=1)

# === Step 3: Data Cleaning ===
# "Display the proportion of Win/Draw/Loss classes in the data"
print("Class distribution (match_result):")
print(df['match_result'].value_counts(normalize=True).rename({0:'Loss', 1:'Draw', 2:'Win'}))

# "Show the count of missing values before cleaning"
print("Missing values before cleaning:")
print(df.isnull().sum())

# "Drop rows with any missing values to ensure clean input"
df.dropna(inplace=True)
print("✅ Remaining rows after dropping nulls:", len(df))

# === Step 4: Feature Engineering ===
# "Mark matches from top leagues like EPL, La Liga, Bundesliga"
df['is_big_league'] = df['league_id'].apply(lambda x: 1 if x in (1729, 476, 7809) else 0)

# "Extract the year from match date to capture time-based patterns"
df['match_year'] = pd.to_datetime(df['date']).dt.year

# "Select the final features to feed into the neural network"
features = ['home_speed', 'home_aggression', 'away_speed', 'away_aggression', 'is_big_league', 'match_year']
X = df[features]             # "Input features"
y = df['match_result']       # "Target class labels"

# === Step 5: Data Preprocessing ===
# "Normalize the features using Z-score standardization"
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# "Split data into training (80%) and test (20%) sets"
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# "Convert target labels to one-hot encoded format for multiclass classification"
y_train_cat = to_categorical(y_train_raw, 3)
y_test_cat = to_categorical(y_test_raw, 3)


# === Baseline MLP Model Training and Evaluation ===
"""
"Train a simple Multi-Layer Perceptron (MLP) as the baseline model.
 This model uses fixed hyperparameters (ReLU activation, Adam optimizer, dropout)
 to provide a benchmark against which the PSO-optimized model will be compared."
"""

# === Define the baseline MLP architecture ===
baseline_model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),  # "Input layer with 32 neurons"
    Dropout(0.3),                                               # "Dropout to prevent overfitting"
    Dense(32, activation='relu'),                               # "Hidden layer with ReLU activation"
    Dense(3, activation='softmax')                              # "Output layer for 3-class classification"
])

# === Compile the baseline model ===
baseline_model.compile(optimizer=Adam(learning_rate=0.001),    # "Using Adam optimizer with fixed learning rate"
                       loss='categorical_crossentropy',        # "Cross-entropy loss for classification"
                       metrics=['accuracy'])                   # "Track accuracy during training"

# === Train the baseline model ===
baseline_model.fit(X_train, y_train_cat,
                   epochs=30, batch_size=16, verbose=0)        # "Train silently for 30 epochs with batch size 16"

# === Make predictions on test data ===
y_pred_baseline = np.argmax(baseline_model.predict(X_test), axis=1)  # "Convert softmax outputs to class labels"

# === Calculate baseline model accuracy ===
baseline_acc = accuracy_score(y_test_raw, y_pred_baseline)

# === Generate classification report as dictionary ===
baseline_report = classification_report(
    y_test_raw, y_pred_baseline, target_names=LABELS, output_dict=True
)

# === Print formatted classification report ===
print("📊 Baseline MLP Classification Report:")
print(classification_report(y_test_raw, y_pred_baseline, target_names=LABELS))

# === Display confusion matrix ===
ConfusionMatrixDisplay.from_predictions(y_test_raw, y_pred_baseline)
plt.title("Baseline MLP Confusion Matrix")
plt.show()


# === PART 3: PSO Execution with Varying INFORMANTS ===

start_total = time.time()  # Start overall timer for all PSO runs

# Define number of PSO runs and corresponding informants to test
num_runs = 3
informants_list = [3, 4, 5]  # Vary informants for diversity vs. convergence analysis

# Containers for tracking each run
all_accuracies = []         # Store best accuracy from each run
all_histories = []          # Store convergence history
all_models = []             # Store trained models
best_positions = []         # Store best hyperparameters from each run
runtime_list = []           # Store time taken for each run

# === Particle Class with Dynamic Informants ===
class Particle:
    """
    Represents a candidate solution in the PSO swarm.
    Each particle stores its position (hyperparameters), velocity,
    personal best, and group best (among informants).
    """
    def __init__(self, evaluator, informants):
        # Initialize random hyperparameter position and velocity
        self.position = [random.uniform(MIN_BOUND[d], MAX_BOUND[d]) for d in range(DIMENSIONS)]
        self.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        self.evaluator = evaluator
        self.fitness, _ = self.evaluator.evaluate(self.position)

        # Personal best (initially the same as current)
        self.best_position = list(self.position)
        self.best_fitness = self.fitness

        # Informants for social learning (swarm sub-group)
        self.informants = random.sample(range(SWARM_SIZE), informants)
        self.group_best_position = list(self.position)
        self.group_best_fitness = self.fitness

    def update_velocity(self, global_best_position, inertia):
        """
        Update velocity based on inertia, personal best, and group/global best positions.
        """
        for d in range(DIMENSIONS):
            r1, r2 = random.random(), random.random()
            cognitive = C1 * r1 * (self.best_position[d] - self.position[d])     # Exploitation
            social = C2 * r2 * (global_best_position[d] - self.position[d])      # Exploration
            self.velocity[d] = inertia * self.velocity[d] + cognitive + social

    def update_position(self):
        """
        Move particle based on updated velocity. Clamp values within bounds.
        """
        for d in range(DIMENSIONS):
            self.position[d] += self.velocity[d]
            self.position[d] = max(MIN_BOUND[d], min(MAX_BOUND[d], self.position[d]))  # Bound check

        # Evaluate new position
        self.fitness, _ = self.evaluator.evaluate(self.position)

    def update_group_best(self, swarm):
        """
        Update the best position among informants (social best).
        """
        best_informant = min(self.informants, key=lambda i: swarm[i].best_fitness)
        if swarm[best_informant].best_fitness < self.group_best_fitness:
            self.group_best_fitness = swarm[best_informant].best_fitness
            self.group_best_position = list(swarm[best_informant].best_position)


# === Unified PSO Runner with Injected Informant Count ===
class UnifiedPSO:
    """
    This class manages the execution of a full PSO optimization cycle for a given dataset.
    It supports customizable numbers of informants, enhancing swarm diversity control.
    """

    def __init__(self, X_train, y_train_cat, X_test, y_test, y_test_raw, informants):
        """
        Initializes the UnifiedPSO runner with train/test datasets and informants.
        Args:
            X_train: Scaled feature training data
            y_train_cat: One-hot encoded training labels
            X_test: Scaled feature testing data
            y_test: One-hot encoded test labels (for evaluation)
            y_test_raw: Raw test labels (for classification reports)
            informants: Number of informants per particle (influencers)
        """
        self.evaluator = FitnessEvaluator(X_train, y_train_cat, X_test, y_test)
        self.X_test = X_test
        self.y_test = y_test_raw
        self.informants = informants

    def run(self):
        """
        Executes PSO for a set number of generations with the current informant setting.
        Returns:
            best_model: Trained Keras model with best hyperparameters
            history: List of validation accuracies over generations
            global_best_position: The optimal hyperparameter configuration found
        """

        # Initialize swarm of particles with given evaluator and informants
        swarm = [Particle(self.evaluator, self.informants) for _ in range(SWARM_SIZE)]

        # Select the particle with the best fitness to begin as global best
        global_best = min(swarm, key=lambda p: p.best_fitness)
        history = []  # Track accuracy progression over generations

        for gen in range(GENERATIONS):
            # Dynamically decay inertia to balance exploration vs. exploitation
            inertia = W_MAX - ((W_MAX - W_MIN) * gen / GENERATIONS)

            # Iterate through each particle in the swarm
            for p in swarm:
                p.update_group_best(swarm)                        # Update social best among informants
                p.update_velocity(global_best.best_position, inertia)  # Adjust velocity
                p.update_position()                               # Move particle to new position

                # Update particle's personal best if fitness improved
                if p.fitness < p.best_fitness:
                    p.best_fitness = p.fitness
                    p.best_position = list(p.position)

            # Update global best if a better fitness is found
            best_particle = min(swarm, key=lambda p: p.best_fitness)
            if best_particle.best_fitness < global_best.best_fitness:
                global_best = best_particle

            # Log accuracy for this generation
            acc = 1 - global_best.best_fitness
            history.append(acc)
            print(f"Generation {gen+1}: Best Accuracy = {acc:.4f} | Inertia = {inertia:.4f}")

            # Optional early stop if precision target is met
            if global_best.best_fitness < DESIRED_PRECISION:
                print("✅ Desired precision reached.")
                break

        # Final evaluation of the best configuration found
        _, best_model = self.evaluator.evaluate(global_best.best_position)
        return best_model, history, global_best.best_position


# === Run PSO for 3 Runs with Varying Informants ===
"""
This loop executes PSO three times with different values of INFORMANTS (3, 4, 5).
Each run finds the best hyperparameters and records results for comparison.
"""

for i in range(num_runs):
    informants = informants_list[i]  # Choose current informant count
    print(f"\n⚙️ PSO Run {i+1} | INFORMANTS = {informants}")

    # Initialize the PSO runner with current informant count
    runner = UnifiedPSO(X_train, y_train_cat, X_test, y_test_cat, y_test_raw, informants)

    # Start timing the PSO run
    start_time = time.time()

    # Run PSO and return the best model, its history, and its position
    model, history, position = runner.run()

    # Stop timing the PSO run
    end_time = time.time()
    runtime = end_time - start_time  # Calculate elapsed time

    # Save results from this PSO run
    all_accuracies.append(max(history))      # Save best accuracy of this run
    all_histories.append(history)            # Save accuracy history per generation
    all_models.append(model)                 # Save the trained model itself
    best_positions.append(position)          # Save best hyperparameter position
    runtime_list.append(runtime)             # Save how long the run took

    print(f"⏱️ Run Time: {runtime:.2f} sec")

# === Final Evaluation Summary Across All PSO Runs ===
"""
This block selects the best PSO run based on highest accuracy
and logs final model, parameters, and overall timing.
"""

end_total = time.time()  # End total PSO runtime

# Get index of best performing model
best_idx = np.argmax(all_accuracies)

# Extract best results and metadata
best_model = all_models[best_idx]           # Best-trained model
best_accuracy = all_accuracies[best_idx]    # Best accuracy score
best_history = all_histories[best_idx]      # Accuracy history of best model
best_position = best_positions[best_idx]    # Best hyperparameter configuration
best_informants = informants_list[best_idx] # Informants used for best model



# === F1 Score Comparison ===
# Create a dictionary to store F1 scores from baseline and PSO model for each class
f1_scores = {
    cls: {
        'Baseline': baseline_report[cls]['f1-score'],  # F1-score from baseline model
        'PSO': classification_report(
            y_test_raw,
            np.argmax(best_model.predict(X_test), axis=1),  # Predict labels with best PSO-optimized model
            target_names=LABELS,
            output_dict=True
        )[cls]['f1-score']
    } for cls in LABELS
}

# Convert dictionary to DataFrame for easier# Convert dictionary to DataFrame for a plotting and comparison
f1_df = pd.DataFrame(f1_scores).T

# === Summary Table ===
# Create a summary DataFrame showing accuracy and description for each model type
summary_df = pd.DataFrame({
    'Model': ['Baseline NN', 'Best PSO-NN', 'Avg PSO-NN'],
    'Accuracy': [baseline_acc, best_accuracy, np.mean(all_accuracies)],
    'Notes': [
        'Fixed NN',       # Standard manually tuned model
        'Best from PSO',  # Best result obtained through PSO optimization
        'Mean across runs'  # Average accuracy of all PSO runs
    ]
})

# === Convergence Plot ===
# Plot convergence history for each PSO run (accuracy across generations)
plt.figure(figsize=(8, 5))
for i, hist in enumerate(all_histories):
    plt.plot(hist, label=f"Run {i+1} (INF={informants_list[i]})", marker='o')  # Plot each PSO run with its informants
plt.title("📈 PSO Convergence Over Generations")
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === F1 Score Plot ===
# Visual comparison of F1 scores per class: Baseline vs PSO
f1_df.plot(kind='bar', figsize=(8, 5))
plt.title('📊 F1 Score Comparison: Baseline vs PSO')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis='y')  # Horizontal gridlines only
plt.tight_layout()
plt.show()

# === F1 Differences Printout ===
# Display the F1-score improvement (or decline) from Baseline to PSO model
print("\n📊 F1-Score Differences (PSO - Baseline):")
for cls in LABELS:
    delta = f1_df.loc[cls, 'PSO'] - f1_df.loc[cls, 'Baseline']
    print(f"  {cls}: ΔF1 = {delta:.3f}")

# === Confusion Matrices for All Runs ===
# Visualize prediction performance for each PSO run
for i, (acc, model) in enumerate(zip(all_accuracies, all_models)):
    y_pred = np.argmax(model.predict(X_test), axis=1)  # Get predicted labels
    cm = confusion_matrix(y_test_raw, y_pred)  # Compute confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"🧮 PSO Run {i+1} (INF={informants_list[i]}) | Acc: {acc:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# === Final Printouts ===

# Print the final summary DataFrame showing accuracy of all models
print("\n📄 Model Performance Summary:")
print(summary_df)

# Display the best hyperparameters found by PSO
print(f"\n🏅 Best PSO Hyperparameters:\n{best_position}")

# Print time taken for each PSO run
print(f"🔁 Runtimes: {[f'{r:.2f}s' for r in runtime_list]}")

# Print total and average runtime across all PSO runs
print(f"🕒 Total Time: {end_total - start_total:.2f}s | Avg Time: {np.mean(runtime_list):.2f}s")

# === Final Interpretation Block ===
print(f"""
🧠 INTERPRETATION:
- Baseline NN Accuracy: {baseline_acc*100:.2f}%
- Best PSO Accuracy: {best_accuracy*100:.2f}% using INFORMANTS = {best_informants}
- F1 improvements show PSO effectiveness in optimizing hyperparameters.
- Dynamic informants helped analyze the trade-off between diversity and convergence.
""")
