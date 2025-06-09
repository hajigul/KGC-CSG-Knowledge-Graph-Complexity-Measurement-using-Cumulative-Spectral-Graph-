# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from spectral_metric.estimator import CumulativeGradientEstimator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

# Path to the dataset
data_path = r'D:\dataset_complexity\data\Nations'  # Adjust the path to your dataset location


# Step 1: Load the Train, Valid, and Test Triplets
train_file = os.path.join(data_path, 'train.txt')  # Adjust the file name if needed
valid_file = os.path.join(data_path, 'valid.txt')
test_file = os.path.join(data_path, 'test.txt')

# Load triplets into dataframes (assuming tab-separated text files, adjust delimiter if needed)
train_data = pd.read_csv(train_file, sep="\t", header=None, names=['head', 'relation', 'tail'])
valid_data = pd.read_csv(valid_file, sep="\t", header=None, names=['head', 'relation', 'tail'])
test_data = pd.read_csv(test_file, sep="\t", header=None, names=['head', 'relation', 'tail'])

# Step 2: Combine all triplets into a single dataset
all_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)

# Step 3: Organize the Data by Tail Entities (Classes)
tail_classes = {}
for idx, row in all_data.iterrows():  # Iterate over each row in the dataset
    tail = row['tail']  # The 'tail' is the entity we want to group by
    composite_vector = np.array([row['head'], row['relation']])  # Create a composite vector of head and relation

    if tail not in tail_classes:  # If the tail entity isn't in the dictionary, add it
        tail_classes[tail] = []
    tail_classes[tail].append(composite_vector)  # Append the composite vector to the corresponding tail class

# Step 4: Map Tail Entities (Strings) to Integer Labels
tail_entity_to_label = {tail: idx for idx, tail in enumerate(tail_classes.keys())}  # Map each tail entity to an integer
print(f"Tail Entity to Label Mapping: {tail_entity_to_label}")

# Map the tail entities in the dataset to integer labels
all_data['tail_label'] = all_data['tail'].map(tail_entity_to_label)  # Create a new column for the integer labels of tail entities

# Step 5: Generate Embeddings for Head and Relation using BERT and Sentence-BERT
# Using BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Initialize BERT tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')  # Initialize BERT model

# Using Sentence-BERT
sentence_embedder = SentenceTransformer('all-MiniLM-L12-v2')  # Initialize Sentence-BERT model

# Function to get BERT embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean over the sequence tokens
    return embeddings

# Function to get Sentence-BERT embeddings
def get_sentence_bert_embeddings(texts):
    return sentence_embedder.encode(texts, convert_to_tensor=True)

# Generate embeddings for head and relation entities using both BERT and Sentence-BERT
bert_head_embeddings = get_bert_embeddings(all_data['head'].tolist())
bert_relation_embeddings = get_bert_embeddings(all_data['relation'].tolist())
sentence_bert_head_embeddings = get_sentence_bert_embeddings(all_data['head'].tolist())
sentence_bert_relation_embeddings = get_sentence_bert_embeddings(all_data['relation'].tolist())

# Concatenate the embeddings to create composite embeddings for both BERT and Sentence-BERT
bert_composite_embeddings = torch.cat((bert_head_embeddings, bert_relation_embeddings), dim=1)
sentence_bert_composite_embeddings = torch.cat((sentence_bert_head_embeddings, sentence_bert_relation_embeddings), dim=1)

# Step 6: Group by Tail and Create the Data Arrays for Spectral Metric
all_embeddings = []  # List to store all the composite embeddings
all_labels = []  # List to store the corresponding labels (tail entity labels)

# For both BERT and Sentence-BERT
for idx, row in all_data.iterrows():  # Iterate through all rows
    bert_composite_vector = bert_composite_embeddings[idx].numpy()  # Get the composite embedding for the current row (BERT)
    sentence_bert_composite_vector = sentence_bert_composite_embeddings[idx].numpy()  # Get the composite embedding for Sentence-BERT
    tail_label = row['tail_label']  # Get the corresponding tail label
    all_embeddings.append(bert_composite_vector)  # Append the composite embedding (BERT)
    all_labels.append(tail_label)  # Append the label

# Convert the lists to numpy arrays for BERT
X_bert = np.array(all_embeddings)  # Features (composite embeddings) for BERT
y_bert = np.array(all_labels)  # Labels (tail entity labels) for BERT

# Reset the embeddings for the second run with Sentence-BERT
all_embeddings = []  # List to store all the composite embeddings for Sentence-BERT
for idx, row in all_data.iterrows():
    sentence_bert_composite_vector = sentence_bert_composite_embeddings[idx].numpy()  # Get the composite embedding for Sentence-BERT
    all_embeddings.append(sentence_bert_composite_vector)  # Append the composite embedding (Sentence-BERT)

# Convert the lists to numpy arrays for Sentence-BERT
X_sentence_bert = np.array(all_embeddings)  # Features (composite embeddings) for Sentence-BERT
y_sentence_bert = np.array(all_labels)  # Labels (tail entity labels) for Sentence-BERT

# Step 7: Test different combinations of k (nearest neighbors) and M_sample (sampling size)
k_values = [3, 20, 30, 50, 100, 150]  # List of different k values
M_sample_values = [30, 50, 120, 200, 300]  # List of different M_sample values

# Loop through all combinations of k and M_sample
for k_value in k_values:
    for M_sample_value in M_sample_values:
        print(f"Testing with k={k_value} and M_sample={M_sample_value}")

        # Test with BERT embeddings
        estimator = CumulativeGradientEstimator(M_sample=M_sample_value, k_nearest=k_value, distance="cosine")
        estimator.fit(data=X_bert, target=y_bert)
        csg_bert = estimator.csg
        print(f"Complexity Score (CSG) for BERT embeddings with k={k_value}, M_sample={M_sample_value}: {csg_bert}")

        # Visualize the Similarity Matrix (W Matrix) for BERT
        #plt.figure(figsize=(10,5))  # Set the figure size
        #sns.heatmap(estimator.W)  # Plot the heatmap of the similarity matrix
        #plt.title(f"Similarity between Tail Classes (BERT) k={k_value}, M_sample={M_sample_value}")
        #plt.show()  # Display the plot

        # Compute Entropy (Most Confused Class) for BERT
        #entropy_per_class_bert = entropy(estimator.W / estimator.W.sum(-1)[:, None], axis=-1)
        #most_confused_class_bert = np.argmax(entropy_per_class_bert)
        #print(f"Class with the highest entropy (most confused) for BERT: {most_confused_class_bert}")

        # Test with Sentence-BERT embeddings
        estimator.fit(data=X_sentence_bert, target=y_sentence_bert)
        csg_sentence_bert = estimator.csg
        print(f"Complexity Score (CSG) for Sentence-BERT embeddings with k={k_value}, M_sample={M_sample_value}: {csg_sentence_bert}")

        # Visualize the Similarity Matrix (W Matrix) for Sentence-BERT
        #plt.figure(figsize=(10,5))  # Set the figure size
        #sns.heatmap(estimator.W)  # Plot the heatmap of the similarity matrix
        #plt.title(f"Similarity between Tail Classes (Sentence-BERT) k={k_value}, M_sample={M_sample_value}")
        #plt.show()  # Display the plot

        # Compute Entropy (Most Confused Class) for Sentence-BERT
        #entropy_per_class_sentence_bert = entropy(estimator.W / estimator.W.sum(-1)[:, None], axis=-1)
        #most_confused_class_sentence_bert = np.argmax(entropy_per_class_sentence_bert)
        #print(f"Class with the highest entropy (most confused) for Sentence-BERT: {most_confused_class_sentence_bert}")