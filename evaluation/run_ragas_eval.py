# This is the main evaluation script that implements the Ragas usage for analysis.

import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.evaluation import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
# Add parent directory to path to resolve retriever package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now import
from retriever.rag_chain import create_qa_chain


# Load API Key
load_dotenv()
ragas_app_token = os.getenv("RAGAS_APP_TOKEN")
if not ragas_app_token:
    raise ValueError("RAGAS_APP_TOKEN is not found. Make sure it's set in your .env file.")
print("RAGAS_APP_TOKEN Loaded Successfully.")  # Debugging step

# Load data
df = pd.read_csv('data/processed/eval_dataset.csv')

# Convert to HuggingFace Dataset format
dataset = Dataset.from_pandas(df)

# Initialize RAG chain
qa_chain = create_qa_chain()

# Generate answers & contexts
def generate(row):
    result = qa_chain.invoke({"query": row["question"]})
    row["response"] = result["result"]
    row["contexts"] = [doc.page_content for doc in result["source_documents"]]
    return row

hf_dataset = dataset.map(generate, desc="Generating responses & contexts")

# Evaluate using Ragas
results = evaluate(
    hf_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

results.upload()

print(results)
for i, score_dict in enumerate(results.scores):
    print(f"Sample {i+1}:")
    for metric_name, score in score_dict.items():
        print(f"{metric_name}: {score:.4f}")

os.makedirs("evaluation/evaluation_results", exist_ok=True)
results_df = pd.DataFrame(results.scores)
results_df.to_csv("evaluation/evaluation_results/results.csv", index=False)

# Average scores per metric
avg_scores = results_df.mean()

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=avg_scores.index, y=avg_scores.values, palette='coolwarm')
plt.title("Average Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("evaluation/evaluation_results/average_scores.png")
plt.show()

print("\nAverage Metric Scores:")
print(avg_scores)
