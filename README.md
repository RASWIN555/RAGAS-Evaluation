# Ragas Evaluation Project

## Overview
The Ragas Evaluation project is designed to facilitate the evaluation of Ragas using advanced retrieval and generation techniques. This project leverages various tools and libraries to preprocess data, retrieve relevant information, and generate insightful evaluations.

## Project Structure
The project is organized into several directories and files, each serving a specific purpose:

- **data/**: Contains raw and processed data.
  - **raw/**: This directory holds the original documents and datasets.
  - **processed/**: This directory contains preprocessed or chunked data ready for analysis.

- **retriever/**: Contains scripts for setting up the retrieval system.
  - **index_setup.py**: Script responsible for creating a Pinecone index and adding embeddings.
  - **retriever_config.py**: Configuration settings for Pinecone and the embedding model.

- **generator/**: Contains scripts for generating answers.
  - **qa_chain.py**: Sets up the LangChain RetrievalQA for generating answers based on retrieved data.

- **evaluation/**: Contains scripts and results for the evaluation process.
  - **run_ragas_eval.py**: Main evaluation script implementing Ragas usage for analysis.
  - **metrics_config.py**: Configures the metrics used in the Ragas evaluation.
  - **evaluation_results/**: Directory for storing evaluation results, including JSON, CSV files, and visualizations.

- **notebooks/**: Contains Jupyter notebooks for demonstrations.
  - **ragas_evaluation_demo.ipynb**: Optional notebook providing a demonstration of the Ragas evaluation process.

- **.env**: File for securely storing API keys for services like OpenAI and Pinecone.

- **requirements.txt**: Lists the required Python packages for the project.

- **README.md**: Project explanation and documentation.

- **main.py**: High-level script that loads the retriever, runs the generator, and performs the evaluation.

## Installation
To set up the project, clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage
1. Configure your API keys in the `.env` file.
2. Run the main script to initiate the evaluation process:

```bash
python main.py
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.