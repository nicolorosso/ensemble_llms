# ensemble_llms

## Ollama Set-up
[Ollama](https://ollama.ai/) installed on Windows. 
Alternatively, it can be installed on linux with:

```linux
curl -fsSL https://ollama.com/install.sh | sh



## Installation 
1. Install the haystack and ollama framework necessary to run the RAG pipeline.

```python
pip install haystack-ai

2. Pull the necessary LLMs model from Ollama
```python
ollama pull llama3.1:8b
ollama pull mistral
ollama pull phi3:medium
ollama pull gemma2

## Data Preparation

The following CSV files are the one used: 
- `scientific_papers.csv`: Contains scientific papers for context
- `antiscience-withlanguage-all-tweets.csv`: Contains tweets for classification

## Configuration

In `main.py`, you can adjust the `sample_size` variable to set the number of tweets to be classified.

## Running the Script

Execute the main script:

The script will process the specified number of tweets and output the classification results. Results will also be saved to `classification_results.csv`.

## Output

The script will print classification results for each tweet, including:
- Original tweet
- Preprocessed tweet
- Final classification
- Confidence score
- Individual model results

A summary of execution time will also be displayed. 

