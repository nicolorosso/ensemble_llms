# Ensemble LLMs for data labelling
Ollama is used to install and manage the LLM models, while Haystack provides the framework for building the RAG pipeline. The two are separate but complementary: first, you'll install the LLM models via Ollama, and then use them through Haystack's pipeline structure. All required Python libraries are listed in the requirements.txt file, except for Ollama itself and the LLM models, which need to be installed separately as per the instructions below

## Ollama Set-up

[Ollama](https://ollama.ai/) can be installed on Windows. Alternatively, it can be installed on Linux with:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Haystack and Ollama set up
1. Install the haystack and Ollama-haystack framework necessary to run the RAG pipeline.

```python
pip install haystack-ai ollama-haystack
```
2. Pull the necessary LLMs model from Ollama
```python
ollama pull llama3.1:8b
ollama pull mistral
ollama pull phi3:medium
ollama pull gemma2
```
## Data Preparation

The following CSV files are the one used: 
- `abstracts_with_topics.csv`: Contains scientific papers for context
- `antiscience-withlanguage-all-tweets.csv`: Contains tweets for classification (this one has been sent via e-mail)

## Configuration

In `main.py`, you can adjust the `sample_size` variable to set the number of tweets to be classified. Ideally, this would set between 100000 and 200000

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

