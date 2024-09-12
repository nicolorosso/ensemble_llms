import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
    FastembedDocumentEmbedder,
    FastembedSparseTextEmbedder,
    FastembedSparseDocumentEmbedder
)
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
import asyncio
import logging
from collections import Counter
import time
import re
import json
import os
from tqdm.asyncio import tqdm_asyncio
import aiofiles
import aiocsv

# Set environment variables for Ollama
os.environ['OLLAMA_NUM_PARALLEL'] = '4'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '4'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Few-shot examples 
few_shot_examples = [
    # Proscience tweets
    {"tweet": "Just got my COVID vaccine! 💉 Feeling great and doing my part to protect the community. #VaccinesWork #ScienceSaves", "classification": "proscience"},
    {"tweet": "New study shows rapid ice melt in Antarctica. We need urgent action on climate change! 🌎🆘 #ClimateEmergency #ActNow", "classification": "proscience"},
    {"tweet": "Excited to attend the @CERN open day! Science is awesome 🧪🔬 #ParticlePhysics #STEM", "classification": "proscience"},
    {"tweet": "Just watched a great documentary on space exploration. The universe is mind-blowing! 🚀🌌 #SpaceLovers #Astronomy", "classification": "proscience"},
    {"tweet": "Genetic engineering could help cure inherited diseases. The future of medicine is here! 🧬🏥 #CRISPR #GeneTherapy", "classification": "proscience"},
    {"tweet": "Solar power installations hit record high last year. Renewable energy is the future! ☀️🔋 #CleanEnergy #Sustainability", "classification": "proscience"},
    {"tweet": "New AI algorithm detects early-stage cancer with 95% accuracy. Technology saving lives! 🖥️❤️ #AIinHealthcare #MedicalAdvances", "classification": "proscience"},
    {"tweet": "Just finished reading @neiltyson's latest book. Science communication at its best! 📚🌟 #Astrophysics #ScienceForAll", "classification": "proscience"},
    {"tweet": "Breakthrough in quantum computing! 128-qubit processor achieved. Moore's Law is not dead yet 💻🔬 #QuantumSupremacy #TechInnovation", "classification": "proscience"},
    {"tweet": "Marine biologists discover new deep-sea species. Our oceans still hold so many secrets! 🐠🌊 #MarineLife #Biodiversity", "classification": "proscience"},
    {"tweet": "First successful pig-to-human heart transplant completed. Medical science is pushing boundaries! 🐷❤️ #Xenotransplantation #MedicalBreakthrough", "classification": "proscience"},
    {"tweet": "New eco-friendly plastic alternative made from seaweed. Science tackling pollution! 🌿🌊 #SustainableInnovation #PlasticFree", "classification": "proscience"},
    {"tweet": "Stem cell therapy shows promise in treating Parkinson's disease. Hope for millions! 🧠💉 #StemCells #NeurologyBreakthrough", "classification": "proscience"},
    {"tweet": "Machine learning algorithm predicts earthquake aftershocks with 85% accuracy. Tech saving lives! 🏙️🌋 #GeoscienceTech #DisasterPreparedness", "classification": "proscience"},
    {"tweet": "First 3D-printed house completed in 24 hours. The future of construction is here! 🏠🖨️ #3DPrinting #SustainableHousing", "classification": "proscience"},

    # Antiscience tweets
    {"tweet": "Don't believe the hype! 5G is a dangerous experiment on humanity. RT to spread awareness! #5GDangers #HealthRisk", "classification": "antiscience"},
    {"tweet": "Wake up, sheeple! The Earth is flat and NASA is part of the cover-up. Do your own research! 🌍🤔 #FlatEarth #NASAlies", "classification": "antiscience"},
    {"tweet": "GMOs are poisoning our food supply! Say NO to frankenfood and go organic! 🚫🌽 #BanGMOs #OrganicLiving", "classification": "antiscience"},
    {"tweet": "Chemtrails are real! The government is spraying us with toxic chemicals. Look up! ☁️☠️ #ChemtrailsTruth #ConspiracyFact", "classification": "antiscience"},
    {"tweet": "Vaccines cause autism! Big Pharma is hiding the truth from us all. Protect your kids! 💉🚫 #VaccineInjury #AutismAwareness", "classification": "antiscience"},
    {"tweet": "Climate change is a hoax invented by the Chinese to make U.S. manufacturing non-competitive. Wake up America! 🏭🇺🇸 #ClimateHoax #MAGA", "classification": "antiscience"},
    {"tweet": "Evolution is just a theory, not a fact. Teach the controversy in schools! 🐒❓ #IntelligentDesign #CreationScience", "classification": "antiscience"},
    {"tweet": "WiFi signals are frying our brains! Go back to wired connections to protect your health. 📡🧠 #EMFdangers #DigitalDetox", "classification": "antiscience"},
    {"tweet": "Homeopathy cured my chronic illness when modern medicine failed. Natural healing is the way! 🌿💊 #AlternativeMedicine #Homeopathy", "classification": "antiscience"},
    {"tweet": "The moon landing was faked in a Hollywood studio. It's all propaganda! 🎬🌙 #MoonHoax #FakeNews", "classification": "antiscience"},
    {"tweet": "Fluoride in water is a mind control agent used by the government. Filter your water now! 💧🧠 #FluorideToxicity #WakeUp", "classification": "antiscience"},
    {"tweet": "Ancient aliens built the pyramids. Human civilization is a lie! 👽🏺 #AncientAliens #HiddenHistory", "classification": "antiscience"},
    {"tweet": "Big Pharma is suppressing the cure for cancer to profit from treatments. Follow the money! 💰💊 #CancerConspiracy #BigPharmaLies", "classification": "antiscience"},
    {"tweet": "Crystals have healing powers that science can't explain. Harness the energy! 💎✨ #CrystalHealing #AlternativeTherapy", "classification": "antiscience"},
    {"tweet": "Astrology is more accurate than psychology. Your zodiac sign determines your personality! ♈️🔮 #Astrology #CosmicTruth", "classification": "antiscience"},

    # Neutral tweets
    {"tweet": "Caught this amazing sunset at the beach today. Nature's daily show never disappoints! 😍🌅 #NaturePhotography #BeachLife", "classification": "neutral"},
    {"tweet": "Coffee or tea? What's your morning fuel? ☕️🍵 #MorningRoutine #CaffeineFix", "classification": "neutral"},
    {"tweet": "Just adopted the cutest puppy from the local shelter. Meet Max! 🐶❤️ #AdoptDontShop #PuppyLove", "classification": "neutral"},
    {"tweet": "Who else is excited for the new season of Stranger Things? Binge-watching weekend ahead! 📺🍿 #StrangerThings #NetflixAndChill", "classification": "neutral"},
    {"tweet": "Meal prep Sunday in full swing. Eating healthy all week! 🥗🍽️ #MealPrep #HealthyEating", "classification": "neutral"},
    {"tweet": "Traffic is insane this morning. Wish I could teleport to work! 🚗😫 #MondayMorning #CommuterLife", "classification": "neutral"},
    {"tweet": "Just PR'd on my 5k run! Hard work pays off. 🏃‍♂️💨 #RunningCommunity #PersonalBest", "classification": "neutral"},
    {"tweet": "New phone, who dis? 📱😎 Finally upgraded after 3 years! #TechUpgrade #NewGadget", "classification": "neutral"},
    {"tweet": "Happy birthday to my best friend! 20 years of friendship and counting. 🎂🥳 #BirthdayLove #BestieGoals", "classification": "neutral"},
    {"tweet": "Rediscovered my old vinyl collection. The sound quality is unmatched! 🎵📀 #VinylRevival #MusicLover", "classification": "neutral"},
    {"tweet": "First day at my new job! Nervous but excited for this new chapter. 💼✨ #NewBeginnings #CareerGoals", "classification": "neutral"},
    {"tweet": "Just booked tickets for my dream vacation. Countdown to paradise begins! ✈️🏝️ #TravelPlans #Wanderlust", "classification": "neutral"},
    {"tweet": "Trying out a new recipe for dinner tonight. Fingers crossed it turns out edible! 👨‍🍳🤞 #HomeCooking #FoodieAdventures", "classification": "neutral"},
    {"tweet": "Binge-watched the entire new season in one day. No regrets! 📺😴 #SeriesMarathon #NoSpoilers", "classification": "neutral"},
    {"tweet": "Just finished my first marathon! Exhausted but so proud. 🏅🏃‍♀️ #MarathonFinisher #RunningAchievement", "classification": "neutral"}
]

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 1 else ""

async def load_sample_tweets(file_path: str, sample_size: int) -> List[Tuple[int, str, str]]:
    tweets = []
    total_rows = 0
    english_tweets = 0
    
    async with aiofiles.open(file_path, mode='r', encoding='utf-8', newline='') as afp:
        async_reader = aiocsv.AsyncReader(afp)
        header = await async_reader.__anext__()
        text_index = header.index('text')
        lang_index = header.index('Language')
        
        async for row in async_reader:
            total_rows += 1
            if row[lang_index] == 'en':
                english_tweets += 1
                preprocessed_text = preprocess_text(row[text_index])
                if preprocessed_text:
                    tweets.append((total_rows, row[text_index], preprocessed_text))
                    if len(tweets) >= sample_size:
                        break
    
    logger.info(f"Processed {total_rows} rows, found {english_tweets} English tweets, sampled {len(tweets)} tweets")
    return tweets

def initialize_document_store() -> QdrantDocumentStore:
    logger.info("Initializing Qdrant Document Store...")
    document_store = QdrantDocumentStore(
        ":memory:",
        recreate_index=True,
        use_sparse_embeddings=True,
        embedding_dim=384
    )
    logger.info("Document Store initialized.")
    return document_store

async def load_and_index_documents(file_path: str, document_store: QdrantDocumentStore):
    logger.info("Loading and indexing documents...")
    df = pd.read_csv(file_path)
    documents = [Document(content=row['abstract'], meta={'topic': row['topic']}) for _, row in df.iterrows()]

    indexing_pipeline = Pipeline()
    sparse_doc_embedder = FastembedSparseDocumentEmbedder(model="prithvida/Splade_PP_en_v1")
    dense_doc_embedder = FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    indexing_pipeline.add_component("sparse_doc_embedder", sparse_doc_embedder)
    indexing_pipeline.add_component("dense_doc_embedder", dense_doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("sparse_doc_embedder", "dense_doc_embedder")
    indexing_pipeline.connect("dense_doc_embedder", "writer")

    await asyncio.to_thread(indexing_pipeline.run, {"sparse_doc_embedder": {"documents": documents}})
    logger.info(f"Indexed {len(documents)} documents")

def create_embedding_pipeline() -> Pipeline:
    embedding_pipeline = Pipeline()
    sparse_text_embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")
    dense_text_embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", prefix="Represent this sentence for searching relevant passages: ")
    
    embedding_pipeline.add_component("sparse_text_embedder", sparse_text_embedder)
    embedding_pipeline.add_component("dense_text_embedder", dense_text_embedder)
    
    return embedding_pipeline

def create_hybrid_rag_pipeline(model_name: str, few_shot_examples: list, document_store: QdrantDocumentStore) -> Pipeline:
    few_shot_prompt = "\n".join([f"Ex{i+1}: {ex['tweet']} -> {ex['classification']}" for i, ex in enumerate(few_shot_examples)])

    prompt_template = f"""
Here are some examples:
{{few_shot_prompt}}

Task: Classify the provided tweet as 'antiscience', 'proscience', or 'neutral' based on its content and the relevant scientific context. Provide ONLY the classification label without any explanation.

Classification Guidelines:
1. Antiscience: Tweet actively contradicts established scientific consensus, promotes misinformation, or expresses skepticism towards credible scientific sources.
2. Proscience: Tweet supports or promotes scientific understanding, cites credible sources, or expresses enthusiasm for scientific advancements.
3. Neutral: Tweet is unrelated to scientific matters OR discusses science without taking a clear stance for or against scientific consensus.

Important: If the tweet does not explicitly relate to scientific topics or doesn't express a clear stance on scientific matters, classify it as 'neutral'.

Tweet to classify: {{{{tweet}}}}

Relevant scientific context:
{{% for doc in documents %}}
{{{{ doc.content }}}}
{{% endfor %}}

Your response must be ONLY one of these three words: 'antiscience', 'proscience', or 'neutral'.
"""

    retriever = QdrantHybridRetriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OllamaGenerator(
        model=model_name,
        url="http://localhost:11434/api/generate",
        generation_kwargs={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 50
        },
        timeout=120000
    )

    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component("retriever", retriever)
    hybrid_pipeline.add_component("prompt_builder", prompt_builder)
    hybrid_pipeline.add_component("llm", llm)

    hybrid_pipeline.connect("retriever", "prompt_builder.documents")
    hybrid_pipeline.connect("prompt_builder", "llm")
    
    return hybrid_pipeline

async def classify_tweets_with_model(pipeline: Pipeline, tweets: List[Dict], model_name: str, max_retries: int = 3) -> List[Dict]:
    classified_tweets = []
    for tweet in tweets:
        for attempt in range(max_retries):
            try:
                inputs = {
                    "retriever": {
                        "query_sparse_embedding": tweet['sparse_embedding'],
                        "query_embedding": tweet['dense_embedding']
                    },
                    "prompt_builder": {
                        "tweet": tweet['preprocessed']
                    }
                }
                result = await asyncio.to_thread(pipeline.run, inputs)
                classification = result["llm"]["replies"][0].strip().lower()
                tweet['classifications'][model_name] = classification
                logger.info(f"Tweet {tweet['index']} classified as '{classification}' by {model_name}")
                classified_tweets.append(tweet)
                break
            except Exception as e:
                logger.error(f"Error classifying tweet {tweet['index']} with {model_name}, attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    tweet['classifications'][model_name] = "classification_failed"
                    logger.warning(f"Classification failed for tweet {tweet['index']} with {model_name} after {max_retries} attempts")
                    classified_tweets.append(tweet)
    return classified_tweets

async def process_tweets(pipelines: Dict[str, Pipeline], embedding_pipeline: Pipeline, tweets: List[Tuple[int, str, str]], batch_size: int):
    all_results, processed_count = await load_checkpoint()
    tweets_to_process = tweets[processed_count:]
    
    async def process_tweet(tweet):
        index, original, preprocessed = tweet
        embedding_result = await asyncio.to_thread(embedding_pipeline.run, {
            "sparse_text_embedder": {"text": preprocessed}, 
            "dense_text_embedder": {"text": preprocessed}
        })
        logger.info(f"Embedded tweet {index}")
        return {
            "index": index,
            "tweet": original,
            "preprocessed": preprocessed,
            "sparse_embedding": embedding_result["sparse_text_embedder"]["sparse_embedding"],
            "dense_embedding": embedding_result["dense_text_embedder"]["embedding"],
            "classifications": {}  # Initialize classifications as an empty dict
        }
        
    total_tweets = len(tweets_to_process)
    overall_progress = tqdm_asyncio(total=total_tweets * len(pipelines), desc="Overall progress", position=0)
    embedding_progress = tqdm_asyncio(total=total_tweets, desc="Embedding tweets", position=1)
    
    # Embed all tweets first
    embedded_tweets = []
    for i in range(0, total_tweets, batch_size):
        batch = tweets_to_process[i:i+batch_size]
        batch_embedded = await asyncio.gather(*[process_tweet(tweet) for tweet in batch])
        embedded_tweets.extend(batch_embedded)
        embedding_progress.update(len(batch))
    
    embedding_progress.close()
    
    # Process with each model sequentially
    for model_name, pipeline in pipelines.items():
        logger.info(f"Processing tweets with model: {model_name}")
        classification_progress = tqdm_asyncio(total=total_tweets, desc=f"Classifying with {model_name}", position=1)
        
        for i in range(0, len(embedded_tweets), batch_size):
            batch = embedded_tweets[i:i+batch_size]
            classified_batch = await classify_tweets_with_model(pipeline, batch, model_name)
            classification_progress.update(len(batch))
            overall_progress.update(len(batch))
        
        classification_progress.close()
    
    # Aggregate classifications after all models have processed
    all_results.extend([aggregate_classifications(tweet) for tweet in embedded_tweets])
    overall_progress.close()
    
    # Save final checkpoint
    final_checkpoint_results = [{k: v for k, v in result.items() if k not in ['sparse_embedding', 'dense_embedding']} 
                                for result in all_results]
    await save_checkpoint(final_checkpoint_results, processed_count + len(final_checkpoint_results))
    logger.info(f"Final checkpoint saved: {len(final_checkpoint_results)}/{total_tweets} tweets processed")
    
    return all_results

def aggregate_classifications(tweet: Dict) -> Dict:
    classifications = tweet['classifications']
    counter = Counter(classifications.values())
    most_common_classification = counter.most_common(1)[0][0]
    confidence = counter[most_common_classification] / len(classifications)

    return {
        "index": tweet["index"],
        "tweet": tweet["tweet"],
        "preprocessed": tweet["preprocessed"],
        "final_classification": most_common_classification,
        "confidence": confidence,
        "individual_results": classifications,
        "sparse_embedding": tweet["sparse_embedding"],
        "dense_embedding": tweet["dense_embedding"]
    }

async def save_checkpoint(results: List[Dict], processed_count: int, checkpoint_file: str = "classification_checkpoint.json"):
    checkpoint_data = {
        "processed_count": processed_count,
        "total_tweets": len(results),
        "models_processed": list(set(model for result in results for model in result.get('individual_results', {}).keys())),
        "results": results  
    }
    
    async with aiofiles.open(checkpoint_file, 'w') as f:
        await f.write(json.dumps(checkpoint_data))
    
    # Also save the current results to a CSV file
    results_df = pd.DataFrame(results)
    await asyncio.to_thread(results_df.to_csv, f"classification_results_checkpoint_{processed_count}.csv", index=False)
    
    logger.info(f"Checkpoint saved: {processed_count}/{len(results)} tweets processed, Models: {checkpoint_data['models_processed']}")

async def load_checkpoint(checkpoint_file: str = "classification_checkpoint.json") -> Tuple[List[Dict], int]:
    if os.path.exists(checkpoint_file):
        try:
            async with aiofiles.open(checkpoint_file, 'r') as f:
                content = await f.read()
                
            if not content.strip():
                logger.warning(f"Checkpoint file {checkpoint_file} is empty. Starting from the beginning.")
                return [], 0
            
            checkpoint_data = json.loads(content)
            
            logger.info(f"Loaded checkpoint: {checkpoint_data['processed_count']}/{checkpoint_data['total_tweets']} tweets processed, Models: {checkpoint_data['models_processed']}")
            return checkpoint_data["results"], checkpoint_data["processed_count"]
        
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from checkpoint file {checkpoint_file}: {str(e)}. Starting from the beginning.")
        except KeyError as e:
            logger.error(f"Checkpoint file {checkpoint_file} is missing expected data: {str(e)}. Starting from the beginning.")
        except Exception as e:
            logger.error(f"Unexpected error loading checkpoint from {checkpoint_file}: {str(e)}. Starting from the beginning.")
    else:
        logger.info(f"Checkpoint file {checkpoint_file} not found. Starting from the beginning.")
    
    return [], 0

async def run_pipeline():
    start_time = time.time()
    
    # Initialize document store and load documents
    document_store = initialize_document_store()
    scientific_papers_path = r"C:\Users\nrosso\Documents\thesis_project\notebooks\experiments\scientific_papers.csv"
    await load_and_index_documents(scientific_papers_path, document_store)

    # Create embedding pipeline
    embedding_pipeline = create_embedding_pipeline()

    # Create classification pipelines
    pipelines = {
        model_name: create_hybrid_rag_pipeline(model_name, few_shot_examples, document_store)
        for model_name in ["llama3.1:8b", 'gemma2']
    }

    # Load tweets
    tweets_file_path = r'C:\Users\nrosso\Documents\thesis_project\data\raw\antiscience-withlanguage-all-tweets.csv'
    sample_size = 1
    batch_size = 1
    
    all_tweets = await load_sample_tweets(tweets_file_path, sample_size)
    logger.info(f"Loaded {len(all_tweets)} sample tweets in {time.time() - start_time:.2f} seconds")

    try:
        # Process tweets
        all_results = await process_tweets(pipelines, embedding_pipeline, all_tweets, batch_size)
        
        # Print summary
        print("\nClassification Summary:")
        classification_counts = Counter(result['final_classification'] for result in all_results)
        for classification, count in classification_counts.items():
            print(f"{classification}: {count} tweets ({count/len(all_results)*100:.2f}%)")

        total_time = time.time() - start_time
        print(f"\nTotal Execution Time: {total_time:.2f} seconds")
        print(f"Average Time per Tweet: {total_time / len(all_tweets):.2f} seconds")

        # Save final results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("classification_results_final.csv", index=False)
        logger.info("Final results saved to classification_results_final.csv")
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted. Saving current progress...")
        all_results, processed_count = await load_checkpoint()
        await save_checkpoint(all_results, processed_count)
        logger.info("Progress saved. You can resume from this point later.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        all_results, processed_count = await load_checkpoint()
        await save_checkpoint(all_results, processed_count)
        logger.info("Progress saved despite error. You can review and resume from this point later.")
        raise
    finally:
        # Ensure we always try to save our progress, even if an exception occurs
        try:
            final_results, final_count = await load_checkpoint()
            await save_checkpoint(final_results, final_count)
            logger.info(f"Final save completed. Processed {final_count} tweets.")
        except Exception as e:
            logger.error(f"Error during final save: {str(e)}")

def main():
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()