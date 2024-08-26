import nest_asyncio
import asyncio
import logging
from collections import Counter
import time
import pandas as pd
from typing import Dict, List
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
import os
from functools import lru_cache
import re
from concurrent.futures import ThreadPoolExecutor

os.environ['OLLAMA_NUM_PARALLEL'] = '3'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '3'

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    return text if len(text) >= 10 else ""

def load_sample_tweets(file_path: str, sample_size: int) -> List[str]:
    chunk_size = 1000
    sampled_tweets = []
    for chunk in pd.read_csv(file_path, usecols=['text', 'Language'], chunksize=chunk_size):
        # Filter for English tweets
        english_tweets = chunk[chunk['Language'] == 'en']['text']
        sampled_tweets.extend(english_tweets.sample(min(sample_size, len(english_tweets))).tolist())
        if len(sampled_tweets) >= sample_size:
            break
    return sampled_tweets[:sample_size]

@lru_cache(maxsize=1)
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

def load_and_index_documents(file_path: str, document_store: QdrantDocumentStore):
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

    indexing_pipeline.run({"sparse_doc_embedder": {"documents": documents}})
    logger.info(f"Indexed {len(documents)} documents")

@lru_cache(maxsize=1)
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
{few_shot_prompt}

Task: Analyze the provided tweet in the context of the relevant scientific background information and the few shot prompt that you recieved as an example. 
Your goal is to classify the tweet strictly as 'antiscience', 'proscience', or 'neutral'. Provide no additional text in your classification.

Instructions:

- Accuracy: Determine whether the tweet's content aligns with the current scientific consensus. Consider whether the information presented is factually correct, even if the tweet does not explicitly discuss scientific matters.
- Source Credibility: Evaluate the credibility of the sources mentioned in the tweet, if any. Are these sources reputable and recognized in the scientific community?
- Tone and Intent: Assess whether the tweet promotes skepticism or misinformation about science. Consider if the tone suggests a constructive or destructive approach towards scientific understanding, even if the tweet is not overtly scientific.
- Complexity: Analyze whether the tweet oversimplifies complex scientific concepts. Does it fail to capture the nuances of the issue it addresses, even if the tweet is tangential to science?
- Context: Consider the tweet within the broader scientific discourse. If the tweet seems unrelated to science, classify it as 'neutral.'

Tweet: {{{{tweet}}}}

Relevant scientific context:
{{% for doc in documents %}}
{{{{ doc.content }}}}
{{% endfor %}}

Provide the classification as 'antiscience', 'proscience', or 'neutral'. No other text is allowed.
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

async def classify_tweet_ensemble(pipelines: Dict[str, Pipeline], embedding_pipeline: Pipeline, original_tweet: str, preprocessed_tweet: str, max_retries: int = 3) -> Dict[str, any]:
    embedding_result = embedding_pipeline.run({"sparse_text_embedder": {"text": preprocessed_tweet}, "dense_text_embedder": {"text": preprocessed_tweet}})
    sparse_embedding = embedding_result["sparse_text_embedder"]["sparse_embedding"]
    dense_embedding = embedding_result["dense_text_embedder"]["embedding"]

    async def classify_with_model(model_name: str, pipeline: Pipeline) -> str:
        for attempt in range(max_retries):
            try:
                logger.info(f"Classifying tweet using {model_name}, attempt {attempt + 1}")
                inputs = {
                    "retriever": {"query_sparse_embedding": sparse_embedding, "query_embedding": dense_embedding},
                    "prompt_builder": {"tweet": preprocessed_tweet}
                }
                result = await asyncio.to_thread(pipeline.run, inputs)
                classification = result["llm"]["replies"][0].strip().lower()
                logger.info(f"Result from {model_name}: {classification}")
                return classification
            except Exception as e:
                logger.error(f"Error classifying tweet with {model_name}, attempt {attempt + 1}: {e}")
        return "classification_failed"

    classifications = await asyncio.gather(*[classify_with_model(model_name, pipeline) for model_name, pipeline in pipelines.items()])
    return {"tweet": original_tweet, "preprocessed": preprocessed_tweet, "classifications": dict(zip(pipelines.keys(), classifications))}

async def process_tweets_concurrently(pipelines: Dict[str, Pipeline], embedding_pipeline: Pipeline, tweets: List[str], max_retries: int = 3) -> List[Dict[str, any]]:
    processed_tweets = [preprocess_text(tweet) for tweet in tweets]
    valid_tweets = [(original, processed) for original, processed in zip(tweets, processed_tweets) if processed]
    
    if not valid_tweets:
        logger.warning("No valid tweets after preprocessing. All tweets were skipped.")
        return []

    results = await asyncio.gather(*[classify_tweet_ensemble(pipelines, embedding_pipeline, original, processed, max_retries) for original, processed in valid_tweets])
    return [aggregate_classifications(result) for result in results if result]

def aggregate_classifications(result: Dict[str, any]) -> Dict[str, any]:
    classifications = result["classifications"]
    counter = Counter(classifications.values())
    most_common_classification = counter.most_common(1)[0][0]
    confidence = counter[most_common_classification] / len(classifications)

    return {
        "tweet": result["tweet"],
        "preprocessed": result["preprocessed"],
        "final_classification": most_common_classification,
        "confidence": confidence,
        "individual_results": classifications
    }

async def run_pipeline():
    document_store = initialize_document_store()

    scientific_papers_path = [PATH_TO_SCIENTIFIC_PAPERS.CSV]
    load_and_index_documents(scientific_papers_path, document_store)

    embedding_pipeline = create_embedding_pipeline()

    pipelines = {
        model_name: create_hybrid_rag_pipeline(model_name, few_shot_examples, document_store)
        for model_name in ["llama3.1:8b", "mistral", "phi3:medium", "gemma2"]
    }

    tweets_file_path = 'PATH_TO_TWEETS.CSV'
    sample_size = 5
    start_time = time.time()
    tweets = load_sample_tweets(tweets_file_path, sample_size)
    logger.info(f"Loaded {len(tweets)} sample tweets in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    results = await process_tweets_concurrently(pipelines, embedding_pipeline, tweets)
    total_time = time.time() - start_time

    for result in results:
        print(f"\nTweet: {result['tweet'][:100]}...")
        print(f"Preprocessed: {result['preprocessed'][:100]}...")
        print(f"Final Classification: {result['final_classification']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("Individual Model Results:")
        for model_name, classification in result['individual_results'].items():
            print(f" {model_name}: {classification}")

    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"Average Time per Tweet: {total_time / len(tweets):.2f} seconds")
    results_df = pd.DataFrame(results)
    results_df.to_csv("classification_results.csv", index=False)

def main():
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()
