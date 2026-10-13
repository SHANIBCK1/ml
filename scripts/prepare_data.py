"""
Dataset preparation script for NewsInsight AI.
Handles downloading, normalizing, and preparing benchmark datasets for
both Topic Classification and Sentiment Classification.

Usage:
    python scripts/prepare_data.py
"""

import sys
import os
from pathlib import Path
import urllib.request
import urllib.error
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    CATEGORIES,
    SENTIMENT_CLASSES
)
from src.data_loader import validate_and_standardize_df, load_news_dataset


# AG News public GitHub mirror (Topic benchmark)
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"


def download_file(url: str, dest_path: Path, timeout: int = 10) -> bool:
    """Attempts to download a file from a URL. Returns True if successful."""
    try:
        print(f"Downloading from {url} to {dest_path}...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NewsInsight-AI/1.0 (Educational Project)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(dest_path, "wb") as f:
                f.write(response.read())
        print(f"Successfully downloaded {dest_path.name}")
        return True
    except Exception as e:
        print(f"Notice: Could not download {url}: {e}")
        return False


def build_synthetic_benchmark_datasets() -> None:
    """
    Creates comprehensive processed datasets for Topic and Sentiment modeling
    derived from diverse, realistic news reporting across all target categories and sentiment classes.
    Ensures full reproducibility even in offline or firewalled environments.
    """
    print("Building comprehensive benchmark datasets in data/processed/ ...")

    # Topic-specific dataset with balanced classes
    topic_records = []
    
    # 1. Politics
    politics_samples = [
        ("Parliament passes historic electoral reform legislation after marathon debate", "Politics", "Positive", "BBC"),
        ("Prime Minister dissolves parliament ahead of snap autumn elections", "Politics", "Neutral", "Reuters"),
        ("Opposition leaders stage protest against proposed tax hike legislation", "Politics", "Negative", "Associated Press"),
        ("Diplomats convene emergency summit to negotiate ceasefire agreement", "Politics", "Neutral", "The Guardian"),
        ("Supreme Court upholds constitutional validity of privacy legislation", "Politics", "Positive", "Financial Times"),
        ("Government watchdog uncovers widespread fiscal irregularities in municipal council", "Politics", "Negative", "Washington Post"),
        ("Cabinet approves bilateral trade corridor treaty with neighboring nations", "Politics", "Positive", "The Hindu"),
        ("Election commission releases updated voter registration protocols", "Politics", "Neutral", "Reuters"),
        ("Bipartisan committee releases investigative report on national infrastructure security", "Politics", "Neutral", "Politico"),
        ("Voters turn out in record numbers across contentious gubernatorial elections", "Politics", "Positive", "CNN"),
        ("Foreign ministry lodges formal diplomatic protest over border infringement", "Politics", "Negative", "Al Jazeera"),
        ("Mayor announces comprehensive urban housing initiative for low-income residents", "Politics", "Positive", "Local Tribune"),
        ("Senators filibuster controversial energy deregulation amendment", "Politics", "Negative", "The Hill"),
        ("United Nations general assembly votes on humanitarian aid resolution", "Politics", "Neutral", "UN Press"),
        ("Political party releases official policy manifesto ahead of campaign trail", "Politics", "Neutral", "Euronews"),
        ("Government facing vote of no confidence after coalition talks collapse", "Politics", "Negative", "BBC"),
        ("Civic organizations launch nonpartisan voting rights awareness campaign", "Politics", "Positive", "NPR"),
        ("Anti-corruption commission freezes offshore assets linked to former ministers", "Politics", "Negative", "Reuters"),
        ("President delivers annual address outlining national economic roadmap", "Politics", "Neutral", "Bloomberg"),
        ("Regional governors sign interstate compact on environmental resource sharing", "Politics", "Positive", "Governing"),
    ]

    # 2. Sports
    sports_samples = [
        ("National soccer squad qualifies for World Cup with dramatic stoppage-time winner", "Sports", "Positive", "BBC Sport"),
        ("Star striker ruled out for six months following anterior cruciate ligament tear", "Sports", "Negative", "ESPN"),
        ("Olympic committee reveals official host candidates for upcoming summer games", "Sports", "Neutral", "Olympic News"),
        ("Tennis champion captures twenty-fourth grand slam title in straight sets", "Sports", "Positive", "Sports Illustrated"),
        ("Basketball franchise fires head coach following eighth consecutive defeat", "Sports", "Negative", "The Athletic"),
        ("Formula One racing team secures one-two podium finish in rainy grand prix", "Sports", "Positive", "Sky Sports"),
        ("Cricket board announces tournament venues and match schedule for championship", "Sports", "Neutral", "Cricinfo"),
        ("Marathon runner shatters world record by forty-five seconds", "Sports", "Positive", "Runner's World"),
        ("Doping agency sanctions three international athletes for anti-doping violations", "Sports", "Negative", "WADA"),
        ("Underdog college hockey team stages miraculous comeback to win national title", "Sports", "Positive", "NCAA"),
        ("Heavyweight boxing title unification bout confirmed for December in Las Vegas", "Sports", "Neutral", "Ring Magazine"),
        ("Stadium renovation project completed ahead of schedule and under budget", "Sports", "Positive", "Sporting News"),
        ("Golf superstar withdraws from tournament due to recurring back spasms", "Sports", "Negative", "Golf Channel"),
        ("Baseball league approves rule modifications to speed up game pace", "Sports", "Neutral", "MLB.com"),
        ("Rookie quarterback throws five touchdowns in sensational debut victory", "Sports", "Positive", "NFL Network"),
        ("Cycling tour organizers disqualify rider for dangerous sprint collision", "Sports", "Negative", "VeloNews"),
        ("National swimming team breaks world record in 4x100m freestyle relay", "Sports", "Positive", "SwimSwam"),
        ("Football club owner faces fan protests over rising season ticket prices", "Sports", "Negative", "Guardian Sport"),
        ("World athletics federation approves new high-tech running shoe guidelines", "Sports", "Neutral", "Athletics Weekly"),
        ("Gymnastics prodigy scores flawless routine at international championships", "Sports", "Positive", "Universal Sports"),
    ]

    # 3. Business
    business_samples = [
        ("Global central bank lowers benchmark interest rate by twenty-five basis points", "Business", "Positive", "Financial Times"),
        ("Major airline files for bankruptcy protection after fuel price surge", "Business", "Negative", "Wall Street Journal"),
        ("Tech conglomerate reports twenty percent surge in quarterly net profit", "Business", "Positive", "Bloomberg"),
        ("Manufacturing activity contracts for third straight month amid weak export demand", "Business", "Negative", "Reuters"),
        ("Investment bank forecasts steady economic expansion and stabilizing inflation", "Business", "Positive", "CNBC"),
        ("Retail chain announces closure of two hundred unprofitable department stores", "Business", "Negative", "Forbes"),
        ("Venture capital firms invest forty billion in renewable energy battery ventures", "Business", "Positive", "TechCrunch"),
        ("Commercial real estate delinquency rate hits ten-year high across urban hubs", "Business", "Negative", "MarketWatch"),
        ("Automaker delivers record half-million electric vehicles in single quarter", "Business", "Positive", "Barron's"),
        ("Stock market falls three percent following weak manufacturing index data", "Business", "Negative", "Bloomberg"),
        ("Pharmaceutical firm completes merger with biotech pioneer for fifty billion", "Business", "Neutral", "Reuters"),
        ("Consumer confidence index rebounds to highest level in eighteen months", "Business", "Positive", "The Economist"),
        ("Major logistics company hit with supply chain delays after port worker strike", "Business", "Negative", "FreightWaves"),
        ("Treasury yields hold steady following latest economic data release", "Business", "Neutral", "FT"),
        ("Export trade volumes expand seven percent buoyed by tariff reduction pact", "Business", "Positive", "Trade Journal"),
        ("Credit rating agency downgrades sovereign debt outlook citing budget deficit", "Business", "Negative", "Moody's"),
        ("Renewable energy developer secures two billion dollar financing facility", "Business", "Positive", "Clean Energy"),
        ("Mortgage applications drop sharply as borrowing costs reach fresh highs", "Business", "Negative", "Housing Wire"),
        ("Labor ministry publishes monthly national unemployment and wage report", "Business", "Neutral", "Bureau of Labor"),
        ("E-commerce merchant reports strong holiday sales volume breaking records", "Business", "Positive", "Business Insider"),
    ]

    # 4. Technology
    technology_samples = [
        ("Engineers develop silicon photonics processor with tenfold speed improvement", "Technology", "Positive", "IEEE Spectrum"),
        ("Critical vulnerability discovered in open-source cryptographic library", "Technology", "Negative", "Wired"),
        ("Open-source consortium releases revolutionary lightweight neural network library", "Technology", "Positive", "Hacker News"),
        ("Massive ransomware attack cripples municipal IT infrastructure across three cities", "Technology", "Negative", "BleepingComputer"),
        ("Semiconductor giant unveils two-nanometer commercial chip manufacturing process", "Technology", "Positive", "EE Times"),
        ("Regulators open antitrust probe into dominant search engine advertising monopoly", "Technology", "Negative", "TechCrunch"),
        ("Cloud platform introduces zero-latency edge computing locations in fifty countries", "Technology", "Positive", "VentureBeat"),
        ("Major social platform suffers data leak exposing hundred million user profiles", "Technology", "Negative", "CyberScoop"),
        ("Scientists develop ultra-dense solid-state battery with five-minute charge capability", "Technology", "Positive", "MIT Tech Review"),
        ("Streaming giant cuts thousands of jobs following slower subscriber growth", "Technology", "Negative", "The Verge"),
        ("Software foundation updates programming language standard with memory safety features", "Technology", "Neutral", "InfoWorld"),
        ("Robotics laboratory debuts bipedal humanoid robot capable of factory assembly", "Technology", "Positive", "Robotics Today"),
        ("Telecom provider begins nationwide rollout of next-generation standalone network", "Technology", "Positive", "ZDNet"),
        ("Smartphone vendor releases annual hardware portfolio during keynote event", "Technology", "Neutral", "Engadget"),
        ("Software bug causes temporary outage across global airline reservation system", "Technology", "Negative", "Ars Technica"),
        ("Startup unveils autonomous electric cargo drone with five-hundred-mile range", "Technology", "Positive", "DroneDJ"),
        ("Security researchers disclose zero-day flaw affecting enterprise firewalls", "Technology", "Negative", "Dark Reading"),
        ("Standards committee approves universal fast-charging specification for laptops", "Technology", "Positive", "Tom's Hardware"),
        ("Operating system developer rolls out biometric passkey authentication to all users", "Technology", "Positive", "MacRumors"),
        ("Browser developer issues urgent emergency security patch for memory corruption flaw", "Technology", "Negative", "CNet"),
    ]

    # 5. Science
    science_samples = [
        ("James Webb Space Telescope detects water vapor on potentially habitable exoplanet", "Science", "Positive", "NASA"),
        ("Severe marine heatwave bleaches eighty percent of shallow coral reefs", "Science", "Negative", "Nature Climate"),
        ("Clinical trial reveals experimental Alzheimer drug slows cognitive decline significantly", "Science", "Positive", "The Lancet"),
        ("Antarctic ice shelf calving creates iceberg twice the size of New York City", "Science", "Negative", "National Geographic"),
        ("Fusion reactor sustains fifty million degree plasma for record duration", "Science", "Positive", "Science Daily"),
        ("Extensive drought devastates biodiversity and agricultural output across Mediterranean", "Science", "Negative", "Scientific American"),
        ("Paleontologists discover pristine fossilized dinosaur embryo inside intact egg", "Science", "Positive", "Paleo Journal"),
        ("Global study confirms microplastic contamination found in deep oceanic trenches", "Science", "Negative", "Environmental Science"),
        ("Biologists synthesize novel enzyme capable of degrading plastic waste in twenty-four hours", "Science", "Positive", "Cell"),
        ("Meteorological agency confirms hottest summer on instrumental record globally", "Science", "Negative", "WMO"),
        ("Space agency launches robotic probe on mission to explore metal-rich asteroid", "Science", "Neutral", "Space.com"),
        ("Geneticists complete first gapless sequence of complex plant genome", "Science", "Positive", "Nature Genetics"),
        ("Invasive pest species threatens native pine forests across western mountains", "Science", "Negative", "Forestry Review"),
        ("Particle physics collaboration announces refined measurement of W boson mass", "Science", "Neutral", "CERN"),
        ("New antibiotic molecule effective against drug-resistant superbugs discovered in soil", "Science", "Positive", "BioCentury"),
        ("Deforestation in tropical basin accelerates according to satellite surveillance", "Science", "Negative", "Rainforest Alert"),
        ("Oceanographers map unexplored underwater mountain chain towering three thousand meters", "Science", "Positive", "Deep Sea Research"),
        ("Solar flare causes brief radio blackout across polar communication networks", "Science", "Negative", "NOAA Space Weather"),
        ("Astronomers track interstellar comet passing through inner solar system", "Science", "Neutral", "Astronomy Now"),
        ("Gene editing therapy demonstrates durable remission in clinical sickle cell trial", "Science", "Positive", "NEJM"),
    ]

    # 6. Entertainment
    entertainment_samples = [
        ("Indie film sweeps major academy awards winning best picture and best director", "Entertainment", "Positive", "Variety"),
        ("Studio cancels anticipated blockbuster following prolonged production turmoil", "Entertainment", "Negative", "Deadline"),
        ("Pop icon announces international stadium tour across five continents", "Entertainment", "Positive", "Billboard"),
        ("Streaming platform faces backlash over controversial documentary editing", "Entertainment", "Negative", "Rolling Stone"),
        ("Renowned symphony orchestra performs magnificent sold-out anniversary gala", "Entertainment", "Positive", "Classic FM"),
        ("Beloved television comedy star passes away at age seventy-eight", "Entertainment", "Negative", "Hollywood Reporter"),
        ("Animated feature crosses one billion mark at global box office", "Entertainment", "Positive", "Box Office Mojo"),
        ("Film festival jury awards prestigious Golden Palm to debut female director", "Entertainment", "Positive", "Cahiers du Cinema"),
        ("Concert promoter cancels music festival due to severe flooding and logistical chaos", "Entertainment", "Negative", "Pitchfork"),
        ("Video game adaptation scores highest video game premiere ratings in history", "Entertainment", "Positive", "IGN"),
        ("Museum opens extensive retrospective exhibition honoring impressionist master", "Entertainment", "Neutral", "Artforum"),
        ("Rock band reunites for special charity concert raising millions for relief", "Entertainment", "Positive", "NME"),
        ("Celebrity couple files for divorce citing irreconcilable differences", "Entertainment", "Negative", "People"),
        ("Television network releases fall programming schedule featuring twelve new dramas", "Entertainment", "Neutral", "TV Guide"),
        ("Broadway musical revival earns rave critical reviews and standing ovations", "Entertainment", "Positive", "Playbill"),
        ("Production on fantasy series halted due to ongoing screenwriters contract dispute", "Entertainment", "Negative", "Entertainment Weekly"),
        ("Documentary revealing wildlife conservation triumphs wins international festival prize", "Entertainment", "Positive", "IndieWire"),
        ("Historical novel wins prestigious literary Booker prize for fiction", "Entertainment", "Positive", "Publishers Weekly"),
        ("Comic book convention draws over one hundred thousand fans for anniversary weekend", "Entertainment", "Neutral", "Polygon"),
        ("Cinemas report lowest ticket sales in decades during disappointing summer season", "Entertainment", "Negative", "CinemaBlend"),
    ]

    all_groups = [
        politics_samples, sports_samples, business_samples,
        technology_samples, science_samples, entertainment_samples
    ]

    dates_base = pd.date_range(start="2026-08-01", periods=20, freq="2D")
    
    rows = []
    idx = 1
    for group in all_groups:
        for i, (headline, cat, sent, source) in enumerate(group):
            pub_date = str(dates_base[i % len(dates_base)])
            rows.append({
                "article_id": f"proc_{idx:04d}",
                "published_at": pub_date,
                "headline": headline,
                "article_text": headline + " More context and details regarding this event are developing.",
                "category": cat,
                "sentiment": sent,
                "source": source
            })
            idx += 1

    # If AG News raw CSV exists, extract an authentic subset
    raw_ag = RAW_DATA_DIR / "ag_news_train.csv"
    ag_records = []
    if raw_ag.exists():
        try:
            print("Processing authentic AG News records from data/raw/ag_news_train.csv...")
            ag_df = pd.read_csv(raw_ag, header=None, names=["class_idx", "headline", "article_text"])
            class_map = {1: "Politics", 2: "Sports", 3: "Business", 4: "Technology"}
            # Sample 400 per class = 1600 authentic records
            sampled_parts = []
            for c_idx, c_name in class_map.items():
                part = ag_df[ag_df["class_idx"] == c_idx].head(350).copy()
                part["category"] = c_name
                part["source"] = "Reuters / AP"
                sampled_parts.append(part)
            ag_sampled = pd.concat(sampled_parts, ignore_index=True)
            
            # Generate synthetic dates over a 30-day timeline for AG news
            ag_dates = pd.date_range(start="2026-08-15", periods=30, freq="D")
            for i, r in ag_sampled.iterrows():
                ag_records.append({
                    "article_id": f"ag_{i:05d}",
                    "published_at": str(ag_dates[i % len(ag_dates)]),
                    "headline": str(r["headline"]).strip(),
                    "article_text": str(r["article_text"]).strip(),
                    "category": r["category"],
                    "sentiment": None,
                    "source": r["source"]
                })
            print(f"Extracted {len(ag_records)} authentic records from AG News.")
        except Exception as e:
            print(f"Notice: Failed to process AG News subset: {e}")

    # Combine benchmark sets
    df_curated = pd.DataFrame(rows)
    if ag_records:
        df_ag = pd.DataFrame(ag_records)
        df_topic_full = pd.concat([df_curated, df_ag], ignore_index=True)
    else:
        df_topic_full = df_curated

    # Save Topic Dataset
    topic_path = PROCESSED_DATA_DIR / "topic_news.csv"
    df_topic_full.to_csv(topic_path, index=False)
    print(f"Saved {len(df_topic_full)} topic records to {topic_path}")

    # Save Sentiment Dataset
    sentiment_path = PROCESSED_DATA_DIR / "sentiment_news.csv"
    df_curated.to_csv(sentiment_path, index=False)
    print(f"Saved {len(df_curated)} sentiment records to {sentiment_path}")

    # Also make sure sample_news.csv is populated
    sample_path = SAMPLE_DATA_DIR / "sample_news.csv"
    if not sample_path.exists():
        df_curated.head(60).to_csv(sample_path, index=False)
        print(f"Saved sample news dataset to {sample_path}")


def main():
    print("=" * 60)
    print("NewsInsight AI - Dataset Preparation Pipeline")
    print("=" * 60)

    # Attempt AG News download if user desires raw benchmark
    raw_ag = RAW_DATA_DIR / "ag_news_train.csv"
    if not raw_ag.exists():
        download_file(AG_NEWS_URL, raw_ag)

    # Build reproducible benchmark datasets
    build_synthetic_benchmark_datasets()
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
