"""
dataset_loader.py
Dataset Acquisition, Preprocessing, and Splitting for Machine Translation
Language Pair: English (en) -> French (fr)
Dataset Source: OPUS Books (https://huggingface.co/datasets/opus_books)
"""

import os
import re
import json
import random
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Reproducibility seed
SEED = 42
random.seed(SEED)

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def normalize_text(text: str) -> str:
    """Normalize unicode characters, trim spaces, and standardize punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    # Replace non-standard apostrophes and quotation marks
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”«»]", '"', text)
    # Standardize spaces around punctuation
    text = re.sub(r"\s+", " ", text)
    return text


def basic_tokenize(text: str) -> List[str]:
    """Lightweight regex-based word-level tokenizer handling contractions."""
    text = text.lower()
    # Separate punctuation while keeping apostrophe contractions (e.g., don't, c'est)
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
    return tokens


class Vocabulary:
    """Vocabulary mapping between tokens and unique integer IDs."""

    def __init__(self, name: str, min_freq: int = 2):
        self.name = name
        self.min_freq = min_freq
        self.token2id = dict(SPECIAL_TOKENS)
        self.id2token = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.counts = Counter()

    def add_sentence(self, sentence: str):
        tokens = basic_tokenize(sentence)
        self.counts.update(tokens)

    def build_vocab(self):
        for token, count in self.counts.items():
            if count >= self.min_freq and token not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[token] = new_id
                self.id2token[new_id] = token

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        tokens = basic_tokenize(sentence)
        ids = [self.token2id.get(tok, SPECIAL_TOKENS["<unk>"]) for tok in tokens]
        if add_special_tokens:
            return [SPECIAL_TOKENS["<sos>"]] + ids + [SPECIAL_TOKENS["<eos>"]]
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            if remove_special_tokens and i in (SPECIAL_TOKENS["<pad>"], SPECIAL_TOKENS["<sos>"], SPECIAL_TOKENS["<eos>"]):
                if i == SPECIAL_TOKENS["<eos>"]:
                    break
                continue
            tokens.append(self.id2token.get(i, "<unk>"))
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token2id)


def load_curated_fallback_data() -> List[Dict[str, str]]:
    """
    Curated high-quality English-French parallel sentences covering diverse domains:
    everyday speech, literary narrative, formal discourse, technical, and idioms.
    Used for offline resilience or fallback.
    """
    sample_pairs = [
        ("The morning sun lit up the ancient stone castle.", "Le soleil du matin illuminait l'ancien château de pierre."),
        ("She opened the heavy wooden door with caution.", "Elle ouvrit la lourde porte en bois avec précaution."),
        ("Can you explain how this neural network works?", "Pouvez-vous expliquer comment fonctionne ce réseau de neurones?"),
        ("The young girl smiled gently at her grandmother.", "La jeune fille sourit doucement à sa grand-mère."),
        ("Knowledge is power, but wisdom comes with experience.", "Le savoir est une force, mais la sagesse vient avec l'expérience."),
        ("He walked through the quiet streets under the moonlight.", "Il marchait dans les rues calmes sous la lumière de la lune."),
        ("Artificial intelligence is transforming modern society.", "L'intelligence artificielle transforme la société moderne."),
        ("They had been waiting for the train for two hours.", "Ils attendaient le train depuis deux heures."),
        ("The painter mixed vibrant colors on his palette.", "Le peintre mélangeait des couleurs vives sur sa palette."),
        ("Do not judge a book by its cover.", "Il ne faut pas juger un livre à sa couverture."),
        ("The doctor examined the patient thoroughly.", "Le médecin a examiné le patient minutieusement."),
        ("The nurse prepared the sterile instruments for surgery.", "L'infirmière prépara les instruments stériles pour l'opération."),
        ("It is raining cats and dogs outside today.", "Il pleut des cordes dehors aujourd'hui."),
        ("He finally decided to bite the bullet and apologize.", "Il a finalement décidé de serrer les dents et de s'excuser."),
        ("Scientists made an extraordinary discovery in the deep ocean.", "Les scientifiques ont fait une découverte extraordinaire dans les profondeurs de l'océan."),
        ("The library was filled with the scent of old paper.", "La bibliothèque était imprégnée de l'odeur du vieux papier."),
        ("She looked out of the window, wondering what the future held.", "Elle regardait par la fenêtre, se demandant ce que l'avenir lui réservait."),
        ("Education plays a vital role in human development.", "L'éducation joue un rôle vital dans le développement humain."),
        ("The violinist performed a breathtaking solo piece.", "Le violoniste a interprété un morceau en solo époustouflant."),
        ("Honesty and integrity are essential values.", "L'honnêteté et l'intégrité sont des valeurs essentielles."),
        ("The river flows gently into the wide blue sea.", "La rivière s'écoule doucement vers la vaste mer bleue."),
        ("He left his umbrella in the cafe yesterday.", "Il a laissé son parapluie dans le café hier."),
        ("Global climate change requires urgent international action.", "Le changement climatique mondial nécessite une action internationale urgente."),
        ("She spoke three languages fluently: English, French, and Spanish.", "Elle parlait trois langues couramment: l'anglais, le français et l'espagnol."),
        ("The mountain peaks were covered in fresh winter snow.", "Les sommets des montagnes étaient recouverts de neige fraîche d'hiver."),
        ("A cup of hot tea is comforting on cold evenings.", "Une tasse de thé chaud est réconfortante lors des soirées fraîches."),
        ("The engineer carefully inspected the bridge structure.", "L'ingénieur a inspecté attentivement la structure du pont."),
        ("Technology should serve humanity, not replace human judgment.", "La technologie devrait servir l'humanité et non remplacer le jugement humain."),
        ("They celebrated the victory with music and laughter.", "Ils célébrèrent la victoire avec de la musique et des rires."),
        ("Silence filled the majestic cathedral after the concert.", "Le silence emplit la majestueuse cathédrale après le concert."),
        ("The chef prepared a delicious traditional dinner.", "Le chef prépara un délicieux dîner traditionnel."),
        ("We must protect natural habitats for future generations.", "Nous devons protéger les habitats naturels pour les générations futures."),
        ("His arguments were convincing and supported by evidence.", "Ses arguments étaient convaincants et étayés par des preuves."),
        ("The sunset cast golden hues across the tranquil lake.", "Le coucher de soleil projetait des teintes dorées sur le lac paisible."),
        ("Books allow us to travel through time and space.", "Les livres nous permettent de voyager dans le temps et l'espace."),
        ("She was determined to achieve her lifelong dream.", "Elle était déterminée à réaliser son rêve de toujours."),
        ("The conference brought together researchers from around the globe.", "La conférence a réuni des chercheurs du monde entier."),
        ("Every problem offers an opportunity for innovation.", "Chaque problème offre une opportunité d'innovation."),
        ("The child laughed happily while chasing colorful butterflies.", "L'enfant riait joyeusement en pourchassant des papillons colorés."),
        ("Clear communication is crucial in any professional team.", "Une communication claire est cruciale au sein de toute équipe professionnelle."),
        ("The castle stood proudly atop the rocky cliff.", "Le château se dressait fièrement au sommet de la falaise rocheuse."),
        ("Music transcends linguistic and cultural boundaries.", "La musique transcende les frontières linguistiques et culturelles."),
        ("The cat slept peacefully in front of the warm fireplace.", "Le chat dormait paisiblement devant la cheminée chaleureuse."),
        ("Patience and perseverance lead to great success.", "La patience et la persévérance mènent à une grande réussite."),
        ("The stars were shining brightly in the dark night sky.", "Les étoiles brillaient intensément dans le ciel nocturne sombre."),
        ("She wrote a thoughtful letter to her childhood friend.", "Elle écrivit une lettre attentionnée à son amie d'enfance."),
        ("The bakery smelled delightfully of freshly baked baguettes.", "La boulangerie sentait délicieusement la baguette fraîchement sortie du four."),
        ("We should never underestimate the importance of kindness.", "Nous ne devrions jamais sous-estimer l'importance de la gentillesse."),
        ("The ancient philosopher questioned the nature of reality.", "Le philosophe antique s'interrogeait sur la nature de la réalité."),
        ("Traveling expands your worldview and challenges preconceptions.", "Voyager élargit la vision du monde et remet en question les idées reçues.")
    ]
    
    # Expand synthetically to 1,000 pairs with varied permutations to support full training loop offline
    expanded = []
    modifiers = [
        ("In the evening, ", "Le soir, "),
        ("Suddenly, ", "Soudainement, "),
        ("Without doubt, ", "Sans aucun doute, "),
        ("At that moment, ", "À ce moment-là, "),
        ("For many years, ", "Pendant de nombreuses années, "),
        ("Fortunately, ", "Heureusement, "),
        ("In this town, ", "Dans cette ville, "),
        ("As expected, ", "Comme prévu, "),
        ("Undoubtedly, ", "Indubitablement, "),
        ("Historically, ", "Historiquement, "),
        ("Naturally, ", "Naturellement, "),
        ("In conclusion, ", "En conclusion, "),
        ("Interestingly, ", "Chose intéressante, "),
        ("Step by step, ", "Pas à pas, "),
        ("Day after day, ", "Jour après jour, "),
        ("To begin with, ", "Pour commencer, "),
        ("In the countryside, ", "À la campagne, "),
        ("Across the valley, ", "À travers la vallée, "),
        ("During the journey, ", "Pendant le voyage, "),
        ("With great care, ", "Avec grand soin, ")
    ]

    for en, fr in sample_pairs:
        expanded.append({"en": en, "fr": fr})
        for mod_en, mod_fr in modifiers:
            clean_en = en[0].lower() + en[1:]
            clean_fr = fr[0].lower() + fr[1:]
            expanded.append({"en": f"{mod_en}{clean_en}", "fr": f"{mod_fr}{clean_fr}"})

    random.seed(SEED)
    random.shuffle(expanded)
    return expanded[:1500]


def load_opus_books_dataset(max_samples: int = 5000, data_dir: str = "data") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Downloads and preprocesses the OPUS Books en-fr dataset from Hugging Face.
    Falls back gracefully to curated literary parallel text if network is unavailable.
    Saves train.json, val.json, and test.json in data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")
    test_path = os.path.join(data_dir, "test.json")

    # If already saved locally, load and return
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"[Dataset] Loaded existing splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data

    raw_pairs = []
    print("[Dataset] Attempting to fetch OPUS Books (en-fr) via Hugging Face...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("opus_books", "en-fr", split="train", trust_remote_code=True)
        for item in dataset:
            en = normalize_text(item["translation"]["en"])
            fr = normalize_text(item["translation"]["fr"])
            en_len = len(en.split())
            fr_len = len(fr.split())
            # Filter length: between 3 and 50 tokens
            if 3 <= en_len <= 50 and 3 <= fr_len <= 50:
                raw_pairs.append({"en": en, "fr": fr})
                if len(raw_pairs) >= max_samples:
                    break
        print(f"[Dataset] Successfully fetched {len(raw_pairs)} pairs from OPUS Books.")
    except Exception as e:
        print(f"[Dataset] Hugging Face download failed ({e}). Using curated parallel corpus fallback.")
        raw_pairs = load_curated_fallback_data()

    # Deduplicate and shuffle
    seen = set()
    unique_pairs = []
    for item in raw_pairs:
        key = (item["en"], item["fr"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(item)

    random.seed(SEED)
    random.shuffle(unique_pairs)

    n_total = len(unique_pairs)
    n_train = int(0.80 * n_total)
    n_val = int(0.10 * n_total)

    train_data = unique_pairs[:n_train]
    val_data = unique_pairs[n_train:n_train + n_val]
    test_data = unique_pairs[n_train + n_val:]

    # Save to disk
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"[Dataset] Saved splits to {data_dir}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


def build_vocabularies(train_data: List[Dict], min_freq: int = 2) -> Tuple[Vocabulary, Vocabulary]:
    """Builds separate source (English) and target (French) vocabularies."""
    src_vocab = Vocabulary("English", min_freq=min_freq)
    tgt_vocab = Vocabulary("French", min_freq=min_freq)

    for item in train_data:
        src_vocab.add_sentence(item["en"])
        tgt_vocab.add_sentence(item["fr"])

    src_vocab.build_vocab()
    tgt_vocab.build_vocab()

    print(f"[Vocab] Source (EN) vocab size: {len(src_vocab)} | Target (FR) vocab size: {len(tgt_vocab)}")
    return src_vocab, tgt_vocab


if __name__ == "__main__":
    train_d, val_d, test_d = load_opus_books_dataset(max_samples=2000, data_dir="../data")
    src_v, tgt_v = build_vocabularies(train_d)
    print("Example Pair:")
    print("EN:", test_d[0]["en"])
    print("FR:", test_d[0]["fr"])
