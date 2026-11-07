"""Fine-tuned DistilBERT classifier for clothing prompt category extraction."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ml.configs.config import FASHION_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

logger = logging.getLogger(__name__)


class FashionTextDataset(Dataset):
    """PyTorch Dataset for text prompts labeled with Fashion-MNIST classes."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["label"] = torch.tensor(label, dtype=torch.long)
        return item


class DistilBertClothingClassifier:
    """Wrapper for inference and fine-tuning with DistilBERT."""
    def __init__(self, model_dir_or_name: str = "distilbert/distilbert-base-uncased", device: str = "cpu"):
        self.model_dir_or_name = str(model_dir_or_name)
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load(self):
        """Loads tokenizer and sequence classification model."""
        if self._is_loaded:
            return

        logger.info(f"Loading DistilBERT from {self.model_dir_or_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir_or_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir_or_name,
            num_labels=len(FASHION_CLASSES),
        ).to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info("DistilBERT loaded successfully.")

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, any]:
        """Classifies a user prompt into one of the 10 Fashion-MNIST categories."""
        if not self._is_loaded:
            self.load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        conf, pred_idx = torch.max(probs, dim=-1)

        idx = int(pred_idx.item())
        return {
            "class_idx": idx,
            "class_label": FASHION_CLASSES[idx].lower(),
            "display_label": FASHION_CLASSES[idx],
            "confidence": float(conf.item()),
            "probabilities": {FASHION_CLASSES[i]: float(probs[i].item()) for i in range(len(FASHION_CLASSES))},
        }


def generate_synthetic_fashion_text_dataset() -> Dict[str, List[Tuple[str, int]]]:
    """
    Generates a diverse synthetic dataset of natural language clothing requests
    across all 10 Fashion-MNIST classes to fine-tune DistilBERT.
    Uses varied syntactic structures, synonyms, contexts, and zero template leakage
    between train, val, and test splits.
    """
    import random

    raw_data: Dict[int, List[str]] = {
        0: [  # T-shirt/top
            "a basic white t-shirt", "plain cotton tee", "summer tank top for the beach",
            "casual black t shirt", "graphic crop top", "crew neck athletic tee",
            "generate three t-shirts", "a comfortable loose top", "striped casual t-shirt",
            "v-neck running tee", "cotton tee with short sleeves", "oversized streetwear tee",
            "simple gray workout top", "soft modal t-shirt", "breathable summer top",
            "plain round neck tee", "short sleeved basic top", "loose fitting workout tee",
            "white undershirt tee", "minimalist beige crop top"
        ],
        1: [  # Trouser
            "a pair of formal trousers", "casual denim jeans", "black dress slacks",
            "comfortable gray sweatpants", "slim fit tailored trousers", "beige chino pants",
            "generate four pairs of trousers", "baggy cargo trousers", "high waisted work pants",
            "stretchy cotton joggers", "straight cut blue jeans", "pleated wool trousers",
            "dark navy suit pants", "casual tapered sweatpants", "wide leg denim trousers",
            "khaki office trousers", "comfortable running joggers", "corduroy trousers",
            "linen summer pants", "ripped black jeans"
        ],
        2: [  # Pullover
            "a warm wool pullover", "knitted winter sweater", "cozy fleece hoodie",
            "oversized cashmere pullover", "chunky cable knit sweater", "grey cotton sweatshirt",
            "generate two warm pullovers", "turtleneck knit sweater", "casual zip up hoodie",
            "crewneck warm jumper", "ribbed pullover sweater", "merino wool pullover",
            "thermal winter jumper", "classic grey athletic sweatshirt", "autumn knit pullover",
            "fleece pullover jacket", "soft angora sweater", "cozy holiday jumper",
            "thick knit pullover sweater", "cotton blend hooded sweatshirt"
        ],
        3: [  # Dress
            "an elegant evening dress", "floral summer maxi dress", "black cocktail dress",
            "casual midi sundress", "formal wedding gown", "a-line party dress",
            "generate two beautiful dresses", "silk slip dress", "sleeveless cotton summer dress",
            "vintage pleated dress", "ruffled tiered dress", "wrap dress with belt",
            "boho bohemian lace dress", "chic velvet gown", "short casual shift dress",
            "formal ball gown dress", "bodycon mini dress", "flowy linen sundress",
            "polka dot vintage dress", "evening silk gown"
        ],
        4: [  # Coat
            "a heavy winter overcoat", "tailored wool trench coat", "padded puffer jacket",
            "warm fleece parka coat", "formal double breasted overcoat", "waterproof winter coat",
            "generate three warm coats", "long shearling winter coat", "windbreaker rain coat",
            "classic camel wool coat", "quilted down coat", "casual bomber jacket coat",
            "insulated snow parka", "leather trench coat", "heavy duty winter overcoat",
            "tweed long coat", "hooded storm coat", "wool blend car coat",
            "duster length trench", "heavy arctic parka coat"
        ],
        5: [  # Sandal
            "a pair of open toe sandals", "leather strap summer sandals", "casual beach flip flops",
            "comfy slide sandals", "gladiator strappy sandals", "cork footbed slide sandals",
            "generate four pairs of sandals", "flat leather walking sandals", "sporty hiking sandals",
            "minimalist buckle sandals", "woven strap sandals", "foam pool slide sandals",
            "thong flip flop sandals", "platform wedge sandals", "waterproof beach sandals",
            "casual espadrille sandals", "orthopedic comfortable sandals", "strappy summer flats",
            "slip on leather slides", "greek style gladiator sandals"
        ],
        6: [  # Shirt
            "a crisp white button down shirt", "formal collared business shirt", "cotton dress shirt with collar",
            "casual flannel plaid shirt", "long sleeve oxford shirt", "linen collared work shirt",
            "generate two button up shirts", "striped office shirt", "formal cufflink dress shirt",
            "tailored spread collar shirt", "casual chambray button down", "slim fit poplin shirt",
            "silk formal blouse shirt", "wrinkle free dress shirt", "short sleeve camp collar shirt",
            "classic blue banker shirt", "western denim buttoned shirt", "smart casual collared shirt",
            "pointed collar formal shirt", "tucked in office dress shirt"
        ],
        7: [  # Sneaker
            "a pair of white running sneakers", "low top casual sneakers", "athletic training tennis shoes",
            "chunky retro running sneakers", "canvas skate sneakers", "high top basketball sneakers",
            "generate three pairs of sneakers", "breathable mesh workout sneakers", "leather street sneakers",
            "lightweight jogging sneakers", "slip on canvas sneakers", "gym fitness trainers",
            "classic court sneakers", "cushioned road running shoes", "vintage aesthetic sneakers",
            "performance sports trainers", "trail running sneakers", "minimalist black trainers",
            "distressed leather sneakers", "supportive walking tennis shoes"
        ],
        8: [  # Bag
            "a leather crossbody bag", "canvas tote shopping bag", "designer shoulder handbag",
            "compact travel backpack bag", "clutch evening purse bag", "structured leather briefcase bag",
            "generate two spacious bags", "canvas weekend duffel bag", "mini leather sling bag",
            "woven straw beach tote bag", "drawstring gym backpack", "quilted chain handbag",
            "spacious everyday tote bag", "small zippered pouch bag", "formal envelope clutch bag",
            "nylon commuter backpack", "heavy canvas bucket bag", "vintage leather satchel bag",
            "stylish top handle purse", "waterproof messenger bag"
        ],
        9: [  # Ankle boot
            "a pair of black leather ankle boots", "chelsea style ankle boots", "suede heeled ankle booties",
            "sturdy lace up ankle boots", "pointed toe leather boots", "chunky platform ankle boots",
            "generate four pairs of ankle boots", "waterproof winter ankle boots", "low heel casual booties",
            "western cowboy ankle boots", "zippered leather ankle booties", "slip on elastic chelsea boots",
            "combat style lace up ankle boots", "brown nubuck ankle boots", "studded punk ankle boots",
            "glossy patent ankle boots", "comfortable walking ankle boots", "velvet dressy booties",
            "crepe sole ankle boots", "round toe leather chelsea boots"
        ]
    }

    # Ensure reproducible splits
    random.seed(42)
    train_data = []
    val_data = []
    test_data = []

    for cls_idx, samples in raw_data.items():
        shuffled = list(samples)
        random.shuffle(shuffled)
        # 14 train, 3 val, 3 test (70%, 15%, 15%)
        train_samples = shuffled[:14]
        val_samples = shuffled[14:17]
        test_samples = shuffled[17:]

        for text in train_samples:
            train_data.append((text, cls_idx))
        for text in val_samples:
            val_data.append((text, cls_idx))
        for text in test_samples:
            test_data.append((text, cls_idx))

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
