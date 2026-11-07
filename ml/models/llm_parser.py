"""LLM Natural Language Parser using Qwen2.5-0.5B-Instruct with Pydantic validation."""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Valid Fashion-MNIST classes
SUPPORTED_CLASSES = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

# Normalized display name map
DISPLAY_NAME_MAP = {
    "t-shirt/top": "T-shirt/top",
    "trouser": "Trouser",
    "pullover": "Pullover",
    "dress": "Dress",
    "coat": "Coat",
    "sandal": "Sandal",
    "shirt": "Shirt",
    "sneaker": "Sneaker",
    "bag": "Bag",
    "ankle boot": "Ankle boot",
}

# Extensive synonym and phrase mapping
SYNONYM_MAP = {
    # T-shirt/top
    "t-shirt": "t-shirt/top",
    "tshirt": "t-shirt/top",
    "tee": "t-shirt/top",
    "top": "t-shirt/top",
    "crop top": "t-shirt/top",
    "tank top": "t-shirt/top",
    "t-shirts": "t-shirt/top",
    "tees": "t-shirt/top",
    "tops": "t-shirt/top",

    # Trouser
    "trouser": "trouser",
    "trousers": "trouser",
    "pants": "trouser",
    "jeans": "trouser",
    "slacks": "trouser",
    "sweatpants": "trouser",
    "joggers": "trouser",
    "pant": "trouser",

    # Pullover
    "pullover": "pullover",
    "pullovers": "pullover",
    "sweater": "pullover",
    "sweaters": "pullover",
    "jumper": "pullover",
    "jumpers": "pullover",
    "hoodie": "pullover",
    "hoodies": "pullover",
    "sweatshirt": "pullover",
    "sweatshirts": "pullover",
    "cardigan": "pullover",

    # Dress
    "dress": "dress",
    "dresses": "dress",
    "gown": "dress",
    "gowns": "dress",
    "frock": "dress",
    "frocks": "dress",

    # Coat
    "coat": "coat",
    "coats": "coat",
    "jacket": "coat",
    "jackets": "coat",
    "overcoat": "coat",
    "blazer": "coat",
    "blazers": "coat",
    "parka": "coat",
    "trench": "coat",
    "trenchcoat": "coat",

    # Sandal
    "sandal": "sandal",
    "sandals": "sandal",
    "flip flop": "sandal",
    "flip flops": "sandal",
    "slides": "sandal",
    "slippers": "sandal",

    # Shirt
    "shirt": "shirt",
    "shirts": "shirt",
    "button down": "shirt",
    "button-down": "shirt",
    "collared shirt": "shirt",
    "blouse": "shirt",
    "dress shirt": "shirt",

    # Sneaker
    "sneaker": "sneaker",
    "sneakers": "sneaker",
    "trainer": "sneaker",
    "trainers": "sneaker",
    "tennis shoe": "sneaker",
    "tennis shoes": "sneaker",
    "running shoe": "sneaker",
    "running shoes": "sneaker",
    "kicks": "sneaker",

    # Bag
    "bag": "bag",
    "bags": "bag",
    "handbag": "bag",
    "handbags": "bag",
    "purse": "bag",
    "purses": "bag",
    "tote": "bag",
    "totes": "bag",
    "backpack": "bag",
    "backpacks": "bag",
    "clutch": "bag",
    "pocketbook": "bag",

    # Ankle boot
    "ankle boot": "ankle boot",
    "ankle boots": "ankle boot",
    "boot": "ankle boot",
    "boots": "ankle boot",
    "bootie": "ankle boot",
    "booties": "ankle boot",
    "chelsea boot": "ankle boot",
    "chelsea boots": "ankle boot",
}

# Known unsupported clothing/accessory categories to explicitly detect and reject
UNSUPPORTED_CATEGORIES = {
    "hat", "hats", "cap", "caps", "beanie", "beanies", "fedora", "fedoras", "beret", "berets",
    "sunglasses", "glasses", "shades", "eyewear",
    "watch", "watches", "wristwatch", "wristwatches", "rolex",
    "necklace", "necklaces", "ring", "rings", "earring", "earrings", "jewelry", "jewellery", "bracelet", "bracelets",
    "belt", "belts",
    "tie", "ties", "bowtie", "bowties", "necktie", "neckties",
    "sock", "socks", "stockings", "tights",
    "scarf", "scarves",
    "glove", "gloves", "mittens",
    "swimsuit", "swimsuits", "bikini", "bikinis", "swimwear", "trunks",
    "underwear", "boxers", "briefs", "bra", "bras",
    "wallet", "wallets", "suit", "suits", "tuxedo", "tuxedos", "umbrella", "umbrellas",
}

WORD_TO_NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "single": 1, "pair": 2, "couple": 2
}


class GenerationRequestSchema(BaseModel):
    """Pydantic validated schema for parsed image generation requests."""
    class_label: str = Field(description="The normalized Fashion-MNIST class label")
    count: int = Field(default=1, ge=1, le=8, description="Number of images to generate (1 to 8)")


class ParseResult(BaseModel):
    """Overall outcome of text prompt interpretation."""
    is_valid: bool
    class_label: Optional[str] = None
    display_label: Optional[str] = None
    class_idx: Optional[int] = None
    count: int = 1
    raw_llm_output: Optional[str] = None
    strategy: str = "zero-shot"
    error_message: Optional[str] = None
    unsupported_requested: Optional[str] = None
    warning: Optional[str] = None


class QwenClothingParser:
    """
    Parser using Qwen2.5-0.5B-Instruct for text interpretation into strict JSON,
    with defensive prompt injection protection, schema validation, and recovery.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self):
        """Lazily load Qwen2.5-0.5B-Instruct tokenizer and model."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading Qwen model from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 0.5B model is fast and light (~1GB VRAM or fast on CPU)
        dtype = torch.float16 if torch.cuda.is_available() and self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info("Qwen model loaded successfully.")

    def build_prompt(self, user_prompt: str, strategy: str = "zero-shot") -> str:
        """Constructs zero-shot or few-shot system and user prompts."""
        system_instruction = (
            "You are a strict JSON extraction assistant for a Fashion-MNIST synthesis system.\n"
            "Fashion-MNIST only supports these 10 categories:\n"
            "0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, "
            "5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot.\n\n"
            "Extract the requested category and the quantity (an integer between 1 and 8, default 1).\n"
            "If the requested item is NOT one of these 10 categories, set class_label to 'unsupported'.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            '{"class_label": "<label_or_unsupported>", "count": <integer_1_to_8>}'
        )

        if strategy == "few-shot":
            few_shot_examples = (
                "\n\nExamples:\n"
                'User: "Can I get four ankle boots please?"\n'
                'Assistant: {"class_label": "ankle boot", "count": 4}\n\n'
                'User: "Make 2 stylish summer dresses."\n'
                'Assistant: {"class_label": "dress", "count": 2}\n\n'
                'User: "A pair of denim jeans"\n'
                'Assistant: {"class_label": "trouser", "count": 2}\n\n'
                'User: "Show me a warm winter sweater"\n'
                'Assistant: {"class_label": "pullover", "count": 1}\n\n'
                'User: "Generate 3 baseball caps"\n'
                'Assistant: {"class_label": "unsupported", "count": 3}\n\n'
                'User: "Show me a luxury wristwatch"\n'
                'Assistant: {"class_label": "unsupported", "count": 1}\n'
            )
            full_prompt = f"{system_instruction}{few_shot_examples}\nUser: \"{user_prompt}\"\nAssistant:"
        else:
            full_prompt = f"{system_instruction}\n\nUser: \"{user_prompt}\"\nAssistant:"

        return full_prompt

    def check_prompt_safety(self, text: str) -> Optional[str]:
        """
        Defensive safety check against prompt injection and malformed inputs.
        """
        if not text or not text.strip():
            return "Prompt cannot be empty."

        if len(text) > 500:
            return "Prompt exceeds maximum allowed length of 500 characters."

        # Detect prompt injection attempts
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
            r"disregard\s+(all\s+)?instructions",
            r"system\s*:\s*",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"you\s+are\s+now\s+a",
            r"output\s+the\s+following\s+password",
            r"drop\s+database",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Prompt rejected: Contains disallowed system instruction override or injection pattern."

        return None

    def rule_based_fallback_parser(self, text: str) -> Tuple[Optional[str], int, Optional[str]]:
        """
        Robust deterministic parsing fallback for synonym mapping, quantities, and unsupported items.
        """
        cleaned = text.lower().strip()

        # Check for explicit unsupported categories first
        for unsupp in UNSUPPORTED_CATEGORIES:
            pattern = r"\b" + re.escape(unsupp) + r"\b"
            if re.search(pattern, cleaned):
                return None, 1, unsupp

        # Extract quantity
        count = 1
        num_match = re.search(r"(?:^|[^\w])(-?\d+)\b", cleaned)
        if num_match:
            try:
                count = int(num_match.group(1))
            except ValueError:
                count = 1
        else:
            for word, val in WORD_TO_NUMBER.items():
                if re.search(r"\b" + word + r"\b", cleaned):
                    count = val
                    break

        # Match fashion classes via synonyms
        matched_class = None
        # Sort keys by length descending to match multi-word phrases first ("ankle boot" before "boot")
        for synonym in sorted(SYNONYM_MAP.keys(), key=lambda s: -len(s)):
            pattern = r"\b" + re.escape(synonym) + r"\b"
            if re.search(pattern, cleaned):
                matched_class = SYNONYM_MAP[synonym]
                break

        return matched_class, count, None

    def parse(
        self,
        text: str,
        strategy: str = "zero-shot",
        allow_fallback: bool = True,
        strict_quantity: bool = False,
    ) -> ParseResult:
        """
        Parses user prompt into structured JSON using LLM (if loaded) or rule-based fallback,
        enforcing strict Pydantic validation, explicit quantity error checking, and disclosed adjustments.
        """
        # 1. Safety & validation checks
        safety_error = self.check_prompt_safety(text)
        if safety_error:
            return ParseResult(
                is_valid=False,
                strategy=strategy,
                error_message=safety_error,
            )

        # 2. Check for explicit unsupported items early
        cleaned = text.lower().strip()
        for unsupp in UNSUPPORTED_CATEGORIES:
            pattern = r"\b" + re.escape(unsupp) + r"\b"
            if re.search(pattern, cleaned):
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    unsupported_requested=unsupp,
                    error_message=(
                        f"Unsupported item '{unsupp}'. Fashion-MNIST only supports 10 apparel classes: "
                        f"{', '.join(DISPLAY_NAME_MAP.values())}."
                    ),
                )

        raw_output = None
        extracted_data = None

        # 3. LLM generation if loaded
        if self._is_loaded and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                prompt = self.build_prompt(text, strategy=strategy)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=60,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                # Attempt JSON extraction from LLM response
                json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group(0))
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}.")
                extracted_data = None

        # 4. If LLM did not run or produced non-JSON, handle fallback
        if not extracted_data:
            if not allow_fallback:
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    raw_llm_output=raw_output,
                    error_message="Standalone LLM parsing did not return valid category JSON (fallback disabled).",
                )

            rb_class, rb_count, rb_unsupp = self.rule_based_fallback_parser(text)
            if rb_unsupp:
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    raw_llm_output=raw_output,
                    unsupported_requested=rb_unsupp,
                    error_message=f"Requested category '{rb_unsupp}' is not supported by Fashion-MNIST.",
                )
            if not rb_class:
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    raw_llm_output=raw_output,
                    error_message=(
                        "Could not identify a supported Fashion-MNIST clothing category in prompt. "
                        "Supported: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot."
                    ),
                )
            extracted_data = {"class_label": rb_class, "count": rb_count}

        # 5. Normalize and validate
        raw_label = str(extracted_data.get("class_label", "")).lower().strip()
        raw_count = extracted_data.get("count", 1)

        # Handle explicit unsupported signal from LLM
        if raw_label in ["unsupported", "unknown", "none", "not supported"]:
            if not allow_fallback:
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    raw_llm_output=raw_output,
                    error_message="The requested item was determined unsupported by standalone LLM.",
                )
            # Try rule fallback to recover synonyms (e.g. tote bag -> bag)
            rb_class, rb_count, _ = self.rule_based_fallback_parser(text)
            if rb_class:
                raw_label = rb_class
                raw_count = rb_count
            else:
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    raw_llm_output=raw_output,
                    error_message="The requested item is not supported by Fashion-MNIST.",
                )

        # Normalize via synonym map
        normalized_label = SYNONYM_MAP.get(raw_label, raw_label)

        if normalized_label not in SUPPORTED_CLASSES:
            # Check if any supported class substring matches
            found = False
            for c in SUPPORTED_CLASSES:
                if c in normalized_label or normalized_label in c:
                    normalized_label = c
                    found = True
                    break
            if not found:
                if not allow_fallback:
                    return ParseResult(
                        is_valid=False,
                        strategy=strategy,
                        raw_llm_output=raw_output,
                        unsupported_requested=raw_label,
                        error_message=f"Category '{raw_label}' is not supported by Fashion-MNIST.",
                    )
                rb_class, rb_count, _ = self.rule_based_fallback_parser(text)
                if rb_class:
                    normalized_label = rb_class
                    raw_count = rb_count
                else:
                    return ParseResult(
                        is_valid=False,
                        strategy=strategy,
                        raw_llm_output=raw_output,
                        unsupported_requested=raw_label,
                        error_message=f"Category '{raw_label}' is not supported by Fashion-MNIST.",
                    )

        # Validate count: explicit validation error on <= 0, disclosed adjustment on > 8
        warning_msg = None
        try:
            parsed_count = int(raw_count)
        except (ValueError, TypeError):
            parsed_count = 1

        if parsed_count <= 0:
            return ParseResult(
                is_valid=False,
                strategy=strategy,
                raw_llm_output=raw_output,
                error_message=f"Invalid quantity: count must be a positive integer between 1 and 8 (got {parsed_count}).",
            )
        elif parsed_count > 8:
            if strict_quantity:
                return ParseResult(
                    is_valid=False,
                    strategy=strategy,
                    raw_llm_output=raw_output,
                    error_message=f"Quantity exceeds maximum supported limit of 8 (got {parsed_count}). Please request between 1 and 8 images.",
                )
            count_val = 8
            warning_msg = f"Requested quantity ({parsed_count}) exceeds maximum supported limit of 8. Adjusted count to 8."
        else:
            count_val = parsed_count

        try:
            schema = GenerationRequestSchema(class_label=normalized_label, count=count_val)
        except ValidationError as ve:
            return ParseResult(
                is_valid=False,
                strategy=strategy,
                raw_llm_output=raw_output,
                error_message=f"Schema validation error: {ve}",
            )

        from ml.configs.config import CLASS_TO_IDX
        class_idx = CLASS_TO_IDX[schema.class_label]
        display_label = DISPLAY_NAME_MAP[schema.class_label]

        return ParseResult(
            is_valid=True,
            class_label=schema.class_label,
            display_label=display_label,
            class_idx=class_idx,
            count=schema.count,
            raw_llm_output=raw_output,
            strategy=strategy,
            warning=warning_msg,
        )
