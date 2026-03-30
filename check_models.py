import os
import re
import warnings

import google.generativeai as genai
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*google\.generativeai.*",
)

# Load the environment variables from .env file
load_dotenv()

# Hard-coded snapshot of published Free Tier limits (Last checked: 2025-08-26 per docs)
# Columns interpreted as: RPM = Requests/Minute, TPM = Input Tokens/Minute, RPD = Requests/Day
# Source: https://ai.google.dev/gemini-api/docs/rate-limits#free-tier
FREE_TIER_LIMITS = {
    "gemini-3.1-pro":   {"RPM": 5,   "TPM": 250_000,   "RPD": 50},   # Assumed/Placeholder
    "gemini-3-pro-preview":   {"RPM": 5,   "TPM": 250_000,   "RPD": 50},   # Assumed/Placeholder
    "gemini-3-flash-preview": {"RPM": 15,  "TPM": 1_000_000, "RPD": 1_500}, # Assumed/Placeholder
    "gemini-3.1-flash-lite": {"RPM": 15, "TPM": 1_000_000, "RPD": 1_500}, # Assumed/Placeholder
    "gemini-2.5-pro":   {"RPM": 5,   "TPM": 250_000,   "RPD": 100},
    "gemini-2.5-flash": {"RPM": 10,  "TPM": 250_000,   "RPD": 250},
    "gemini-2.5-flash-lite": {"RPM": 15, "TPM": 250_000,  "RPD": 1_000},
    "gemini-2.0-flash": {"RPM": 15,  "TPM": 1_000_000, "RPD": 200},
    "gemini-2.0-flash-lite": {"RPM": 30, "TPM": 1_000_000, "RPD": 200},
    "gemini-1.5-flash": {"RPM": 15,  "TPM": 250_000,   "RPD": 50},
    "gemini-1.5-flash-8b": {"RPM": 15, "TPM": 250_000,  "RPD": 50},
}

# Descending "strength" ordering (capability/quality focus, larger -> smaller) for sorting.
# Rationale: Pro > Flash > Flash-Lite within same major/minor version; newer versions before older.
MODEL_STRENGTH_ORDER = [
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]
_ORDER_INDEX = {k: i for i, k in enumerate(MODEL_STRENGTH_ORDER)}

DOC_URL = "https://ai.google.dev/gemini-api/docs/rate-limits#free-tier"

def normalize_model_key(full_name: str) -> str:
    """Extract a canonical key for lookup from API-returned model name.

    Examples:
        models/gemini-2.5-pro                -> gemini-2.5-pro
        models/gemini-3-flash-preview        -> gemini-3-flash-preview
        models/gemini-2.5-pro-latest         -> gemini-2.5-pro
        models/gemini-2.5-flash-lite         -> gemini-2.5-flash-lite
        models/gemini-2.5-flash-lite-latest  -> gemini-2.5-flash-lite
    """
    name = full_name.lower()
    # Strip leading namespace like 'models/'
    if "/" in name:
        name = name.split("/")[-1]
    # Remove common suffixes
    name = re.sub(r"-(latest|exp|experimental|preview)$", "", name)
    return name

def print_limits_row(model_name: str, limits: dict | None, inferred_key: str):
    if limits:  # If SDK provided runtime limits (uncommon currently)
        print("    SDK rate_limits:")
        for k, v in limits.items():
            print(f"      {k}: {v}")
    mapped = FREE_TIER_LIMITS.get(inferred_key)
    if mapped:
        print("    Free Tier (snapshot): " + ", ".join(f"{k}={v}" for k, v in mapped.items()))
    else:
        print(f"    (No hard-coded limits for '{inferred_key}' — see {DOC_URL})")

def main():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in .env file or environment.")
            return

        genai.configure(api_key=api_key)

        print("--- Free Tier Gemini Models & Published Limits Snapshot ---")
        print(f"Documentation Source: {DOC_URL}")
        print("Snapshot Date: 2025-08-26 (update this if the docs change)\n")

        # Collect then sort models according to strength order
        collected = []
        for m in genai.list_models():
            name_l = m.name.lower()
            # Filter: plausible free-tier text models (exclude preview/vision/ultra/native audio/live etc.)
            exclude_tokens = ["vision", "ultra", "live", "audio", "imagen", "video", "tts"]
            # Allow preview models that we actively use (e.g., 3.1-flash-lite-preview)
            is_known_preview = "flash-lite-preview" in name_l
            if any(tok in name_l for tok in exclude_tokens):
                continue
            if "preview" in name_l and not is_known_preview:
                continue
            # Removed strict version check to allow future models (e.g., gemini-4.0)
            # if not ('flash' in name_l or 'pro' in name_l ...): continue
            if 'generateContent' not in m.supported_generation_methods:
                continue
            inferred_key = normalize_model_key(m.name)
            collected.append((inferred_key, m))

        # Sort by predefined strength order; unknown keys go to the end preserving name order
        collected.sort(key=lambda tup: (_ORDER_INDEX.get(tup[0], len(MODEL_STRENGTH_ORDER)), tup[0]))

        if not collected:
            print("No candidate free-tier models were detected with current filters.")
        else:
            for inferred_key, m in collected:
                rank = _ORDER_INDEX.get(inferred_key)
                rank_str = f"rank={rank+1}" if rank is not None else "rank=unknown"
                print(f"- {m.name}  (key: {inferred_key}, {rank_str})")
                print_limits_row(getattr(m, 'name', m), getattr(m, 'rate_limits', None), inferred_key)

        print("\nLegend: RPM = requests/minute, TPM = input tokens/minute, RPD = requests/day")
        print("Values are indicative; refer to docs for authoritative & latest limits.")
        print("----------------------------------------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
