"""
Interactive setup utility for Kaggle API credentials.
Generates or updates a local .env file in the repository root.
Cross-platform: works on Linux, macOS, and Windows.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"



def prompt_kaggle_token(force_overwrite: bool = False) -> bool:
    """
    Guides the user through setting their Kaggle API token.
    Saves the key to .env and updates os.environ in memory.
    Returns True if a token was configured, False otherwise.
    """
    print("=" * 70)
    print("       KAGGLE API TOKEN CONFIGURATION (Cross-Platform Setup)")
    print("=" * 70)
    print(
        "\nTo download the Golub benchmark dataset automatically, an API token is required.\n"
        "Follow these quick steps:\n"
        "  1. Log into your Kaggle account (https://www.kaggle.com)\n"
        "  2. Go to your Account Settings -> https://www.kaggle.com/settings\n"
        "  3. Switch to the 'API Tokens' tab\n"
        "  4. Click 'Generate New Token' (under the title 'API Tokens (Recommended)')\n"
        "  5. Enter a token name (e.g., 'leukemia-subtype-benchmark') and click 'Generate'\n"
        "  6. Copy the generated token string (starts with 'KGAT_...') (first field in the popup)"
    )
    print("-" * 70)

    # Check for existing token
    existing_token = os.environ.get("KAGGLE_API_TOKEN")
    if not existing_token and ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("KAGGLE_API_TOKEN="):
                    existing_token = line.strip().split("=", 1)[1].strip("'\"")

    if existing_token and not force_overwrite:
        masked = existing_token[:8] + "..." + existing_token[-4:] if len(existing_token) > 12 else "****"
        print(f"\n[!] A Kaggle token is already configured ({masked})")
        overwrite = input("    Do you want to overwrite it? [y/N]: ").strip().lower()
        if overwrite not in ["y", "yes"]:
            print("[+] Keeping existing token.")
            os.environ["KAGGLE_API_TOKEN"] = existing_token
            return True

    while True:
        try:
            token = input("\nPaste your Kaggle API Token: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[!] Setup cancelled by user.", file=sys.stderr)
            return False

        if not token:
            print("[!] Token cannot be empty. Please enter your token or press Ctrl+C to cancel.")
            continue
        break

    # Read existing lines to preserve other user configurations in .env
    env_lines = []
    token_written = False
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("KAGGLE_API_TOKEN="):
                    env_lines.append(f"KAGGLE_API_TOKEN={token}\n")
                    token_written = True
                else:
                    env_lines.append(line)

    if not token_written:
        env_lines.append(f"KAGGLE_API_TOKEN={token}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(env_lines)

    # Update active environment in memory
    os.environ["KAGGLE_API_TOKEN"] = token

    print(f"\n[✓] Token successfully saved to: {ENV_PATH.name}")
    return True


if __name__ == "__main__":
    success = prompt_kaggle_token()
    if not success:
        sys.exit(1)