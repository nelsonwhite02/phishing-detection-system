import random
import pandas as pd

# Load your dataset (assuming it’s in CSV format with a 'url' column)
df = pd.read_csv("malicious_phish.csv")
original_urls = df['url'].tolist()

# Define functions for common phishing transformations
def add_subdomain(url):
    # Add a random subdomain like 'login', 'secure', or 'account'
    subdomain = random.choice(["login", "secure", "account"])
    parts = url.split("//")
    if len(parts) > 1:
        return parts[0] + "//" + subdomain + "." + parts[1]
    return url

def misspell_domain(url):
    # Replace a letter in the domain with a similar-looking character
    url = url.replace("bank", "bnk").replace("secure", "s3cure")
    return url

def add_special_characters(url):
    # Add dashes or numbers to simulate phishing patterns
    return url.replace(".", "-") if "." in url else url

def use_ip_address(url):
    # Replace the domain with a random IP address
    ip_address = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    parts = url.split("//")
    if len(parts) > 1:
        return parts[0] + "//" + ip_address
    return url

# Generate synthetic phishing URLs
synthetic_phishing_urls = []
for url in original_urls[:10000]:  # Limit to a subset, e.g., 10,000 URLs, to keep dataset balanced
    transformation = random.choice([add_subdomain, misspell_domain, add_special_characters, use_ip_address])
    synthetic_url = transformation(url)
    synthetic_phishing_urls.append(synthetic_url)

# Create a DataFrame for synthetic URLs and label them as phishing
synthetic_df = pd.DataFrame(synthetic_phishing_urls, columns=['url'])
synthetic_df['label'] = 'phishing'

# Combine with original data and save to a new file
df['label'] = 'legitimate'
augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
augmented_df.to_csv("augmented_dataset.csv", index=False)
