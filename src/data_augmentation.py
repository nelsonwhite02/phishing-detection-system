import pandas as pd
import random

# Function to apply phishing-like transformations
def add_subdomain(url):
    subdomain = random.choice(["login", "secure", "account"])
    parts = url.split("//")
    return parts[0] + "//" + subdomain + "." + parts[1] if len(parts) > 1 else url

def misspell_domain(url):
    return url.replace("bank", "bnk").replace("secure", "s3cure")

def add_special_characters(url):
    return url.replace(".", "-") if "." in url else url

def use_ip_address(url):
    ip_address = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    parts = url.split("//")
    return parts[0] + "//" + ip_address if len(parts) > 1 else url

def augment_data(input_file, output_file, num_samples=10000):
    df = pd.read_csv(input_file)
    original_urls = df['url'].tolist()

    synthetic_phishing_urls = []
    for url in original_urls[:num_samples]:  # Limiting to a subset for balance
        transformation = random.choice([add_subdomain, misspell_domain, add_special_characters, use_ip_address])
        synthetic_url = transformation(url)
        synthetic_phishing_urls.append(synthetic_url)

    synthetic_df = pd.DataFrame(synthetic_phishing_urls, columns=['url'])
    synthetic_df['label'] = 'phishing'

    df['label'] = 'legitimate'
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    augmented_df.to_csv(output_file, index=False)

# Run the augmentation
if __name__ == "__main__":
    augment_data("data/malicious_phish.csv", "data/augmented_dataset.csv")
