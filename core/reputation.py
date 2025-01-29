import pandas as pd

def calculate_reputation(search_results, reputation_csv_path):
    """Calculate reputation scores based on ranked domains."""
    try:
        reputation_df = pd.read_csv(reputation_csv_path)
        reputation_df = reputation_df.set_index("domain")["score"].to_dict()

        ranked_domains = []
        total_score = 0
        count = 0

        for engine, urls in search_results.items():
            for url in urls:
                domain = extract_domain(url)
                score = reputation_df.get(domain, 50)  # Default score if domain not found
                ranked_domains.append({"domain": domain, "score": score})
                total_score += score
                count += 1

        reputation_score = round(total_score / count, 2) if count > 0 else 50
        return ranked_domains, reputation_score  # ✅ Ensure returning both values

    except Exception as e:
        print(f"Error in reputation calculation: {e}")
        return [], 50  # ✅ Ensure returning a tuple even in case of failure

def extract_domain(url):
    """Extract domain name from URL."""
    return url.split("//")[-1].split("/")[0].replace("www.", "")
