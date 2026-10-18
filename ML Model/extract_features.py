import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

# Multi-part top-level domains for accurate registered domain parsing
MULTI_TLDS = {
    "co.uk", "com.au", "com.br", "co.in", "net.in", "org.in", "gen.in",
    "co.za", "co.jp", "com.sg", "com.mx", "co.kr", "com.tr", "com.tw",
    "org.uk", "gov.uk", "ac.uk", "gov.in", "edu.au", "gov.au", "co.nz"
}

# Verified major global domains
KNOWN_TOP_DOMAINS = {
    # Search & Portals
    "google.com", "google.co.in", "google.co.uk", "google.de", "google.fr",
    "google.ca", "google.com.br", "google.co.jp", "google.it", "google.es",
    "bing.com", "yahoo.com", "duckduckgo.com", "baidu.com", "yandex.ru",
    # Video & Streaming
    "youtube.com", "youtu.be", "netflix.com", "twitch.tv", "spotify.com",
    "vimeo.com", "hulu.com", "disneyplus.com",
    # Social & Communication
    "facebook.com", "fb.com", "instagram.com", "whatsapp.com", "twitter.com",
    "x.com", "linkedin.com", "reddit.com", "tiktok.com", "pinterest.com",
    "snapchat.com", "telegram.org", "discord.com", "zoom.us",
    # Knowledge & Developer
    "wikipedia.org", "wikimedia.org", "github.com", "gitlab.com",
    "stackoverflow.com", "stackexchange.com", "medium.com", "quora.com",
    "wordpress.org", "wordpress.com",
    # E-Commerce & Cloud
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.in", "amazon.co.jp",
    "ebay.com", "ebay.co.uk", "apple.com", "icloud.com", "microsoft.com",
    "office.com", "live.com", "azure.com", "paypal.com", "stripe.com",
    "cloudflare.com", "dropbox.com", "adobe.com", "salesforce.com", "canva.com",
    # News & Media
    "nytimes.com", "cnn.com", "bbc.com", "bbc.co.uk", "reuters.com",
    "theguardian.com", "forbes.com", "imdb.com", "craigslist.org"
}

# Targeted brands frequently spoofed by phishing kits
BRANDS = {
    "google", "youtube", "facebook", "apple", "microsoft", "amazon",
    "netflix", "paypal", "chase", "wellsfargo", "bankofamerica", "ebay",
    "instagram", "whatsapp", "binance", "coinbase", "metamask", "yahoo",
    "twitter", "linkedin", "dropbox", "icloud", "outlook", "gmail",
    "tiktok", "snapchat", "telegram", "discord", "spotify"
}

# TLDs with disproportionately high abuse/phishing incidence
SUSPICIOUS_TLDS = {
    "xyz", "top", "buzz", "fit", "tk", "ml", "ga", "cf", "gq", "work",
    "icu", "rest", "center", "info", "click", "link", "country", "kim",
    "science", "party", "gdn", "mom", "loan", "racing", "cricket", "win"
}

# Sensitive security/credential keywords
SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "verification", "account", "secure",
    "update", "password", "banking", "confirm", "wallet", "payment",
    "recover", "suspend", "unlock", "webscr", "credential", "admin", "ebayisapi"
)

# Known URL shortener domains
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly"
}

VOWELS = set("aeiouAEIOU")


def canonicalize_url(url):
    """
    Standardize URL so that scheme variations (http vs https) and default prefixes (www.)
    do not cause false positives.
    Example: 'https://www.google.com' -> 'google.com/'
             'google.com'            -> 'google.com/'
             'www.youtube.com'       -> 'youtube.com/'
             'example.com?query=1'   -> 'example.com/?query=1'
    """
    u = str(url).strip()
    # Strip protocol prefix (http://, https://, ftp://, etc.)
    u = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", u)
    u = re.sub(r"^/+", "", u)
    
    # Split host and path/query cleanly
    for delim in ("/", "?", "#"):
        if delim in u:
            idx = u.index(delim)
            host = u[:idx].lower().strip()
            path_and_query = u[idx:] if delim == "/" else "/" + u[idx:]
            break
    else:
        host = u.lower().strip()
        path_and_query = "/"
    
    # Normalize common www. prefix
    if host.startswith("www.") and len(host) > 4:
        host = host[4:]
        
    return host + path_and_query


def shannon_entropy(text):
    """Compute Shannon entropy to detect algorithmic randomness or obfuscation."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def get_registered_domain(hostname):
    """
    Extract the registered domain (eTLD+1), e.g.:
    'mail.google.com' -> 'google.com'
    'login.ebay.co.uk' -> 'ebay.co.uk'
    'paypal.com-verify.sec-update.xyz' -> 'sec-update.xyz'
    """
    if not hostname:
        return ""
    parts = hostname.lower().strip(".").split(".")
    if len(parts) <= 1:
        return hostname
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract_features(url):
    """
    Extract a comprehensive set of 29 security and structural features
    from a URL for machine learning classification.
    """
    canon = canonicalize_url(url)
    parse_url = "//" + canon
    
    hostname = ""
    path = "/"
    query = ""
    port = 0

    try:
        parsed = urlparse(parse_url)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or "/"
        query = parsed.query or ""
        if parsed.port:
            port = 1
    except Exception:
        path = "/"

    # IP address check
    has_ip = 0
    if hostname:
        if ":" in hostname or re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname):
            try:
                ipaddress.ip_address(hostname)
                has_ip = 1
            except ValueError:
                pass

    reg_domain = get_registered_domain(hostname)
    reg_sld = reg_domain.split(".")[0] if "." in reg_domain else reg_domain
    tld = reg_domain.split(".")[-1] if "." in reg_domain else ""

    lower_path = path.lower()

    # Brand impersonation check (e.g. 'paypal-verify.xyz' where reg domain is not paypal.com)
    # Check within hostname (and subdomains) only, NOT the URL path/query to avoid false positives
    brand_impersonation = 0
    for brand in BRANDS:
        if brand in hostname:
            if reg_sld != brand:
                brand_impersonation = 1
                break

    # Verified top domain check (only authentic domains without spoofing)
    is_known_top = 1 if (reg_domain in KNOWN_TOP_DOMAINS and brand_impersonation == 0) else 0

    # Subdomain depth (excluding www and respecting multi-part TLDs)
    subdomain_count = 0
    if hostname and reg_domain:
        subdomain_part = hostname[: -len(reg_domain)].rstrip(".")
        if subdomain_part:
            subdomain_count = len(subdomain_part.split("."))

    # Consecutive consonants in host (random gibberish indicator)
    consonants = re.findall(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+", hostname)
    max_consonants = max((len(c) for c in consonants), default=0)

    # Vowel ratio in hostname
    vowel_count = sum(1 for c in hostname if c in VOWELS)
    vowel_ratio = vowel_count / max(1, len(hostname))

    url_len = len(canon)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    return {
        "url_length": url_len,
        "hostname_length": host_len,
        "path_length": path_len,
        "query_length": query_len,
        "dot_count_host": hostname.count("."),
        "dot_count_total": canon.count("."),
        "hyphen_count_host": hostname.count("-"),
        "hyphen_count_path": path.count("-"),
        "underscore_count": canon.count("_"),
        "slash_count_path": path.count("/"),
        "question_count": canon.count("?"),
        "equals_count": canon.count("="),
        "at_count": canon.count("@"),
        "digit_count": sum(c.isdigit() for c in canon),
        "digit_ratio_url": round(sum(c.isdigit() for c in canon) / max(1, url_len), 4),
        "digit_ratio_host": round(sum(c.isdigit() for c in hostname) / max(1, host_len), 4),
        "entropy_hostname": round(shannon_entropy(hostname), 4),
        "entropy_url": round(shannon_entropy(canon), 4),
        "has_ip": has_ip,
        "has_port": port,
        "subdomain_count": subdomain_count,
        "is_root_domain": int(path in ("", "/") and query == ""),
        "is_shortener": int(reg_domain in SHORTENERS or hostname in SHORTENERS),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "suspicious_keyword_count": sum(1 for w in SUSPICIOUS_WORDS if w in lower_path or w in query),
        "brand_impersonation": brand_impersonation,
        "is_known_top_domain": is_known_top,
        "vowel_ratio_host": round(vowel_ratio, 4),
        "max_consonants_host": max_consonants,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    data_folder = base_dir / "dataset2"
    input_file = data_folder / "malicious_phish_clean.csv"
    output_file = data_folder / "url_features.csv"

    if not input_file.exists():
        print("ERROR: Clean dataset not found at:", input_file.resolve())
        raise SystemExit(1)

    print("Loading cleaned dataset from:", input_file)
    df = pd.read_csv(input_file)
    print("Initial dataset shape:", df.shape)

    # Augment with legitimate top domains to ensure homepages and major sites are well-represented
    sample_top_domains = [
        "google.com", "google.co.in", "google.co.uk", "google.de", "google.fr", "google.ca", "google.com.br",
        "youtube.com", "facebook.com", "instagram.com", "wikipedia.org", "en.wikipedia.org",
        "amazon.com", "amazon.co.uk", "amazon.in", "apple.com", "microsoft.com", "netflix.com",
        "github.com", "yahoo.com", "reddit.com", "twitter.com", "x.com", "whatsapp.com",
        "spotify.com", "zoom.us", "cloudflare.com", "wordpress.org", "stackoverflow.com",
        "linkedin.com", "dropbox.com", "adobe.com", "nytimes.com", "cnn.com", "bbc.com",
        "ebay.com", "paypal.com", "craigslist.org", "pinterest.com", "quora.com"
    ]

    aug_rows = []
    for d in sample_top_domains:
        aug_rows.append({"url": d, "type": "benign"})
        aug_rows.append({"url": f"https://{d}", "type": "benign"})
        aug_rows.append({"url": f"https://www.{d}", "type": "benign"})
        aug_rows.append({"url": f"http://{d}", "type": "benign"})
        aug_rows.append({"url": f"https://{d}/", "type": "benign"})

    aug_df = pd.DataFrame(aug_rows)
    df = pd.concat([df, aug_df], ignore_index=True)
    df = df.drop_duplicates(subset=["url", "type"])

    print(f"Augmented dataset shape: {df.shape}")
    print("Extracting canonical features for all URLs... Please wait.")

    feature_list = [extract_features(u) for u in df["url"]]
    features_df = pd.DataFrame(feature_list)
    features_df["type"] = df["type"].values

    features_df.to_csv(output_file, index=False)
    print("\nFeature extraction completed successfully!")
    print("Output dataset shape:", features_df.shape)
    print("Features extracted:", [c for c in features_df.columns if c != "type"])
    print("Saved feature file to:", output_file.resolve())


if __name__ == "__main__":
    main()
