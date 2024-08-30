import yaml
import json
import anthropic
import time
import re
from typing import Dict, List, Tuple

def load_api_key(config_path: str) -> str:
    """Load the Anthropic API key from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['anthropic_api_key']

def load_queries(file_path: str) -> List[Dict]:
    """Load queries from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def is_temporal_query(query: str) -> bool:
    pattern = re.compile(r"""
        \d{4}(?:\s*-\s*\d{4})?|  # Years or year ranges
        \d{1,2}(?:st|nd|rd|th)\s+century|  # Century references
        (?:last|past|next|coming|future)\s+\d+\s+(?:year|decade|century)|  # Relative time references
        (?:recent|latest|current|ongoing|future|past|historical|ancient)|  # Time-related adjectives
        (?:evolution|changes?|advancements?|developments?|progress)|  # Process words
        (?:solar\s+eclipse|equinox|solstice)|  # Astronomical events
        (?:mission|probe|telescope|spacecraft)|  # Space missions
        (?:nobel\s+prize|discovery|detection)|  # Scientific milestones
        (?:before\s+and\s+after|compared\s+with|vs\.?)|  # Comparative phrases
        (?:since|until|prior\s+to|post-|pre-)|  # Temporal prepositions
        (?:AD|BC)|  # Historical era indicators
        (?:release|launch)  # Events related to data or missions
    """, re.VERBOSE | re.IGNORECASE)
    
    return bool(pattern.search(query))

def analyze_temporal_query(query: str, client: anthropic.Anthropic) -> Tuple[Dict, float]:
    """Use regex and Claude to analyze the temporal aspects of a query."""
    start_time = time.time()

    has_temporal_aspect = is_temporal_query(query)

    if not has_temporal_aspect:
        end_time = time.time()
        return {
            'has_temporal_aspect': False,
            'expected_year_filter': None,
            'expected_recency_weight': None
        }, end_time - start_time

    prompt = f"""Analyze the following query for its temporal aspects. Provide your analysis in a Python dictionary format with the following keys:
    - 'expected_year_filter': A string representing a Boolean expression for filtering years, or None if not applicable. Use 'year' as the variable name. Use lowercase 'and', 'or', 'not' for logical operators. Only use simple comparisons with years (e.g., 'year >= 2000', 'year < 1990', 'year == 2019', 'year != 2020'). Do not include any string comparisons or references to the query itself.
    - 'expected_recency_weight': An integer from 0 to 10 representing the importance of recency (0 for no recency bias, 10 for extreme recency bias), or None if not applicable.

    Here are two examples of correct output:

    1. Query: "What are the latest developments in exoplanet detection since 2015?"
    {{
        'expected_year_filter': 'year >= 2015',
        'expected_recency_weight': 8
    }}

    2. Query: "Compare galaxy formation theories from the 1990s and 2020s."
    {{
        'expected_year_filter': '(year >= 1990 and year < 2000) or (year >= 2020 and year < 2030)',
        'expected_recency_weight': 5
    }}

    Now, analyze the following query:
    Query: "{query}"

    Respond only with the Python dictionary, no other text.
    """

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    
    # Extract the dictionary from the response
    claude_result = eval(response.content[0].text)
    
    result = {
        'has_temporal_aspect': True,
        'expected_year_filter': claude_result['expected_year_filter'],
        'expected_recency_weight': claude_result['expected_recency_weight']
    }

    end_time = time.time()
    return result, end_time - start_time