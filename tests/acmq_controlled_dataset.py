"""
Controlled dataset for ACMQ-RAG memory decision experiments.

This dataset is designed to exercise:
    1. ACCEPT
    2. MERGE
    3. UPDATE
    4. ARCHIVE
    5. DISCARD
    6. LOW-RELIABILITY SAFEGUARD
"""

CONTROLLED_CASES = [

    # ---------------------------------------------------------
    # CASE 1: HIGH-QUALITY FACT
    # Expected: ACCEPT
    # ---------------------------------------------------------
    {
        "name": "high_quality_fact",
        "triples": [
            ("Steve Jobs", "founded", "Apple")
        ],
        "reliability": 1.0,
        "expected": "ACCEPT",
    },

    # ---------------------------------------------------------
    # CASE 2: EXACT DUPLICATE
    # Expected: MERGE
    # ---------------------------------------------------------
    {
        "name": "exact_duplicate",
        "triples": [
            ("Steve Jobs", "founded", "Apple"),
            ("Steve Jobs", "founded", "Apple")
        ],
        "reliability": 1.0,
        "expected": "MERGE",
    },

    # ---------------------------------------------------------
    # CASE 3: UPDATED KNOWLEDGE
    # Expected: UPDATE
    # ---------------------------------------------------------
    {
        "name": "updated_knowledge",
        "triples": [
            ("TechCorp", "has CEO", "John Smith"),
            ("TechCorp", "has CEO", "Jane Smith")
        ],
        "reliability": 1.0,
        "expected": "UPDATE",
    },

    # ---------------------------------------------------------
    # CASE 4: LOW-IMPORTANCE MEMORY
    # Expected: ARCHIVE
    # ---------------------------------------------------------
    {
        "name": "low_importance",
        "triples": [
            ("Random Person", "likes", "blue pens")
        ],
        "reliability": 1.0,
        "expected": "ARCHIVE",
    },

    # ---------------------------------------------------------
    # CASE 5: LOW-QUALITY / DISCARD CANDIDATE
    # Expected: DISCARD
    # ---------------------------------------------------------
    {
        "name": "low_quality",
        "triples": [
            ("unknown", "unknown", "unknown")
        ],
        "reliability": 1.0,
        "expected": "DISCARD",
    },

    # ---------------------------------------------------------
    # CASE 6: LOW SOURCE RELIABILITY
    # Expected: DISCARD
    #
    # This specifically tests the ACMQ-RAG critical
    # source-reliability safeguard.
    # ---------------------------------------------------------
    {
        "name": "low_reliability",
        "triples": [
            ("Random Source", "claims", "Unverified Fact")
        ],
        "reliability": 0.10,
        "expected": "DISCARD",
    },
]


def get_controlled_cases():
    """Return all controlled ACMQ-RAG experiment cases."""
    return CONTROLLED_CASES


if __name__ == "__main__":

    print("=" * 60)
    print("ACMQ-RAG CONTROLLED DATASET")
    print("=" * 60)

    for case in CONTROLLED_CASES:
        print()
        print(f"Case       : {case['name']}")
        print(f"Expected   : {case['expected']}")
        print(f"Reliability: {case['reliability']}")

        for triple in case["triples"]:
            print(
                f"  {triple[0]} | "
                f"{triple[1]} | "
                f"{triple[2]}"
            )

    print()
    print("=" * 60)
    print(f"Total cases: {len(CONTROLLED_CASES)}")
    print("=" * 60)
