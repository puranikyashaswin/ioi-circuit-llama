import random

NAMES = ["John", "Mary", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

TEMPLATES = [
    "When {a1} and {b1} went to the market, {a2} gave a book to",
    "After {a1} and {b1} finished the meeting, {a2} passed the note to",
    "While {a1} and {b1} were at the park, {a2} handed the ball to",
    "Once {a1} and {b1} arrived at the office, {a2} sent the report to",
    "As {a1} and {b1} left the restaurant, {a2} returned the keys to",
    "Before {a1} and {b1} entered the theater, {a2} offered a ticket to",
]


def make_pairs(n, corruption="mild", seed=42):
    """Generate IOI clean/corrupt sentence pairs with name labels.
    Mild swaps only the second subject mention, strong swaps both
    name positions.
    """
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        tpl = TEMPLATES[i % len(TEMPLATES)]
        a, b = rng.sample(NAMES, 2)

        clean = tpl.format(a1=a, b1=b, a2=a)

        if corruption == "mild":
            corrupt = tpl.format(a1=a, b1=b, a2=b)
        elif corruption == "strong":
            corrupt = tpl.format(a1=b, b1=a, a2=b)
        else:
            raise ValueError(f"bad corruption type: {corruption}")

        # b is the answer, the indirect object
        pairs.append({
            "clean": clean,
            "corrupt": corrupt,
            "correct_name": b,
            "wrong_name": a,
            "tpl_idx": i % len(TEMPLATES),
        })
    return pairs
