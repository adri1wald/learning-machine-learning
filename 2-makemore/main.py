from typing import Optional

Bigram = tuple[str, str]

def get_dataset() -> list[str]:
    with open('./names.txt') as f:
        return f.read().splitlines()

def print_dataset_stats(dataset: list[str]) -> None:
    count = len(dataset)
    first_ten = dataset[:10]
    min_len = min(len(word) for word in dataset)
    max_len = max(len(word) for word in dataset)
    print(f"{count=}\n{first_ten=}\n{min_len=}\n{max_len=}")

def get_bigrams(
    dataset: list[str],
    start_token: str,
    end_token: Optional[str] = None
) -> list[Bigram]:
    end_token = end_token if end_token is not None else start_token
    bigrams: list[Bigram] = []
    for word in dataset:
        chars = [start_token] + list(word) + [end_token]
        for first_char, second_char in zip(chars, chars[1:]):
            bigrams.append((first_char, second_char))
    return bigrams

def compute_bigram_frequencies(bigrams: list[Bigram]) -> dict[Bigram, int]:
    freqs: dict[Bigram, int] = {}
    for bigram in bigrams:
        freqs[bigram] = freqs.get(bigram, 0) + 1
    return freqs

def main():
    dataset = get_dataset()
    print_dataset_stats(dataset)
    START_TOKEN = '<S>'
    END_TOKEN = '<E>'
    bigrams = get_bigrams(dataset, START_TOKEN, END_TOKEN)
    bigram_freqs = compute_bigram_frequencies(bigrams)
    print(sorted(bigram_freqs.items(), key=lambda kv: -kv[1]))

if __name__ == '__main__':
    main()
