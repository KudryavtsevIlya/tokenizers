from typing import List, Dict, Tuple
from collections import Counter
import json
import re


class BPETokenizer:
    def __init__(self, vocab_size: int, verbose: bool = False):
        """
        Initialize the BPETokenizer.

        Args:
            vocab_size (int): Desired size of the vocabulary.
            verbose (bool): If True, enables verbose mode for debugging.
        """
        self.vocab_size = vocab_size
        self.verbose = verbose

        self.vocab: Dict[str, int] = {"[UNK]": 0}

        # for debugging
        self.merge_history: List[Tuple[str, str]] = []
        self.token_frequencies: Dict[str, int] = {}

    def train(self, corpus: List[str]):
        """
        Train the tokenizer on the provided corpus.

        Args:
            corpus (List[str]): List of strings to train the tokenizer on.
        """
        tokenized_corpus = self._tokenize_corpus(corpus)

        chars = set(char for tokens in tokenized_corpus for char in tokens)
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_byte_pair_frequencies(tokenized_corpus)
            if not pair_freqs:
                if self.verbose:
                    print("No more pairs to merge.")
                break

            most_common_pair = max(pair_freqs, key=pair_freqs.get)
            self.merge_history.append(most_common_pair)

            new_token = "".join(most_common_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if self.verbose:
                self._update_token_frequencies(tokenized_corpus)

            tokenized_corpus = self._merge_byte_pair(tokenized_corpus, most_common_pair)

            if len(self.vocab) >= self.vocab_size:
                break

    def _tokenize_corpus(self, corpus: List[str]) -> List[List[str]]:
        """
        Tokenize the corpus into a list of lists of symbols.

        Args:
            corpus (List[str]): List of strings to tokenize.

        Returns:
            List[List[str]]: Tokenized corpus.
        """
        tokenized_corpus = []
        for text in corpus:
            for word in text.strip().split():
                tokens = self._tokenize_word(word)
                tokenized_corpus.append(tokens)

        return tokenized_corpus

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into symbols.

        Args:
            word (str): The word to tokenize.

        Returns:
            List[str]: List of symbols representing the word.
        """
        return list(word) + ["</w>"]

    def _get_byte_pair_frequencies(
        self, tokenized_corpus: List[List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of each adjacent symbol pair in the corpus.

        Args:
            tokenized_corpus (List[List[str]]): The tokenized corpus.

        Returns:
            Dict[Tuple[str, str], int]: Mapping of symbol pairs to their frequencies.
        """
        pairs = Counter()
        for tokens in tokenized_corpus:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_byte_pair(
        self, tokenized_corpus: List[List[str]], pair_to_merge: Tuple[str, str]
    ) -> List[List[str]]:
        """
        Merge all occurrences of the specified symbol pair in the corpus.

        Args:
            tokenized_corpus (List[List[str]]): The tokenized corpus.
            pair_to_merge (Tuple[str, str]): The symbol pair to merge.

        Returns:
            List[List[str]]: Updated tokenized corpus after merging.
        """
        pattern = re.escape(' '.join(pair_to_merge))
        replacement = ''.join(pair_to_merge)
        new_corpus = []
        for tokens in tokenized_corpus:
            token_str = ' '.join(tokens)
            token_str = re.sub(pattern, replacement, token_str)
            new_tokens = token_str.split()
            new_corpus.append(new_tokens)
        return new_corpus
    
    def _update_token_frequencies(self, tokenized_corpus: List[List[str]]):
        """
        Update the token frequencies based on the current corpus.

        Args:
            tokenized_corpus (List[List[str]]): The tokenized corpus.
        """
        self.token_frequencies = Counter()
        for tokens in tokenized_corpus:
            self.token_frequencies.update(tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text using the learned BPE merges.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: The list of tokens.
        """
        tokens = []
        for word in text.strip().split():
            symbols = self._tokenize_word(word)
            word_tokens = self._apply_merges(symbols)
            tokens.extend(word_tokens)
        return tokens
    
    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """
        Apply the BPE merges to the list of symbols.

        Args:
            symbols (List[str]): List of symbols representing a word.

        Returns:
            List[str]: Symbols after applying merges.
        """
        merge_pairs = {pair: ''.join(pair) for pair in self.merge_history}
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            merge_candidate = None
            for i, pair in enumerate(pairs):
                if pair in merge_pairs:
                    merge_candidate = (i, pair)
                    break
            if not merge_candidate:
                break

            i, pair = merge_candidate

            symbols = symbols[:i] + [merge_pairs[pair]] + symbols[i + 2:]
        
        tokens = [token if token in self.vocab else '[UNK]' for token in symbols]
        return tokens

    def add_special_tokens(self, special_tokens: List[str]) -> None:
        """
        Add special tokens to the vocabulary.

        Args:
            special_tokens (List[str]): List of special tokens to add.
        """
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """
        Retrieve the tokenizer's vocabulary.

        Returns:
            Dict[str, int]: The vocabulary mapping tokens to their indices.
        """
        return self.vocab

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the tokenizer's training.

        Returns:
            Dict[str, any]: Dictionary containing statistics.
        """
        if not self.verbose:
            return {
                "vocab_size": len(self.vocab),
                "merge_history": "Verbose mode is off; no history available.",
                "token_frequencies": "Verbose mode is off; no frequency data available.",
            }
        return {
            "vocab_size": len(self.vocab),
            "merge_history": self.merge_history,
            "token_frequencies": self.token_frequencies,
        }

    def save(self, path: str):
        """
        Save the tokenizer's state to a file.

        Args:
            path (str): The file path to save the tokenizer.
        """
        data = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "verbose": self.verbose,
            "merge_history": [list(pair) for pair in self.merge_history],
            "token_frequencies": self.token_frequencies,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            if self.verbose:
                print(f"Tokenizer saved to {path}")
        except (FileNotFoundError, IOError) as e:
            print(f"Error saving tokenizer: {e}")

    def load(self, path: str):
        """
        Load the tokenizer's state from a file.

        Args:
            path (str): The file path to load the tokenizer from.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.vocab = data["vocab"]
            self.vocab_size = data["vocab_size"]
            self.verbose = data["verbose"]
            self.merge_history = [tuple(pair) for pair in data.get("merge_history", [])]
            self.token_frequencies = data.get("token_frequencies", {})
            if self.verbose:
                print(f"Tokenizer successfully loaded from {path}")
        except FileNotFoundError:
            print(f"Error: The file {path} does not exist.")
        except IOError as e:
            print(f"Error reading file {path}: {e}")
