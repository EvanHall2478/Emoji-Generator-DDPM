from typing import List

from configs import Config
from sentence_transformers import SentenceTransformer
from sklearn import metrics
import torch
import utils
from utils import not_implemented


class TextEncoder:
    """Text encoder using T5-efficient-tiny"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = utils.get_device()

        model_name = cfg.model_name
        print(f"Loading text encoder.")
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={
                "local_files_only": True
            },
        ).to(self.device)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Text embedding dimension: {self.embedding_dim}")

        # Freeze text encoder
        # ---------------------------------------- Part 3(a)(i) [your code here]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # ----------------------------------------

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings"""
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
        )


def find_most_similar(
    query_text: str,
    query_embedding: torch.Tensor,
    candidate_texts: List[str],
    candidate_embeddings: torch.Tensor,
):
    """Finds the most similar text in a list of candidates for a given query.

    Args:
        query_text: The reference text.
        query_embedding: The embedding for the query text. Shape: (1, hidden_dim)
        candidate_texts: A list of texts to compare against.
        candidate_embeddings: Embeddings for the candidates. Shape: (num_candidates, hidden_dim)
    """
    # Move tensors to CPU and convert to numpy for scikit-learn
    query_np = query_embedding.cpu().numpy()
    candidates_np = candidate_embeddings.cpu().numpy()

    # Calculate cosine similarity between the single query and all candidates.
    # The result will have a shape of (1, num_candidates).
    similarities = metrics.pairwise.cosine_similarity(query_np, candidates_np)

    # Find the index of the highest similarity score in the resulting array
    max_idx = similarities.argmax()
    min_idx = similarities.argmin()

    print(f"Query text: '{query_text}'")
    print(f"Most similar: '{candidate_texts[max_idx]}' ({similarities[0, max_idx]})")
    print(f"Least similar: '{candidate_texts[min_idx]} ({similarities[0, min_idx]})")


def main():
    # Initialize configuration and encoder
    cfg = Config()
    encoder = TextEncoder(cfg)

    # Define 5 distinct emoji descriptions
    emoji_texts = [
        "a laughing emoji with tears of joy",
        "a smiling emoji",
        "a crying emoji",
        "an angry red face emoji",
        "a heart-eyes emoji showing love",
    ]
    query_index = 0

    # 1. Encode the `emoji_texts`
    # 2 Find the most similar text for a specific query
    # ---------------------------------------- Part 3(a)(ii) [your code here]
    # encode
    embeddings = encoder.encode(emoji_texts)

    # find most similar to query
    query_text = emoji_texts[query_index]
    query_emb = embeddings[query_index].unsqueeze(0)

    # setup other emoji texts as potential candiadates
    candidates = emoji_texts[:query_index] + emoji_texts[query_index + 1:]
    candidate_emb = torch.cat([embeddings[:query_index], embeddings[query_index + 1:]])

    find_most_similar(query_text, query_emb, candidates, candidate_emb)
    # ----------------------------------------


if __name__ == "__main__":
    main()
