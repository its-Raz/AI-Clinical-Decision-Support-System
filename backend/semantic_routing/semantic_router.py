"""
backend/semantic_router/semantic_router.py

Custom semantic router using LangChain OpenAIEmbeddings + cosine similarity.
No external routing libraries — raw numpy + LangChain embeddings only.
"""

import os
import logging
import numpy as np
import yaml
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
log = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Because both vectors are unit-length, this reduces to a dot product.
    Result is in the range [-1.0, 1.0].
    """
    return float(np.dot(vec_a, vec_b))


def _normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise a vector to unit length."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class RouteIndex:
    """
    Builds and caches the embedding centroid for each route at startup.

    Uses LangChain's OpenAIEmbeddings, which handles batching, retries,
    and base_url routing automatically.
    """

    def __init__(self):
        cfg = _load_config()

        model    = cfg["embedding"]["EMBEDDING_MODEL"]
        base_url = cfg["embedding"]["BASE_URL"]
        api_key  = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")

        self.lower_threshold = cfg["routing"]["lower_threshold"]
        self.upper_threshold = cfg["routing"]["upper_threshold"]
        self.routes_cfg      = cfg["routes"]

        # Initialise the LangChain embeddings client
        # openai_api_base is the LangChain kwarg for a custom base URL
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        print(f"🧭 [SemanticRouter] Initialising with model: {model}")
        print(f"   base_url : {base_url}")

        self.centroids: dict[str, np.ndarray] = {}
        self._build_index()
        print(f"   ✅ Index ready. Routes: {list(self.centroids.keys())}")

    def _build_index(self):
        """
        For each route, embed all example utterances in a single batched API call,
        then average them into one centroid vector and re-normalise it.

        LangChain's embed_documents() sends all examples in one request,
        which is more efficient than calling embed_query() in a loop.
        """
        for route_name, route_data in self.routes_cfg.items():
            examples = route_data.get("examples", [])
            if not examples:
                log.warning("Route '%s' has no examples — skipping.", route_name)
                continue

            print(f"   Embedding {len(examples)} examples for '{route_name}' …")

            # embed_documents returns a plain list of lists (one per example)
            raw_vectors = self.embeddings.embed_documents(examples)

            # Convert to numpy matrix: shape (num_examples, embedding_dim)
            matrix   = np.array(raw_vectors, dtype=np.float32)

            # Average across examples → shape (embedding_dim,)
            centroid = matrix.mean(axis=0)

            # Re-normalise: averaging moves the centroid off the unit sphere
            self.centroids[route_name] = _normalise(centroid)

            log.info("Centroid built for '%s' (%d examples).", route_name, len(examples))

    def route(self, user_text: str) -> dict:
        """
        Embed the user's text and compare against every route centroid.

        Returns:
        {
            "category":   str,    # best route name, or "unmatched"
            "score":      float,  # cosine similarity of the best match
            "all_scores": dict,   # similarity score per route
            "passed":     bool,   # False if score < lower_threshold (spam gate)
            "confidence": str,    # "high" | "medium" | "spam"
        }
        """
        # embed_query returns a plain list — convert and normalise
        raw_vec   = self.embeddings.embed_query(user_text)
        query_vec = _normalise(np.array(raw_vec, dtype=np.float32))

        all_scores = {
            route_name: round(_cosine_similarity(query_vec, centroid), 4)
            for route_name, centroid in self.centroids.items()
        }

        best_route = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_route]

        if best_score < self.lower_threshold:
            confidence, passed = "spam", False
        elif best_score < self.upper_threshold:
            confidence, passed = "medium", True
        else:
            confidence, passed = "high", True

        print(f"\n🧭 [SemanticRouter] route()")
        print(f"   input      : {user_text[:80]}")
        print(f"   scores     : {all_scores}")
        print(f"   best match : {best_route} ({best_score:.4f}) — {confidence}")

        return {
            "category":   best_route if passed else "unmatched",
            "score":      best_score,
            "all_scores": all_scores,
            "passed":     passed,
            "confidence": confidence,
        }


# Module-level singleton — built once when the module is first imported
# This means the 24 API calls for centroid building happen at startup,
# not on the first user request.
_router_instance: Optional[RouteIndex] = None


def get_router() -> RouteIndex:
    """Return the shared RouteIndex singleton, building it if necessary."""
    global _router_instance
    if _router_instance is None:
        _router_instance = RouteIndex()
    return _router_instance


