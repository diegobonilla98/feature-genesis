from __future__ import annotations

import base64
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu

from feature_genesis.config import Config


ANALYST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypotheses",
        "shared_visual_trigger",
        "spatial_role",
        "counterevidence",
        "falsification_tests",
        "local_geometry",
        "contextual_attachment",
        "negative_space",
        "scale_and_orientation",
        "confidence",
    ],
    "properties": {
        "hypotheses": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "support", "contradictions"],
                "properties": {
                    "label": {"type": "string"},
                    "support": {"type": "array", "items": {"type": "string"}},
                    "contradictions": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "shared_visual_trigger": {"type": "string"},
        "spatial_role": {"type": "string"},
        "counterevidence": {"type": "array", "items": {"type": "string"}},
        "falsification_tests": {"type": "array", "items": {"type": "string"}},
        "local_geometry": {"type": "string"},
        "contextual_attachment": {"type": "string"},
        "negative_space": {"type": "string"},
        "scale_and_orientation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "canonical_label",
        "plain_english_summary",
        "visual_primitives",
        "visual_signature",
        "trigger_conditions",
        "non_trigger_conditions",
        "activation_spectrum_description",
        "class_confounding_assessment",
        "description",
        "ontology",
        "scope",
        "status",
        "confidence",
        "evidence",
        "counterevidence",
        "alternative_labels",
        "falsification_tests",
        "synthetic_test_prompts",
        "edit_tests",
        "do_not_conflate_with",
    ],
    "properties": {
        "canonical_label": {"type": "string"},
        "plain_english_summary": {"type": "string"},
        "visual_primitives": {"type": "array", "items": {"type": "string"}},
        "visual_signature": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "ink_or_background",
                "stroke_density",
                "dominant_orientation",
                "curvature",
                "topology",
                "attachment",
                "relative_scale",
                "location_dependence",
            ],
            "properties": {
                "ink_or_background": {"type": "string"},
                "stroke_density": {"type": "string"},
                "dominant_orientation": {"type": "string"},
                "curvature": {"type": "string"},
                "topology": {"type": "string"},
                "attachment": {"type": "string"},
                "relative_scale": {"type": "string"},
                "location_dependence": {"type": "string"},
            },
        },
        "trigger_conditions": {"type": "array", "items": {"type": "string"}},
        "non_trigger_conditions": {"type": "array", "items": {"type": "string"}},
        "activation_spectrum_description": {"type": "string"},
        "class_confounding_assessment": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "description": {"type": "string"},
        "ontology": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "entity",
                "part",
                "appearance",
                "geometry_or_pose",
                "context",
                "spatial_role",
            ],
            "properties": {
                "entity": {"type": "string"},
                "part": {"type": "string"},
                "appearance": {"type": "string"},
                "geometry_or_pose": {"type": "string"},
                "context": {"type": "string"},
                "spatial_role": {"type": "string"},
            },
        },
        "scope": {
            "type": "string",
            "enum": ["local_visual", "global_visual", "class", "context", "mixed", "unclear"],
        },
        "status": {
            "type": "string",
            "enum": ["provisional", "ambiguous", "mixed", "dead"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "counterevidence": {"type": "array", "items": {"type": "string"}},
        "alternative_labels": {"type": "array", "items": {"type": "string"}},
        "falsification_tests": {"type": "array", "items": {"type": "string"}},
        "synthetic_test_prompts": {"type": "array", "items": {"type": "string"}},
        "edit_tests": {"type": "array", "items": {"type": "string"}},
        "do_not_conflate_with": {"type": "array", "items": {"type": "string"}},
    },
}

VALIDATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tile_scores", "validation_summary", "failure_modes"],
    "properties": {
        "tile_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tile_id", "label_fit", "reason"],
                "properties": {
                    "tile_id": {"type": "string"},
                    "label_fit": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "validation_summary": {"type": "string"},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
    },
}

DISTINCTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["group_assessment", "features"],
    "properties": {
        "group_assessment": {"type": "string"},
        "features": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "feature_id",
                    "relationship",
                    "revised_canonical_label",
                    "revised_plain_english_summary",
                    "discriminating_visual_cues",
                    "nearest_confusable_feature_id",
                    "difference_from_nearest",
                    "confidence",
                ],
                "properties": {
                    "feature_id": {"type": "integer"},
                    "relationship": {
                        "type": "string",
                        "enum": ["distinct", "probable_duplicate", "uncertain"],
                    },
                    "revised_canonical_label": {"type": "string"},
                    "revised_plain_english_summary": {"type": "string"},
                    "discriminating_visual_cues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "nearest_confusable_feature_id": {"type": "integer"},
                    "difference_from_nearest": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


class GeminiQuotaError(RuntimeError):
    pass


class GeminiBudgetError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, config: Config, api_key: str, cache_root: Path):
        self.model = config.evaluation.gemini_model
        self.thinking_level = config.evaluation.gemini_thinking_level
        self.timeout = config.evaluation.gemini_timeout_seconds
        self.attempts = config.evaluation.gemini_request_attempts
        self.api_key = api_key
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.reuse_latest_role_cache = config.evaluation.gemini_reuse_latest_role_cache
        self.max_uncached_requests = config.evaluation.gemini_max_uncached_requests
        self.max_run_cost_usd = config.evaluation.gemini_max_run_cost_usd
        self.media_resolution = config.evaluation.gemini_media_resolution
        self.lock = threading.Lock()
        self.uncached_requests_started = 0
        self.uncached_requests_completed = 0
        self.run_cost_usd = 0.0
        self.role_cache = self.latest_role_cache()

    def latest_role_cache(self) -> dict[str, Path]:
        paths = {}
        if not self.reuse_latest_role_cache:
            return paths
        for path in self.cache_root.glob("*.json"):
            cached = json.loads(path.read_text(encoding="utf-8"))
            role = str(cached["fingerprint"]["role"])
            previous = paths.get(role)
            if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
                paths[role] = path
        return paths

    def run_summary(self) -> dict[str, Any]:
        return {
            "uncached_requests_started": self.uncached_requests_started,
            "uncached_requests_completed": self.uncached_requests_completed,
            "estimated_cost_usd_from_usage": round(self.run_cost_usd, 6),
            "maximum_uncached_requests": self.max_uncached_requests,
            "maximum_run_cost_usd": self.max_run_cost_usd,
            "media_resolution": self.media_resolution,
            "role_cache_entries": len(self.role_cache),
        }

    def model_metadata(self) -> dict[str, Any]:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}",
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        fields = [
            "name",
            "version",
            "displayName",
            "description",
            "inputTokenLimit",
            "outputTokenLimit",
        ]
        return {field: body.get(field) for field in fields}

    def request(
        self,
        prompt: str,
        image_paths: list[Path],
        schema: dict[str, Any],
        role: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        fingerprint = {
            "model": self.model,
            "thinking_level": self.thinking_level,
            "media_resolution": self.media_resolution,
            "prompt": prompt,
            "schema": schema,
            "role": role,
            "images": [
                {
                    "name": evidence_image_name(path),
                    "sha256": sha256_file(path),
                }
                for path in image_paths
            ],
        }
        cache_key = hashlib.sha256(canonical_json(fingerprint)).hexdigest()
        cache_path = self.cache_root / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["parsed"], cached["raw"], cache_key
        role_cache_path = self.role_cache.get(role)
        if role_cache_path is not None:
            cached = json.loads(role_cache_path.read_text(encoding="utf-8"))
            return cached["parsed"], cached["raw"], role_cache_path.stem
        with self.lock:
            if (
                self.max_uncached_requests > 0
                and self.uncached_requests_started >= self.max_uncached_requests
            ):
                raise GeminiBudgetError(
                    f"Uncached request ceiling reached: {self.max_uncached_requests}"
                )
            self.uncached_requests_started += 1
        parts = [{"text": prompt}]
        for image_path in image_paths:
            parts.append({"text": f"\nIMAGE FILE: {evidence_image_name(image_path)}\n"})
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": encode_base64(image_path.read_bytes()),
                    }
                }
            )
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": self.thinking_level},
                "mediaResolution": self.media_resolution,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        response = None
        for attempt in range(self.attempts):
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if response.status_code == 429 and "prepayment credits are depleted" in response.text.lower():
                raise GeminiQuotaError(response.text)
            if attempt + 1 < self.attempts:
                time.sleep(min(30.0, 2**attempt + random.random()))
        if response is None:
            raise RuntimeError("Gemini request did not return an HTTP response")
        if response.status_code == 429:
            raise GeminiQuotaError(response.text)
        response.raise_for_status()
        raw = response.json()
        text = "".join(
            part["text"]
            for candidate in raw.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if "text" in part and not part.get("thought", False)
        )
        parsed = json.loads(text)
        cache_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "parsed": parsed,
                    "raw": raw,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        usage = raw.get("usageMetadata", {})
        request_cost = (
            float(usage.get("promptTokenCount", 0)) * 1.5e-6
            + float(usage.get("candidatesTokenCount", 0)) * 7.5e-6
            + float(usage.get("thoughtsTokenCount", 0)) * 7.5e-6
        )
        with self.lock:
            self.uncached_requests_completed += 1
            self.run_cost_usd += request_cost
            if self.max_run_cost_usd > 0 and self.run_cost_usd > self.max_run_cost_usd:
                raise GeminiBudgetError(
                    f"Run cost ceiling reached: ${self.run_cost_usd:.4f}"
                )
        return parsed, raw, cache_key


def evaluate_feature_cards(
    config: Config,
    feature_inputs: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    env = read_env(Path(config.evaluation.gemini_env_path))
    api_key = env.get("GEMINI_API_KEY")
    if not api_key:
        return write_blocked_manifest(
            config,
            output_root,
            "GEMINI_API_KEY is missing",
            feature_inputs,
        )
    client = GeminiClient(config, api_key, output_root / "api_cache")
    try:
        model_metadata = client.model_metadata()
    except requests.RequestException as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    analyst_results: dict[int, dict[str, Any]] = {
        int(item["card"]["feature_id"]): {} for item in feature_inputs
    }
    raw_results: dict[int, dict[str, Any]] = {
        int(item["card"]["feature_id"]): {} for item in feature_inputs
    }
    cache_keys: dict[int, dict[str, str]] = {
        int(item["card"]["feature_id"]): {} for item in feature_inputs
    }
    futures = {}
    try:
        with ThreadPoolExecutor(
            max_workers=config.evaluation.gemini_parallel_requests
        ) as executor:
            for item in feature_inputs:
                feature_id = int(item["card"]["feature_id"])
                visual = executor.submit(
                    client.request,
                    visual_prompt(item["card"]),
                    visual_image_paths(item["discovery_images"]),
                    ANALYST_SCHEMA,
                    f"visual:{feature_id}",
                )
                quantitative = executor.submit(
                    client.request,
                    quantitative_prompt(item["card"]),
                    [],
                    ANALYST_SCHEMA,
                    f"quantitative:{feature_id}",
                )
                futures[visual] = (feature_id, "visual")
                futures[quantitative] = (feature_id, "quantitative")
            for future in as_completed(futures):
                feature_id, role = futures[future]
                parsed, raw, cache_key = future.result()
                analyst_results[feature_id][role] = parsed
                raw_results[feature_id][role] = raw
                cache_keys[feature_id][role] = cache_key
    except (GeminiQuotaError, GeminiBudgetError) as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    labels = {}
    adjudication_futures = {}
    try:
        with ThreadPoolExecutor(
            max_workers=config.evaluation.gemini_parallel_requests
        ) as executor:
            for item in feature_inputs:
                feature_id = int(item["card"]["feature_id"])
                future = executor.submit(
                    client.request,
                    adjudication_prompt(
                        item["card"],
                        analyst_results[feature_id]["visual"],
                        analyst_results[feature_id]["quantitative"],
                    ),
                    adjudication_image_paths(item["discovery_images"]),
                    LABEL_SCHEMA,
                    f"adjudicator:{feature_id}",
                )
                adjudication_futures[future] = feature_id
            for future in as_completed(adjudication_futures):
                feature_id = adjudication_futures[future]
                parsed, raw, cache_key = future.result()
                labels[feature_id] = parsed
                raw_results[feature_id]["adjudicator"] = raw
                cache_keys[feature_id]["adjudicator"] = cache_key
    except (GeminiQuotaError, GeminiBudgetError) as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    distinction_results, distinction_status = resolve_label_collisions(
        config,
        client,
        feature_inputs,
        labels,
        raw_results,
        cache_keys,
    )
    validation_results = {}
    validation_futures = {}
    try:
        with ThreadPoolExecutor(
            max_workers=config.evaluation.gemini_parallel_requests
        ) as executor:
            for item in feature_inputs:
                feature_id = int(item["card"]["feature_id"])
                future = executor.submit(
                    client.request,
                    validator_prompt(item["card"], labels[feature_id]),
                    item["validation_images"],
                    VALIDATOR_SCHEMA,
                    f"validator:{feature_id}",
                )
                validation_futures[future] = feature_id
            for future in as_completed(validation_futures):
                feature_id = validation_futures[future]
                parsed, raw, cache_key = future.result()
                validation_results[feature_id] = parsed
                raw_results[feature_id]["validator"] = raw
                cache_keys[feature_id]["validator"] = cache_key
    except (GeminiQuotaError, GeminiBudgetError) as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as error:
        return write_blocked_manifest(config, output_root, str(error), feature_inputs)
    manual_adjudications = read_manual_adjudications(config)
    records = []
    for item in feature_inputs:
        card = item["card"]
        feature_id = int(card["feature_id"])
        validation = score_validation(card, validation_results[feature_id])
        label = labels[feature_id]
        decision = feature_acceptance_decision(
            config,
            card,
            label,
            validation,
            distinction_results.get(feature_id),
            manual_adjudications.get(feature_id),
        )
        resolved_label = resolved_feature_label(label, decision["manual_adjudication"])
        record = {
            "feature_uid": feature_uid(config, card),
            "trajectory_uid": card.get("trajectory_uid"),
            "feature_id": feature_id,
            "model_step": int(card["model_step"]),
            "sae_kind": card["sae_kind"],
            "artifact_version": config.sae.artifact_version,
            "sae_seed": config.sae.primary_seed,
            "model": config.evaluation.gemini_model,
            "model_metadata": model_metadata,
            "thinking_level": config.evaluation.gemini_thinking_level,
            "prompt_version": config.evaluation.gemini_prompt_version,
            "evidence_sha256": {
                "discovery": {
                    path.name: sha256_file(path)
                    for path in item["discovery_images"]
                },
                "validation": {
                    path.name: sha256_file(path)
                    for path in item["validation_images"]
                },
            },
            "card_sha256": hashlib.sha256(canonical_json(card)).hexdigest(),
            "analysts": analyst_results[feature_id],
            "label": resolved_label,
            "llm_label": label,
            "label_source": decision["acceptance_mode"],
            "cross_feature_distinction": distinction_results.get(feature_id),
            "heldout_validation": {
                "model_response": validation_results[feature_id],
                "metrics": validation,
            },
            "acceptance_checks": decision["acceptance_checks"],
            "automatic_acceptance_checks": decision["automatic_acceptance_checks"],
            "manual_acceptance_checks": decision["manual_acceptance_checks"],
            "manual_adjudication": decision["manual_adjudication"],
            "acceptance_mode": decision["acceptance_mode"],
            "rejection_reasons": decision["rejection_reasons"],
            "validated_status": decision["validated_status"],
            "cache_keys": cache_keys[feature_id],
            "usage": {
                role: raw.get("usageMetadata", {})
                for role, raw in raw_results[feature_id].items()
            },
        }
        records.append(record)
        (output_root / f"feature_{feature_id:05d}_evaluation.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    jsonl_path = output_root / "feature_evaluations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    accepted_records = [
        record for record in records if record["validated_status"] == "accepted"
    ]
    accepted_jsonl_path = output_root / "accepted_feature_evaluations.jsonl"
    with accepted_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in accepted_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "model": config.evaluation.gemini_model,
        "model_metadata": model_metadata,
        "thinking_level": config.evaluation.gemini_thinking_level,
        "prompt_version": config.evaluation.gemini_prompt_version,
        "features": len(records),
        "accepted": len(accepted_records),
        "excluded_from_interpretation": len(records) - len(accepted_records),
        "collision_resolution": distinction_status,
        "minimum_validation_auc": config.evaluation.gemini_minimum_validation_auc,
        "manual_minimum_validation_auc": config.evaluation.gemini_manual_minimum_validation_auc,
        "maximum_validation_p_value": config.evaluation.gemini_maximum_validation_p_value,
        "minimum_confidence": config.evaluation.gemini_minimum_confidence,
        "minimum_positive_fit": config.evaluation.gemini_minimum_positive_fit,
        "gate_tolerance": config.evaluation.gemini_gate_tolerance,
        "manual_adjudication_path": str(manual_adjudication_path(config)),
        "jsonl": str(jsonl_path),
        "accepted_jsonl": str(accepted_jsonl_path),
        "run_api_usage": client.run_summary(),
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["path"] = str(manifest_path)
    return manifest


def adjudication_image_paths(image_paths: list[Path]) -> list[Path]:
    groups = grouped_image_paths(image_paths)
    spectrum = spectrum_extremes(groups["spectrum_"], 2)
    return groups["top_"][:3] + spectrum + groups["weak_"][:1] + groups["negative_"][:2]


def visual_image_paths(image_paths: list[Path]) -> list[Path]:
    groups = grouped_image_paths(image_paths)
    spectrum = one_image_per_spectrum_band(groups["spectrum_"])
    return groups["top_"][:8] + spectrum + groups["weak_"][:2] + groups["negative_"][:4]


def collision_image_paths(image_paths: list[Path]) -> list[Path]:
    groups = grouped_image_paths(image_paths)
    return groups["top_"][:2] + groups["negative_"][:1] + groups["weak_"][:1]


def grouped_image_paths(image_paths: list[Path]) -> dict[str, list[Path]]:
    prefixes = ["top_", "spectrum_", "weak_", "negative_"]
    return {
        prefix: [path for path in image_paths if path.name.startswith(prefix)]
        for prefix in prefixes
    }


def one_image_per_spectrum_band(image_paths: list[Path]) -> list[Path]:
    selected = []
    seen = set()
    for path in image_paths:
        parts = path.name.split("_")
        band = parts[1] if len(parts) > 1 else path.name
        if band not in seen:
            selected.append(path)
            seen.add(band)
    return selected


def spectrum_extremes(image_paths: list[Path], count: int) -> list[Path]:
    bands = one_image_per_spectrum_band(image_paths)
    if len(bands) <= count:
        return bands
    if count == 1:
        return bands[:1]
    return [bands[0], bands[-1]][:count]


def label_collision_groups(
    labels: dict[int, dict[str, Any]],
    similarity_threshold: float,
    maximum_group_size: int,
    maximum_groups: int,
) -> list[dict[str, Any]]:
    feature_ids = sorted(labels)
    if len(feature_ids) < 2 or maximum_groups <= 0:
        return []
    documents = []
    for feature_id in feature_ids:
        label = labels[feature_id]
        signature = label.get("visual_signature", {})
        documents.append(
            " ".join(
                [
                    str(label.get("canonical_label", "")),
                    str(label.get("plain_english_summary", "")),
                    " ".join(label.get("visual_primitives", [])),
                    " ".join(str(value) for value in signature.values()),
                ]
            ).lower()
        )
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(documents)
    similarity = cosine_similarity(vectors)
    adjacency = {feature_id: set() for feature_id in feature_ids}
    pair_scores = {}
    for left_index, left_id in enumerate(feature_ids):
        for right_index in range(left_index + 1, len(feature_ids)):
            right_id = feature_ids[right_index]
            left_tokens = canonical_label_tokens(labels[left_id])
            right_tokens = canonical_label_tokens(labels[right_id])
            overlap = len(left_tokens & right_tokens) / max(
                1,
                min(len(left_tokens), len(right_tokens)),
            )
            shared_head = bool(
                left_tokens
                and right_tokens
                and canonical_label_head(labels[left_id])
                == canonical_label_head(labels[right_id])
            )
            score = max(
                float(similarity[left_index, right_index]),
                float(overlap),
                0.8 if shared_head else 0.0,
            )
            if score >= similarity_threshold:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)
                pair_scores[(left_id, right_id)] = score
    components = []
    visited = set()
    for feature_id in feature_ids:
        if feature_id in visited or not adjacency[feature_id]:
            continue
        stack = [feature_id]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        component = sorted(component)[:maximum_group_size]
        scores = [
            pair_scores.get((min(left, right), max(left, right)), 0.0)
            for position, left in enumerate(component)
            for right in component[position + 1 :]
        ]
        components.append(
            {
                "feature_ids": component,
                "maximum_text_similarity": max(scores, default=0.0),
                "mean_text_similarity": float(np.mean(scores)) if scores else 0.0,
            }
        )
    components.sort(key=lambda row: row["maximum_text_similarity"], reverse=True)
    return components[:maximum_groups]


def canonical_label_tokens(label: dict[str, Any]) -> set[str]:
    value = str(label.get("canonical_label", "")).lower().replace("-", " ")
    return {
        token.strip(".,:;()[]{}")
        for token in value.split()
        if token.strip(".,:;()[]{}")
    }


def canonical_label_head(label: dict[str, Any]) -> str:
    value = str(label.get("canonical_label", "")).lower().replace("-", " ")
    tokens = [token.strip(".,:;()[]{}") for token in value.split()]
    return tokens[-1] if tokens else ""


def resolve_label_collisions(
    config: Config,
    client: GeminiClient,
    feature_inputs: list[dict[str, Any]],
    labels: dict[int, dict[str, Any]],
    raw_results: dict[int, dict[str, Any]],
    cache_keys: dict[int, dict[str, str]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if not config.evaluation.gemini_collision_enabled:
        return {}, {"status": "disabled", "requests": 0, "groups": []}
    groups = label_collision_groups(
        labels,
        config.evaluation.gemini_collision_similarity_threshold,
        config.evaluation.gemini_collision_max_group_size,
        config.evaluation.gemini_collision_max_requests,
    )
    if not groups:
        return {}, {"status": "not_needed", "requests": 0, "groups": []}
    inputs_by_id = {
        int(item["card"]["feature_id"]): item for item in feature_inputs
    }
    distinctions = {}
    completed_groups = []
    errors = []
    for group_index, group in enumerate(groups):
        feature_ids = group["feature_ids"]
        images = []
        for feature_id in feature_ids:
            paths = collision_image_paths(inputs_by_id[feature_id]["discovery_images"])
            images.extend(paths[: config.evaluation.gemini_collision_images_per_feature])
        try:
            parsed, raw, cache_key = client.request(
                distinction_prompt(feature_ids, labels, inputs_by_id),
                images,
                DISTINCTION_SCHEMA,
                "collision:" + ",".join(str(feature_id) for feature_id in feature_ids),
            )
        except (GeminiQuotaError, GeminiBudgetError, requests.RequestException, json.JSONDecodeError, KeyError) as error:
            errors.append(str(error))
            break
        returned = {
            int(row["feature_id"]): row
            for row in parsed.get("features", [])
            if int(row["feature_id"]) in feature_ids
        }
        for feature_id, row in returned.items():
            label = labels[feature_id]
            label["canonical_label"] = row["revised_canonical_label"]
            label["plain_english_summary"] = row["revised_plain_english_summary"]
            label["confidence"] = min(float(label["confidence"]), float(row["confidence"]))
            contrast = (
                f"feature {row['nearest_confusable_feature_id']}: "
                f"{row['difference_from_nearest']}"
            )
            if contrast not in label["do_not_conflate_with"]:
                label["do_not_conflate_with"].append(contrast)
            distinctions[feature_id] = {
                **row,
                "group_assessment": parsed["group_assessment"],
                "text_similarity": group,
            }
            raw_results[feature_id][f"collision_{group_index}"] = raw
            cache_keys[feature_id][f"collision_{group_index}"] = cache_key
        completed_groups.append(group)
    status = "complete" if not errors else "partial"
    return distinctions, {
        "status": status,
        "requests": len(completed_groups),
        "groups": completed_groups,
        "errors": errors,
    }


def distinction_prompt(
    feature_ids: list[int],
    labels: dict[int, dict[str, Any]],
    inputs_by_id: dict[int, dict[str, Any]],
) -> str:
    candidates = {
        feature_id: {
            "label": labels[feature_id],
            "score_summary": inputs_by_id[feature_id]["card"]["score_summary"],
        }
        for feature_id in feature_ids
    }
    return (
        "You are the one-time differential adjudicator for visual SAE features whose independent descriptions "
        "are unusually similar. Compare the features directly. Each image filename contains its feature directory. "
        "Every positive evidence image is a triptych: full drawing, activation box, and the corresponding local "
        "crop. Decide whether each pair is genuinely the same learned visual concept, a nearby but separable "
        "concept, or unresolved. Use topology, number of strokes, ink-versus-background polarity, orientation, "
        "curvature, thickness, attachment to surrounding strokes, negative space, relative scale, and spatial role. "
        "Do not invent distinctions unsupported by repeated evidence. If distinct, revise each label so its shortest "
        "human-readable wording contains the discriminating cue. If duplicate, say so plainly and keep compatible "
        "labels. This is the only cross-feature comparison pass.\n\nCANDIDATES:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
    )


def visual_prompt(card: dict[str, Any]) -> str:
    return (
        f"You are the blinded visual analyst for SAE feature {card['feature_id']}. "
        "Every positive file is a triptych containing the full drawing, the full drawing with the exact activation "
        "box, and a local crop of that same box. Never interpret the crop without its full-drawing context. Files "
        "beginning top_ are the strongest activations. "
        "Files beginning spectrum_ sample the full nonzero activation range from strong to weak. Files beginning "
        "weak_ are threshold-level activations, and negative_ files are matched zero-activation controls. These are "
        "actual thresholded feature activations, not independently normalized heatmaps. "
        "Ignore drawing class identity and identify the smallest repeated stroke, contour, junction, enclosed "
        "shape, orientation, or object part that distinguishes positive evidence from controls. A whole-object or "
        "class label is acceptable only if no smaller shared visual structure explains the evidence. Explicitly "
        "separate topology, number of strokes, ink-versus-background polarity, orientation, curvature, thickness, "
        "attachment, negative space, relative scale, and location dependence. Describe contradictions and changes "
        "in meaning across activation bands."
    )


def quantitative_prompt(card: dict[str, Any]) -> str:
    evidence = {
        "feature_id": card["feature_id"],
        "model_step": card["model_step"],
        "score_summary": card["score_summary"],
        "spatial_evidence": {
            key: card["evidence"].get(key)
            for key in [
                "activation_frequency_per_image",
                "active_images",
                "localized_active_images",
                "part_hit_rate",
                "median_nearest_part_distance",
                "attribute_certainty_available",
                "top_attributes",
                "top_parts",
            ]
        },
        "class_distribution": {
            group: [
                {
                    "class_name": row["class_name"],
                    "activation": row["activation"],
                }
                for row in rows
            ]
            for group, rows in card["contrastive_groups"].items()
            if group in {
                "top_positive",
                "activation_spectrum",
                "borderline_positive",
                "hard_negative",
            }
        },
        "individual_evidence": card.get("individual_evidence", {}).get(
            "discovery",
            [],
        ),
        "supplemental_evidence": card.get("supplemental_evidence", {}),
    }
    return (
        "You are the independent quantitative analyst for a sparse visual feature. Class identity is a potential "
        "confounder, not a label source. Use activation frequency, class entropy, top-tail modal-class share, exact "
        "localized response boxes, activation-spectrum behavior, and contrastive examples to propose the smallest "
        "visual mechanism. Penalize hypotheses that merely rename a QuickDraw class. Distinguish reusable stroke or "
        "part features from whole-class detectors. Explicitly report local geometry, contextual attachment, negative "
        "space, scale and orientation, and what evidence would falsify the interpretation.\n\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
    )


def adjudication_prompt(
    card: dict[str, Any],
    visual: dict[str, Any],
    quantitative: dict[str, Any],
) -> str:
    return (
        f"Adjudicate two independent analyses of SAE feature {card['feature_id']}. "
        "Return a short compositional human label using forms such as '<appearance> <part>', "
        "'<geometry or pose> <entity>', or '<entity> in <context>'. Stable identity is separate from wording. "
        "The label is a provisional hypothesis, not ground truth. Also produce a plain-English explanation, atomic "
        "visual primitives and a complete visual signature. The canonical label must be three to eight words and "
        "include the most discriminating geometric or attachment cue, not merely a generic word such as edge, blob, "
        "curve, or junction. Provide concrete trigger and non-trigger conditions, how meaning changes from strong to weak "
        "activation, and an explicit class-confounding assessment. Preserve alternatives, counterevidence, explicit "
        "falsifiers, synthetic concept/control prompts, and edit or occlusion tests. Use ambiguous or mixed status "
        "when the evidence is not clean.\n\nVISUAL ANALYST:\n"
        + json.dumps(visual, ensure_ascii=False, indent=2)
        + "\n\nQUANTITATIVE ANALYST:\n"
        + json.dumps(quantitative, ensure_ascii=False, indent=2)
    )


def validator_prompt(card: dict[str, Any], label: dict[str, Any]) -> str:
    tile_ids = [tile["tile_id"] for tile in card["blind_validation_tiles"]]
    return (
        "You are a held-out validator. The proposed feature label was formed without seeing these images. Every "
        "file is an original drawing with no activation overlay, box, score, or target leak. For every tile ID, "
        "predict from the proposed label alone how likely the learned feature is to activate somewhere in the "
        "drawing. Tile IDs and ordering reveal nothing about activation status. Return exactly one "
        f"score for each of these tile IDs: {tile_ids}.\n\nPROPOSED LABEL:\n"
        + json.dumps(label, ensure_ascii=False, indent=2)
    )


def score_validation(
    card: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    truth = {
        tile["tile_id"]: {
            "target": int(tile["target"]),
            "activation": float(tile["example"]["activation"]),
        }
        for tile in card["blind_validation_tiles"]
    }
    returned = {
        str(row["tile_id"]): float(row["label_fit"])
        for row in response.get("tile_scores", [])
    }
    common = [tile_id for tile_id in truth if tile_id in returned]
    targets = np.asarray([truth[tile_id]["target"] for tile_id in common], dtype=np.int64)
    scores = np.asarray([returned[tile_id] for tile_id in common], dtype=np.float32)
    auc = float(roc_auc_score(targets, scores)) if len(np.unique(targets)) == 2 else None
    positive_scores = scores[targets == 1]
    negative_scores = scores[targets == 0]
    p_value = (
        float(
            mannwhitneyu(
                positive_scores,
                negative_scores,
                alternative="greater",
                method="auto",
            ).pvalue
        )
        if len(positive_scores) and len(negative_scores)
        else None
    )
    return {
        "expected_tiles": len(truth),
        "scored_tiles": len(common),
        "coverage": len(common) / max(1, len(truth)),
        "auc": auc,
        "one_sided_mannwhitney_p": p_value,
        "positive_mean_fit": float(scores[targets == 1].mean()) if (targets == 1).any() else None,
        "negative_mean_fit": float(scores[targets == 0].mean()) if (targets == 0).any() else None,
    }


def feature_acceptance_decision(
    config: Config,
    card: dict[str, Any],
    label: dict[str, Any],
    validation: dict[str, Any],
    distinction: dict[str, Any] | None,
    manual_adjudication: dict[str, Any] | None,
) -> dict[str, Any]:
    tolerance = config.evaluation.gemini_gate_tolerance
    automatic_checks = {
        "heldout_auc": validation["auc"] is not None
        and validation["auc"] + tolerance
        >= config.evaluation.gemini_minimum_validation_auc,
        "complete_coverage": validation["coverage"] + tolerance >= 1.0,
        "minimum_confidence": float(label["confidence"]) + tolerance
        >= config.evaluation.gemini_minimum_confidence,
        "clean_provisional_status": label["status"] == "provisional",
        "minimum_positive_fit": validation["positive_mean_fit"] is not None
        and validation["positive_mean_fit"] + tolerance
        >= config.evaluation.gemini_minimum_positive_fit,
        "class_diversity": card["score_summary"]["class_confounding"][
            "effective_classes_by_activation_mass"
        ]
        + tolerance
        >= config.evaluation.gemini_minimum_effective_classes,
        "maximum_modal_class_share": card["score_summary"]["class_confounding"][
            "top_32_modal_class_share"
        ]
        <= config.evaluation.gemini_max_modal_class_share + tolerance,
        "llm_class_confounding": label["class_confounding_assessment"] != "high",
        "cross_feature_distinct": (distinction or {}).get("relationship")
        not in {"probable_duplicate", "uncertain"},
        "heldout_significance": validation.get("one_sided_mannwhitney_p") is not None
        and validation["one_sided_mannwhitney_p"]
        <= config.evaluation.gemini_maximum_validation_p_value + tolerance,
    }
    manual = manual_adjudication or {}
    automatic_checks["manual_not_rejected"] = manual.get("decision") not in {
        "reject",
        "control_only",
    }
    manual_checks = {
        "explicit_manual_acceptance": manual.get("decision") == "accept_provisional",
        "manual_minimum_confidence": float(manual.get("confidence", 0.0)) + tolerance
        >= config.evaluation.gemini_minimum_confidence,
        "heldout_auc": validation["auc"] is not None
        and validation["auc"] + tolerance
        >= config.evaluation.gemini_manual_minimum_validation_auc,
        "heldout_significance": automatic_checks["heldout_significance"],
        "complete_coverage": automatic_checks["complete_coverage"],
        "clean_provisional_status": automatic_checks["clean_provisional_status"],
        "minimum_positive_fit": automatic_checks["minimum_positive_fit"],
        "class_diversity": automatic_checks["class_diversity"],
        "maximum_modal_class_share": automatic_checks["maximum_modal_class_share"],
        "llm_class_confounding": automatic_checks["llm_class_confounding"],
        "cross_feature_distinct": automatic_checks["cross_feature_distinct"],
    }
    automatic_accepted = all(automatic_checks.values())
    manual_accepted = all(manual_checks.values())
    if automatic_accepted:
        acceptance_mode = "automatic"
        acceptance_checks = automatic_checks
    elif manual_accepted:
        acceptance_mode = "manual_adjudication"
        acceptance_checks = manual_checks
    else:
        acceptance_mode = "rejected"
        acceptance_checks = automatic_checks
    rejection_reasons = [
        name for name, passed in acceptance_checks.items() if not passed
    ]
    return {
        "validated_status": "accepted"
        if acceptance_mode != "rejected"
        else "rejected_or_ambiguous",
        "acceptance_mode": acceptance_mode,
        "acceptance_checks": acceptance_checks,
        "automatic_acceptance_checks": automatic_checks,
        "manual_acceptance_checks": manual_checks,
        "manual_adjudication": manual_adjudication,
        "rejection_reasons": rejection_reasons,
    }


def resolved_feature_label(
    label: dict[str, Any],
    manual_adjudication: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved = dict(label)
    manual = manual_adjudication or {}
    if manual.get("decision") != "accept_provisional":
        return resolved
    if manual.get("canonical_label"):
        resolved["canonical_label"] = manual["canonical_label"]
    if manual.get("plain_english_summary"):
        resolved["plain_english_summary"] = manual["plain_english_summary"]
    resolved["confidence"] = float(manual["confidence"])
    return resolved


def manual_adjudication_path(config: Config) -> Path:
    path = Path(config.evaluation.gemini_manual_adjudication_path)
    if path.is_absolute():
        return path
    return Path(config.project.run_dir) / path


def read_manual_adjudications(config: Config) -> dict[int, dict[str, Any]]:
    path = manual_adjudication_path(config)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["feature_id"]): row
        for row in payload.get("features", [])
    }


def write_blocked_manifest(
    config: Config,
    output_root: Path,
    reason: str,
    feature_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    feature_ids = {
        int(item["card"]["feature_id"]) for item in feature_inputs
    }
    cached_feature_ids = {
        "visual": set(),
        "quantitative": set(),
        "adjudicator": set(),
        "validator": set(),
    }
    cached_collision_requests = 0
    for cache_path in (output_root / "api_cache").glob("*.json"):
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        role = str(cached["fingerprint"]["role"])
        stage, separator, identifier = role.partition(":")
        if stage == "collision":
            cached_collision_requests += 1
        elif separator and stage in cached_feature_ids:
            feature_id = int(identifier)
            if feature_id in feature_ids:
                cached_feature_ids[stage].add(feature_id)
    completed_validations = cached_feature_ids["validator"]
    manifest = {
        "status": "blocked",
        "model": config.evaluation.gemini_model,
        "thinking_level": config.evaluation.gemini_thinking_level,
        "prompt_version": config.evaluation.gemini_prompt_version,
        "reason": reason,
        "features_requested": len(feature_ids),
        "features_pending": sorted(feature_ids - completed_validations),
        "labels_generated": len(cached_feature_ids["adjudicator"]),
        "cached_by_stage": {
            stage: len(ids) for stage, ids in cached_feature_ids.items()
        },
        "cached_collision_requests": cached_collision_requests,
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["path"] = str(path)
    return manifest


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def feature_uid(config: Config, card: dict[str, Any]) -> str:
    identity = {
        "project": config.project.name,
        "artifact_version": config.sae.artifact_version,
        "sae_seed": config.sae.primary_seed,
        "model_step": card["model_step"],
        "sae_kind": card["sae_kind"],
        "feature_id": card["feature_id"],
    }
    suffix = hashlib.sha256(canonical_json(identity)).hexdigest()[:16]
    return f"feature_{suffix}"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def evidence_image_name(path: Path) -> str:
    parents = list(path.parts[-3:])
    return "/".join(parents)
