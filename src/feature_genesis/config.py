from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from feature_genesis.paths import cub_dataset_dir, resolve_cub_root


class ProjectSection(BaseModel):
    name: str = "feature_genesis"
    run_dir: str = "runs/default"
    seed: int = 1729
    device: str = "cuda"


class ReproSection(BaseModel):
    strict: bool = True
    deterministic_algorithms: bool = True
    warn_only: bool = False
    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    cpu_threads: int = 1
    verify_environment: bool = True


class DataSection(BaseModel):
    name: Literal["cub200", "quickdraw", "toy_shapes"] = "cub200"
    root: str = Field(default_factory=lambda: str(cub_dataset_dir()))
    image_size: int = 160
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = False
    split_seed: int = 8191
    validation_per_class: int = 3
    bbox_margin: float = 0.1
    train_scale_min: float = 0.72
    train_scale_max: float = 1.0
    horizontal_flip_probability: float = 0.5
    color_jitter: float = 0.12
    toy_train_count: int = 512
    toy_test_count: int = 256
    quickdraw_train_limit: int = 0
    quickdraw_validation_limit: int = 0
    quickdraw_test_limit: int = 0
    quickdraw_rotation_degrees: float = 10.0
    quickdraw_translation_fraction: float = 0.08
    quickdraw_scale_min: float = 0.9
    quickdraw_scale_max: float = 1.1

    @field_validator("root", mode="before")
    @classmethod
    def normalize_root(cls, value: str | None) -> str:
        if value is None:
            return str(cub_dataset_dir())
        return str(value)

    @model_validator(mode="after")
    def resolve_dataset_root(self) -> "DataSection":
        if self.name == "cub200":
            self.root = resolve_cub_root(self.root)
        return self


class ModelSection(BaseModel):
    name: Literal["quickdraw_cnn", "quickdraw_resnet34_gn", "resnet18_gn", "tiny_cnn"] = "resnet18_gn"
    num_classes: int = 0
    hook_layer: str = "layer3"
    group_norm_groups: int = 32


class TrainSection(BaseModel):
    epochs: int = 120
    max_steps: int | None = None
    learning_rate: float = 0.08
    min_learning_rate: float = 1e-5
    warmup_steps: int = 400
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    anchor_every: int = 500
    validation_every: int = 500
    log_every: int = 20
    analysis_steps: list[int] = Field(default_factory=lambda: [0, 10, 25, 50, 100, 200, 400, 800])
    gradient_clip_norm: float | None = None
    amp_dtype: Literal["none", "bfloat16"] = "none"
    channels_last: bool = False


class ActivationSection(BaseModel):
    split: Literal["train", "validation", "test"] = "validation"
    evaluation_split: Literal["train", "validation", "test"] = "validation"
    max_images: int = 1200
    evaluation_max_images: int = 1200
    batch_size: int = 64
    dtype: Literal["float16", "float32"] = "float16"
    flatten_spatial: bool = True
    cache_dir: str = "activations"
    checkpoint_steps: list[int] = Field(default_factory=list)
    random_spatial_tokens_per_image: int = 0


class SAESection(BaseModel):
    kind: Literal["batch_topk", "cosine_batch_topk", "jumprelu", "matryoshka", "matching_pursuit", "topk"] = "batch_topk"
    expansion_factor: int = 16
    dictionary_size: int | None = None
    k: int = 16
    batch_size: int = 4096
    steps: int = 6000
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.0
    aux_weight: float = 0.03125
    aux_k: int = 256
    dead_steps: int = 200
    matryoshka_prefixes: list[int] = Field(default_factory=lambda: [256, 1024, 4096])
    matryoshka_weights: list[float] = Field(default_factory=lambda: [0.2, 0.3, 0.5])
    temporal_weight: float = 0.0
    threshold_ema: float = 0.99
    cosine_per_feature: bool = True
    jump_sparsity_coefficient: float = 0.001
    jump_initial_threshold: float = 0.1
    jump_initial_l0: int | None = None
    jump_bandwidth: float = 0.05
    jump_sparsity_warmup_steps: int = 500
    mp_steps: int = 16
    mp_residual_tolerance: float = 0.0
    input_normalization: Literal["none", "dataset_rms"] = "dataset_rms"
    initialization_samples: int = 32768
    checkpoint_every: int = 500
    gradient_clip_norm: float | None = 1.0
    artifact_version: str = "v2_repaired"
    seeds: list[int] = Field(default_factory=lambda: [0])
    primary_seed: int = 0
    warm_start_mode: Literal["none", "aligned_directions"] = "aligned_directions"
    warm_start_alignment_rows: int = 50000
    selection_fraction: float = 0.1
    calibration_fraction: float = 0.05
    evaluation_fraction: float = 0.1
    minimum_fraction_variance_explained: float = 0.8
    maximum_dead_fraction: float = 0.5
    minimum_mean_l0_fraction: float = 0.5
    maximum_mean_l0_fraction: float = 2.0
    minimum_cross_seed_cosine: float = 0.5
    minimum_cross_seed_match_fraction: float = 0.1
    minimum_cross_seed_match_count: int = 0
    minimum_feature_frequency: float = 0.0
    minimum_usable_features: int = 0
    minimum_image_frequency: float = 0.0
    maximum_image_frequency: float = 1.0
    minimum_contrastive_features: int = 0
    maximum_mutual_coherence: float = 1.0
    maximum_positive_decoder_cosine: float = 1.0

    @model_validator(mode="after")
    def validate_prefixes(self) -> "SAESection":
        if self.kind == "matryoshka" and len(self.matryoshka_prefixes) != len(self.matryoshka_weights):
            raise ValueError("matryoshka_prefixes and matryoshka_weights must have equal length")
        if not self.seeds:
            raise ValueError("sae.seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("sae.seeds must be unique")
        if self.primary_seed not in self.seeds:
            raise ValueError("sae.primary_seed must be included in sae.seeds")
        fractions = [self.selection_fraction, self.calibration_fraction, self.evaluation_fraction]
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("SAE data fractions must be between 0 and 1")
        if sum(fractions) >= 1:
            raise ValueError("SAE selection, calibration, and evaluation fractions must sum to less than 1")
        if not 0 <= self.minimum_cross_seed_match_fraction <= 1:
            raise ValueError("sae.minimum_cross_seed_match_fraction must be between 0 and 1")
        if self.minimum_cross_seed_match_count < 0:
            raise ValueError("sae.minimum_cross_seed_match_count must not be negative")
        if not 0 <= self.minimum_feature_frequency <= 1:
            raise ValueError("sae.minimum_feature_frequency must be between 0 and 1")
        if self.minimum_usable_features < 0:
            raise ValueError("sae.minimum_usable_features must not be negative")
        if not 0 <= self.minimum_image_frequency <= self.maximum_image_frequency <= 1:
            raise ValueError("SAE image-frequency bounds must satisfy 0 <= minimum <= maximum <= 1")
        if self.minimum_contrastive_features < 0:
            raise ValueError("sae.minimum_contrastive_features must not be negative")
        if not 0 <= self.maximum_mutual_coherence <= 1:
            raise ValueError("sae.maximum_mutual_coherence must be between 0 and 1")
        if not 0 <= self.maximum_positive_decoder_cosine <= 1:
            raise ValueError("sae.maximum_positive_decoder_cosine must be between 0 and 1")
        return self


class LineageSection(BaseModel):
    decoder_weight: float = 0.35
    activation_weight: float = 0.35
    concept_weight: float = 0.2
    continue_threshold: float = 0.72
    secondary_threshold: float = 0.56
    split_mass_threshold: float = 1.15
    death_frequency_threshold: float = 1e-5
    procrustes_samples: int = 50000
    maximum_alignment_residual_ratio: float = 0.25


class AttributionSection(BaseModel):
    feature_ids: list[int] = Field(default_factory=list)
    target_steps: list[int] = Field(default_factory=list)
    candidate_batches: int = 24
    exact_replay_batches: int = 8
    gradient_refresh_steps: int = 25
    replay_start_step: int | None = None
    replay_end_step: int | None = None
    intervention: Literal["mask_loss", "replace_same_class", "part_occlusion"] = "mask_loss"
    parameter_scope: Literal["all", "hook_block", "classifier"] = "hook_block"
    probe_images: int = 256
    directional_epsilon: float = 1e-3
    per_example_candidates: int = 8


class EvaluationSection(BaseModel):
    top_examples: int = 24
    max_features: int = 512
    minimum_activations: int = 20
    part_radius_fraction: float = 0.12
    report_dir: str = "reports"
    gemini_enabled: bool = True
    gemini_model: str = "gemini-3.6-flash"
    gemini_thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    gemini_env_path: str = ".env"
    gemini_prompt_version: str = "feature-genesis-multimodal-v2"
    gemini_parallel_requests: int = 3
    gemini_request_attempts: int = 6
    gemini_timeout_seconds: int = 180
    gemini_reuse_latest_role_cache: bool = False
    gemini_max_uncached_requests: int = 0
    gemini_max_run_cost_usd: float = 0.0
    gemini_media_resolution: Literal[
        "MEDIA_RESOLUTION_LOW",
        "MEDIA_RESOLUTION_MEDIUM",
        "MEDIA_RESOLUTION_HIGH",
    ] = "MEDIA_RESOLUTION_HIGH"
    gemini_max_features: int = 16
    gemini_feature_ids: list[int] = Field(default_factory=list)
    gemini_maximum_image_frequency: float = 0.8
    gemini_diversity_penalty: float = 0.15
    gemini_top_positives: int = 8
    gemini_borderline_positives: int = 4
    gemini_hard_negatives: int = 4
    gemini_validation_positives: int = 4
    gemini_validation_negatives: int = 4
    gemini_minimum_validation_auc: float = 0.65
    gemini_manual_minimum_validation_auc: float = 0.75
    gemini_maximum_validation_p_value: float = 0.05
    gemini_minimum_confidence: float = 0.6
    gemini_minimum_positive_fit: float = 0.5
    gemini_gate_tolerance: float = 1e-6
    gemini_manual_adjudication_path: str = "manual_adjudication.json"
    gemini_activation_bins: int = 6
    gemini_examples_per_bin: int = 2
    gemini_max_examples_per_class: int = 2
    gemini_max_modal_class_share: float = 0.5
    gemini_minimum_effective_classes: float = 3.0
    gemini_collision_enabled: bool = True
    gemini_collision_similarity_threshold: float = 0.72
    gemini_collision_max_group_size: int = 4
    gemini_collision_max_requests: int = 1
    gemini_collision_images_per_feature: int = 3
    region_relative_threshold: float = 0.35
    region_padding_fraction: float = 0.12
    region_minimum_fraction: float = 0.12


class CircuitSection(BaseModel):
    artifact_version: str = "v1_sparse_visual"
    layers: list[str] = Field(default_factory=lambda: ["layer1", "layer2", "layer3", "layer4"])
    target_layer: str = "layer2"
    target_feature_ids: list[int] = Field(default_factory=list)
    checkpoint_steps: list[int] = Field(default_factory=list)
    reuse_target_layer_sae: bool = True
    cache_split: Literal["train", "validation", "test"] = "train"
    cache_images: int = 12000
    evaluation_cache_images: int = 12000
    cache_batch_size: int = 1024
    spatial_tokens_per_image: int = 64
    sae_kind: Literal["topk"] = "topk"
    sae_expansion_factor: int = 4
    sae_k: int = 8
    sae_steps: int = 4000
    sae_batch_size: int = 8192
    sae_dictionary_size_by_layer: dict[str, int] = Field(
        default_factory=lambda: {"layer1": 256, "layer3": 1024, "layer4": 1024}
    )
    sae_k_by_layer: dict[str, int] = Field(
        default_factory=lambda: {"layer1": 4, "layer3": 12, "layer4": 16}
    )
    sae_steps_by_layer: dict[str, int] = Field(
        default_factory=lambda: {"layer1": 3000, "layer3": 4000, "layer4": 4000}
    )
    sae_batch_size_by_layer: dict[str, int] = Field(
        default_factory=lambda: {"layer1": 8192, "layer3": 4096, "layer4": 2048}
    )
    sae_initialization_samples: int = 32768
    discovery_scan_images: int = 4096
    discovery_positive_images: int = 64
    discovery_batch_size: int = 16
    max_nodes_per_layer: int = 16
    max_edges_per_target: int = 8
    pruning_node_counts: list[int] = Field(default_factory=lambda: [2, 4, 8, 12, 16])
    minimum_edge_attribution: float = 1e-7
    bootstrap_samples: int = 200
    bootstrap_confidence: float = 0.95
    validation_images: int = 64
    validation_edges: int = 16
    minimum_faithfulness: float = 0.8
    minimum_edge_sign_agreement: float = 0.75
    minimum_final_edge_similarity: float = 0.5
    persistence_checkpoints: int = 2
    attribution_candidate_batches: int = 12
    attribution_gradient_refresh_steps: int = 100
    attribution_probe_images: int = 64
    exact_replay_ranks: int = 1
    gemini_max_new_nodes: int = 24
    gemini_max_run_cost_usd: float = 5.0

    @model_validator(mode="after")
    def validate_circuit(self) -> "CircuitSection":
        if not self.layers:
            raise ValueError("circuit.layers must not be empty")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("circuit.layers must be unique")
        if self.target_layer not in self.layers:
            raise ValueError("circuit.target_layer must be included in circuit.layers")
        if self.sae_expansion_factor <= 0 or self.sae_k <= 0:
            raise ValueError("Circuit SAE expansion and k must be positive")
        mappings = [
            self.sae_dictionary_size_by_layer,
            self.sae_k_by_layer,
            self.sae_steps_by_layer,
            self.sae_batch_size_by_layer,
        ]
        if any(any(value <= 0 for value in mapping.values()) for mapping in mappings):
            raise ValueError("Per-layer circuit SAE settings must be positive")
        if self.max_nodes_per_layer <= 0 or self.max_edges_per_target <= 0:
            raise ValueError("Circuit graph limits must be positive")
        if (
            not self.pruning_node_counts
            or any(value <= 0 for value in self.pruning_node_counts)
            or sorted(set(self.pruning_node_counts)) != self.pruning_node_counts
            or self.pruning_node_counts[-1] > self.max_nodes_per_layer
        ):
            raise ValueError("Circuit pruning counts must be sorted, unique, positive, and within the graph limit")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("circuit.bootstrap_confidence must be between zero and one")
        if not 0 <= self.minimum_faithfulness <= 1:
            raise ValueError("circuit.minimum_faithfulness must be between zero and one")
        if not 0 <= self.minimum_edge_sign_agreement <= 1:
            raise ValueError("circuit.minimum_edge_sign_agreement must be between zero and one")
        if not 0 <= self.minimum_final_edge_similarity <= 1:
            raise ValueError("circuit.minimum_final_edge_similarity must be between zero and one")
        if self.persistence_checkpoints <= 0:
            raise ValueError("circuit.persistence_checkpoints must be positive")
        return self


class Config(BaseModel):
    project: ProjectSection = Field(default_factory=ProjectSection)
    reproducibility: ReproSection = Field(default_factory=ReproSection)
    data: DataSection = Field(default_factory=DataSection)
    model: ModelSection = Field(default_factory=ModelSection)
    train: TrainSection = Field(default_factory=TrainSection)
    activations: ActivationSection = Field(default_factory=ActivationSection)
    sae: SAESection = Field(default_factory=SAESection)
    lineage: LineageSection = Field(default_factory=LineageSection)
    attribution: AttributionSection = Field(default_factory=AttributionSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    circuit: CircuitSection = Field(default_factory=CircuitSection)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config.model_validate(data)


def save_resolved_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
