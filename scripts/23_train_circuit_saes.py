import json
from pathlib import Path

from feature_genesis.circuits.paths import (
    circuit_cache_root,
    circuit_layer_config,
    circuit_root,
    circuit_steps,
)
from feature_genesis.config import load_config
from feature_genesis.sae.io import sae_root
from feature_genesis.sae.trainer import train_continual_sae_sequence, train_sae
from feature_genesis.sae.validation import validate_sae_artifact


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
LAYERS = None
FINAL_CANDIDATE_FIRST = True
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    layers = LAYERS or config.circuit.layers
    all_results = {}
    for layer in layers:
        if layer == config.circuit.target_layer and config.circuit.reuse_target_layer_sae:
            print(f"reuse existing target-layer SAE sequence: {layer}")
            continue
        layer_config = circuit_layer_config(config, layer)
        roots = {
            step: circuit_cache_root(config, layer, step)
            for step in circuit_steps(config)
        }
        missing = [str(root) for root in roots.values() if not (root / "activations.npy").exists()]
        if missing:
            raise FileNotFoundError(f"Circuit activation caches are missing: {missing[:3]}")
        final_step = max(roots)
        if FINAL_CANDIDATE_FIRST:
            candidate = train_sae(
                layer_config,
                roots[final_step],
                final_step,
                config.circuit.sae_kind,
                sae_seed=layer_config.sae.primary_seed,
                role="independent",
                resume=REUSE_EXISTING,
            )
            candidate_gate = validate_sae_artifact(
                layer_config,
                sae_root(
                    layer_config,
                    final_step,
                    config.circuit.sae_kind,
                    layer_config.sae.primary_seed,
                    "independent",
                ),
                final_step,
                layer_config.sae.primary_seed,
                "independent",
            )
            if not candidate_gate["passed"]:
                failed = [row["name"] for row in candidate_gate["checks"] if not row["passed"]]
                raise RuntimeError(f"Circuit SAE candidate failed for {layer}: {failed}")
            print(candidate["path"])
        summaries = train_continual_sae_sequence(
            layer_config,
            roots,
            config.circuit.sae_kind,
            layer_config.sae.primary_seed,
            resume=REUSE_EXISTING,
        )
        gates = []
        for step in roots:
            gate = validate_sae_artifact(
                layer_config,
                sae_root(
                    layer_config,
                    step,
                    config.circuit.sae_kind,
                    layer_config.sae.primary_seed,
                    "continual",
                ),
                step,
                layer_config.sae.primary_seed,
                "continual",
            )
            gates.append(gate)
        alignment_failures = [
            int(summary["model_step"])
            for summary in summaries
            if summary.get("warm_start_alignment") is not None
            and float(summary["warm_start_alignment"]["heldout_residual_ratio"])
            > config.lineage.maximum_alignment_residual_ratio
        ]
        all_results[layer] = {
            "summaries": summaries,
            "gates": gates,
            "alignment_failures": alignment_failures,
            "passed": all(row["passed"] for row in gates) and not alignment_failures,
        }
        if not all_results[layer]["passed"]:
            failed_steps = [row["model_step"] for row in gates if not row["passed"]]
            raise RuntimeError(
                f"Circuit SAE sequence failed for {layer}; "
                f"artifact steps={failed_steps}, alignment steps={alignment_failures}"
            )
    path = circuit_root(config) / "sae_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
