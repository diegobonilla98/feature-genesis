from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_feature_dossiers(
    run_dir,
    evaluation_root,
    lineage_path,
    attribution_root,
    feature_quality_path,
    sae_summary_path,
    output_root,
):
    run_dir = Path(run_dir)
    evaluation_root = Path(evaluation_root)
    lineage_path = Path(lineage_path)
    attribution_root = Path(attribution_root)
    feature_quality_path = Path(feature_quality_path)
    sae_summary_path = Path(sae_summary_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lineage = read_json(lineage_path)
    quality = read_json(feature_quality_path)
    sae_summary = read_json(sae_summary_path)
    evaluation_paths = {
        int(read_json(path)["feature_id"]): path
        for path in sorted(
            (evaluation_root / "gemini").glob("feature_*_evaluation.json")
        )
    }
    evaluations = {
        feature_id: read_json(path)
        for feature_id, path in evaluation_paths.items()
    }
    rows = []
    for quality_row in quality["features"]:
        feature_id = int(quality_row["feature_id"])
        evaluation = evaluations.get(feature_id)
        if evaluation is not None:
            dossier = build_dossier(
                run_dir,
                evaluation_root,
                evaluation_paths[feature_id],
                evaluation,
                lineage_path,
                lineage,
                attribution_root,
            )
            dossier["quality"] = quality_row
            render = render_dossier
            metrics = evaluation["heldout_validation"]["metrics"]
            label = evaluation["label"]["canonical_label"]
            summary = evaluation["label"]["plain_english_summary"]
            status = evaluation["validated_status"]
            auc = metrics["auc"]
        else:
            dossier = build_candidate_dossier(
                feature_id,
                quality_row,
                lineage_path,
                lineage,
            )
            render = render_candidate_dossier
            label = f"Unlabeled candidate feature {feature_id}"
            summary = "Visual semantics have not yet been evaluated. Genesis and activation evolution are available."
            status = "candidate_unlabeled"
            auc = None
        feature_root = output_root / f"feature_{feature_id:05d}"
        feature_root.mkdir(parents=True, exist_ok=True)
        (feature_root / "dossier.json").write_text(
            json.dumps(dossier, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (feature_root / "index.html").write_text(
            render(dossier, feature_root, output_root),
            encoding="utf-8",
        )
        rows.append(
            {
                "feature_id": feature_id,
                "label": label,
                "summary": summary,
                "status": status,
                "auc": auc,
                "candidate_score": float(quality_row["candidate_score"]),
                "active_image_frequency": float(
                    quality_row["active_image_frequency"]
                ),
                "birth_step": dossier["ancestry"]["structural_birth_step"],
                "phase_status": dossier["phase"]["status"],
                "causal_records": len(dossier.get("causal_records", [])),
                "intervention_available": bool(dossier.get("interventions")),
                "report": str(feature_root / "index.html"),
            }
        )
    feature_frequency = sae_summary["metrics"]["feature_frequency"]
    dead_features = [
        feature_id
        for feature_id, frequency in enumerate(feature_frequency)
        if float(frequency) <= 0
    ]
    candidate_ids = {int(row["feature_id"]) for row in quality["features"]}
    evaluated_ids = set(evaluations)
    accepted_ids = {
        feature_id
        for feature_id, evaluation in evaluations.items()
        if evaluation["validated_status"] == "accepted"
    }
    inventory = {
        "dictionary_slots": len(feature_frequency),
        "live_features": len(feature_frequency) - len(dead_features),
        "dead_features": len(dead_features),
        "explorable_candidates": len(candidate_ids),
        "gemini_evaluated": len(evaluated_ids),
        "validated": len(accepted_ids),
        "evaluated_rejected_or_control": len(evaluated_ids - accepted_ids),
        "unlabeled_candidates": len(candidate_ids - evaluated_ids),
        "other_live_features": len(feature_frequency)
        - len(dead_features)
        - len(candidate_ids),
        "developmental_feature_instances": len(feature_frequency)
        * 18
        * 3,
    }
    manifest = {
        "schema_version": 1,
        "feature_count": len(rows),
        "inventory": inventory,
        "features": rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index_path = output_root / "index.html"
    index_path.write_text(render_index(rows, inventory), encoding="utf-8")
    return index_path


def build_candidate_dossier(feature_id, quality, lineage_path, lineage):
    return {
        "schema_version": 1,
        "feature_id": feature_id,
        "evaluation": None,
        "card": None,
        "quality": quality,
        "trajectory": lineage["trajectories"][str(feature_id)],
        "phase": lineage["phases"][str(feature_id)],
        "ancestry": lineage["strongest_ancestry"][str(feature_id)],
        "training_influence": [],
        "causal_records": [],
        "intervention": None,
        "interventions": {},
        "sources": {
            "feature_quality": None,
            "lineage": str(lineage_path),
        },
    }


def build_dossier(
    run_dir,
    evaluation_root,
    evaluation_path,
    evaluation,
    lineage_path,
    lineage,
    attribution_root,
):
    feature_id = int(evaluation["feature_id"])
    card_path = evaluation_root / "cards" / f"feature_{feature_id:05d}.json"
    influence_path = (
        attribution_root
        / f"feature_{feature_id:05d}"
        / "training_influence_manifest.json"
    )
    causal_path = attribution_root / "causal_validation_summary.json"
    intervention_root = (
        run_dir
        / "evaluation"
        / f"step_{int(evaluation['model_step']):08d}"
        / evaluation["sae_kind"]
        / evaluation["artifact_version"]
        / "continual"
        / f"seed_{int(evaluation['sae_seed']):04d}"
        / "interventions"
    )
    intervention_paths = {
        mode: intervention_root / f"feature_{feature_id:05d}_{mode}.json"
        for mode in ["ablate", "steer"]
    }
    interventions = {
        mode: read_json(path)
        for mode, path in intervention_paths.items()
        if path.exists()
    }
    causal = read_json(causal_path) if causal_path.exists() else {"records": []}
    return {
        "schema_version": 1,
        "feature_id": feature_id,
        "evaluation": evaluation,
        "card": read_json(card_path),
        "trajectory": lineage["trajectories"][str(feature_id)],
        "phase": lineage["phases"][str(feature_id)],
        "ancestry": lineage["strongest_ancestry"][str(feature_id)],
        "training_influence": (
            read_json(influence_path) if influence_path.exists() else []
        ),
        "causal_records": [
            row
            for row in causal["records"]
            if int(row["feature_id"]) == feature_id
        ],
        "intervention": (
            interventions.get("ablate")
        ),
        "interventions": interventions,
        "emergence_assessment": emergence_assessment(
            evaluation,
            lineage["strongest_ancestry"][str(feature_id)],
            lineage["phases"][str(feature_id)],
        ),
        "sources": {
            "evaluation": str(evaluation_path),
            "card": str(card_path),
            "lineage": str(lineage_path),
            "training_influence": (
                str(influence_path) if influence_path.exists() else None
            ),
            "causal_validation": (
                str(causal_path) if causal_path.exists() else None
            ),
            "interventions": {
                mode: str(path)
                for mode, path in intervention_paths.items()
                if path.exists()
            },
        },
    }


def emergence_assessment(evaluation, ancestry, phase):
    edges = ancestry["edges"]
    edge_scores = [float(edge["score"]) for edge in edges]
    structural_birth = int(ancestry["structural_birth_step"])
    first_reliable = int(phase["measurement_window"][0])
    metrics = evaluation["heldout_validation"]["metrics"]
    return {
        "semantic_onset_status": phase["status"],
        "earliest_structural_trace_step": structural_birth,
        "concept_observed_no_later_than_step": first_reliable,
        "semantic_onset_interval": [structural_birth, first_reliable],
        "consolidation_interval": phase.get("consolidation"),
        "semantic_validation_auc": metrics.get("auc"),
        "semantic_validation_coverage": metrics.get("coverage"),
        "gemini_label_confidence": evaluation["llm_label"].get("confidence"),
        "lineage_mean_edge_score": (
            sum(edge_scores) / len(edge_scores) if edge_scores else 1.0
        ),
        "lineage_weakest_edge_score": min(edge_scores) if edge_scores else 1.0,
        "lineage_merges": sum(edge["relation"] == "merge" for edge in edges),
        "interpretation": (
            f"A connected structural precursor is traceable from step {structural_birth:,}. "
            f"The final concept is measurement-reliable by step {first_reliable:,}, but "
            "semantic onset inside that interval is left-censored."
        ),
    }


def render_candidate_dossier(dossier, report_root, output_root):
    feature_id = dossier["feature_id"]
    quality = dossier["quality"]
    ancestry = dossier["ancestry"]
    phase = dossier["phase"]
    trajectory = dossier["trajectory"]
    reliable = [row for row in trajectory if row["alignment_reliable"]]
    reliable_start = reliable[0]["step"] if reliable else None
    birth_step = int(ancestry["structural_birth_step"])
    body = f"""
<header class="hero"><a class="back" href="{relative_uri(output_root / 'index.html', report_root)}">← Full feature atlas</a><div class="eyebrow">Feature {feature_id:05d} · explorable candidate</div><h1>Unlabeled candidate feature {feature_id}</h1><p class="lede">This live SAE feature passed the quantitative exploration filter, but its visual meaning has not yet been exported or tested by Gemini.</p><div class="badges"><span class="badge warning">unlabeled candidate</span><span class="badge">no Gemini tokens spent</span></div></header>
<main><section><h2>What is known now</h2><div class="metrics">
{metric('Candidate score', f"{float(quality['candidate_score']):.3f}", 'ranking score for visual exploration')}
{metric('Image frequency', f"{float(quality['active_image_frequency']):.1%}", f"{int(quality['active_images']):,} of 48,000 images")}
{metric('Median active', f"{float(quality['median_when_active']):.3f}", 'image-maximum activation')}
{metric('99th percentile', f"{float(quality['p99_when_active']):.3f}", 'upper activation tail')}
{metric('Effective classes', f"{float(quality['effective_classes_by_activation_mass']):.1f}", 'activation-mass diversity')}
</div><div class="callout warning"><strong>No semantic claim yet.</strong> A high candidate score means this feature is worth inspecting, not that it has a coherent human concept. Positive/negative images and held-out Gemini validation must come next.</div></section>
<section><h2>Genesis and structural evolution</h2><div class="metrics">{metric('Structural ancestor', f"step {birth_step:,}", f"feature {int(ancestry['structural_birth_feature_id'])}")}{metric('Attribution target', f"step {int(ancestry['attribution_target_step']):,}", f"feature {int(ancestry['attribution_target_feature_id'])}")}{metric('Reliable window', f"{int(phase['measurement_window'][0]):,}–{int(phase['measurement_window'][1]):,}", 'alignment residual ≤ 0.25')}{metric('Phase status', phase['status'].replace('_', ' '), 'semantic onset is not established')}</div><p>The strongest-edge path is structurally traceable from step {birth_step:,}. The connected precursor may not share the final feature's eventual visual meaning.</p><div class="ancestry">{render_ancestry(ancestry)}</div></section>
<section><h2>Evolution of the aligned response</h2><p class="note">Grey projections fail the alignment gate and are excluded from phase inference. Blue measurements are reliable.</p><div class="chart-grid">{line_chart(trajectory, 'mean_positive_response', 'Mean response when active', 'Sparse-code response among active locations', birth_step, reliable_start, False)}{line_chart(trajectory, 'positive_frequency', 'Positive spatial frequency', 'Fraction of spatial locations with nonzero activity', birth_step, reliable_start, True)}</div><div class="callout neutral"><strong>Phase interpretation.</strong> {escape(phase_text(phase))}</div></section>
<section><h2>What is missing before this becomes a concept report</h2><ol class="ladder"><li>Export diverse strongest positives, activation-spectrum examples, and matched inactive controls.</li><li>Generate localized original/box/zoom evidence.</li><li>Ask Gemini for a falsifiable visual hypothesis.</li><li>Score that hypothesis on unseen positive and negative images.</li><li>Run exact training attribution and causal replay only after semantic acceptance.</li></ol></section>
<details class="sources"><summary>Source artifacts and reproducibility</summary><ul><li><code>{escape(dossier['sources']['lineage'])}</code></li></ul></details></main>"""
    return document(f"Candidate feature {feature_id}", body)


def render_dossier(dossier, report_root, output_root):
    feature_id = dossier["feature_id"]
    evaluation = dossier["evaluation"]
    label = evaluation["label"]
    metrics = evaluation["heldout_validation"]["metrics"]
    card = dossier["card"]
    score = card["score_summary"]
    ancestry = dossier["ancestry"]
    phase = dossier["phase"]
    trajectory = dossier["trajectory"]
    reliable = [row for row in trajectory if row["alignment_reliable"]]
    reliable_start = reliable[0]["step"] if reliable else None
    birth_step = int(ancestry["structural_birth_step"])
    positive_gallery = gallery(
        card["individual_evidence"]["discovery"],
        report_root,
        "top_positive",
        6,
    )
    negative_gallery = gallery(
        card["individual_evidence"]["discovery"],
        report_root,
        "hard_negative",
        4,
    )
    status_class = (
        "accepted" if evaluation["validated_status"] == "accepted" else "warning"
    )
    sources = "".join(
        f"<li><code>{escape(value)}</code></li>"
        for value in dossier["sources"].values()
        if value is not None
    )
    body = f"""
<header class="hero">
<a class="back" href="{relative_uri(output_root / 'index.html', report_root)}">← All feature dossiers</a>
<div class="eyebrow">Feature {feature_id:05d} · model step {int(evaluation['model_step']):,}</div>
<h1>{escape(label['canonical_label'])}</h1>
<p class="lede">{escape(label['plain_english_summary'])}</p>
<div class="badges"><span class="badge {status_class}">{escape(evaluation['validated_status'])}</span><span class="badge">{escape(evaluation['acceptance_mode'].replace('_', ' '))}</span><span class="badge">Gemini {escape(evaluation['model_metadata']['version'])}</span></div>
</header>
<main>
<section><h2>Technical summary</h2><p>{escape(label['description'])}</p>
<div class="metrics">
{metric('Held-out AUC', format_decimal(metrics['auc']), '12 unseen positives vs 12 matched inactive controls')}
{metric('One-sided p', format_probability(metrics['one_sided_mannwhitney_p']), 'Mann–Whitney label-fit separation')}
{metric('Positive fit', format_decimal(metrics['positive_mean_fit']), 'Mean fit on unseen active images')}
{metric('Negative fit', format_decimal(metrics['negative_mean_fit']), 'Lower is better on matched controls')}
{metric('Image frequency', f"{float(score['activation_frequency_per_image']):.1%}", f"{int(score['active_images']):,} of 48,000 validation images")}
</div><div class="callout"><strong>Claim boundary.</strong> The label is a held-out visual hypothesis. Structural ancestry tracks SAE identity, not guaranteed semantic identity. Builder or breaker language appears only after exact replay.</div></section>
<section><h2>What switches this feature on</h2><div class="two-col"><div><h3>Trigger conditions</h3>{bullet_list(label['trigger_conditions'])}</div><div><h3>Non-trigger conditions</h3>{bullet_list(label['non_trigger_conditions'])}</div></div><div class="two-col compact"><div><h3>Visual signature</h3>{definition_list(label['visual_signature'])}</div><div><h3>Do not conflate with</h3>{bullet_list(label['do_not_conflate_with'])}</div></div></section>
<section><h2>Visual evidence</h2><p class="note">Positive panels show original, localized response box, and zoom. Controls are matched zero-activation drawings.</p><h3>Strongest triggering examples</h3><div class="gallery">{positive_gallery}</div><h3>Matched inactive controls</h3><div class="gallery controls">{negative_gallery}</div></section>
<section><h2>Genesis: a traceable ancestor, not a semantic birth certificate</h2><div class="metrics">
{metric('Structural ancestor', f"step {birth_step:,}", f"feature {int(ancestry['structural_birth_feature_id'])}")}
{metric('Attribution target', f"step {int(ancestry['attribution_target_step']):,}", f"feature {int(ancestry['attribution_target_feature_id'])}")}
{metric('Reliable window', f"{int(phase['measurement_window'][0]):,}–{int(phase['measurement_window'][1]):,}", 'alignment residual ≤ 0.25')}
{metric('Phase status', phase['status'].replace('_', ' '), 'earlier projections excluded')}
</div><p>The strongest incoming-edge rule traces the final latent backward to step {birth_step:,}. Activation measurements become trustworthy only at step {int(reliable_start):,}. Since the connected ancestor predates that window, emergence is <strong>left-censored</strong>; this run cannot identify semantic onset more precisely.</p><div class="ancestry">{render_ancestry(ancestry)}</div></section>
{render_emergence_assessment(dossier['emergence_assessment'])}
<section><h2>Evolution of the aligned response</h2><p class="note">Grey points fail the alignment-residual gate and are context only. Blue points are admitted into phase inference.</p><div class="chart-grid">{line_chart(trajectory, 'mean_positive_response', 'Mean response when active', 'Sparse-code response among active locations', birth_step, reliable_start, False)}{line_chart(trajectory, 'positive_frequency', 'Positive spatial frequency', 'Fraction of spatial locations with nonzero activity', birth_step, reliable_start, True)}</div><div class="callout neutral"><strong>Phase interpretation.</strong> {escape(phase_text(phase))}</div></section>
{render_causal(dossier, report_root)}
{render_interventions(dossier)}
<section><h2>How to falsify this interpretation</h2><div class="two-col"><div><h3>Recommended edits</h3>{bullet_list(label['edit_tests'])}</div><div><h3>Direct falsification tests</h3>{bullet_list(label['falsification_tests'])}</div></div><div class="callout warning"><strong>Remaining uncertainty.</strong> Semantic identity across early checkpoints is unverified. Without exact replay, no training image is called a builder or breaker.</div></section>
<details class="sources"><summary>Source artifacts and reproducibility</summary><ul>{sources}</ul><p>Generated from saved artifacts only. No Gemini calls are made.</p></details>
</main><dialog id="viewer"><button id="close">×</button><img id="viewer-image" alt="Expanded evidence"></dialog>"""
    return document(
        f"Feature {feature_id}: {label['canonical_label']}",
        body,
    )


def render_index_legacy(rows):
    cards = []
    for row in rows:
        feature_id = int(row["feature_id"])
        causal = (
            f"{int(row['causal_records'])} replay results"
            if row["causal_records"]
            else "causal replay not yet run"
        )
        cards.append(
            f"<a class=\"feature-card\" href=\"feature_{feature_id:05d}/index.html\"><div class=\"eyebrow\">Feature {feature_id:05d}</div><h2>{escape(row['label'])}</h2><p>{escape(row['summary'])}</p><div class=\"stats\"><span>AUC {float(row['auc']):.3f}</span><span>ancestor step {int(row['birth_step']):,}</span><span>{escape(causal)}</span></div></a>"
        )
    body = f"""
<header class="hero index-hero"><div class="eyebrow">QuickDraw · ResNet-34 GN · v6 TopK SAE</div><h1>Feature Genesis Dossiers</h1><p class="lede">One evidence-led report per validated feature: visual trigger, counterexamples, structural ancestry, measurable evolution, training influence, causal replay, and uncertainty.</p></header>
<main><section><h2>Validated feature worlds</h2><p class="note">A structural ancestor is the earliest node on the strongest-edge path. It is not automatically the moment the human-readable concept emerged.</p><div class="feature-grid">{''.join(cards)}</div></section>
<section><h2>Evidence ladder used on every page</h2><ol class="ladder"><li><strong>Visual hypothesis:</strong> localized positives and matched inactive controls.</li><li><strong>Held-out validation:</strong> Gemini scores unseen images without activation targets.</li><li><strong>Structural evolution:</strong> strongest-edge lineage and reliability-gated measurements.</li><li><strong>Training attribution:</strong> clipped gradient alignment proposes candidates only.</li><li><strong>Causal validation:</strong> exact single-step masking assigns builder or breaker status.</li><li><strong>Functional effect:</strong> final-model SAE ablation measures downstream sensitivity.</li></ol></section></main>"""
    return document("Feature Genesis Dossiers", body)


def render_index(rows, inventory):
    validated = [row for row in rows if row["status"] == "accepted"]
    evaluated_rejected = [
        row
        for row in rows
        if row["status"] not in {"accepted", "candidate_unlabeled"}
    ]
    cards = []
    for row in validated[:12]:
        feature_id = int(row["feature_id"])
        causal = (
            f"{int(row['causal_records'])} replay results"
            if row["causal_records"]
            else "causal replay not yet run"
        )
        cards.append(
            f"<a class=\"feature-card\" href=\"feature_{feature_id:05d}/index.html\"><div class=\"eyebrow\">Feature {feature_id:05d}</div><h2>{escape(row['label'])}</h2><p>{escape(row['summary'])}</p><div class=\"stats\"><span>AUC {float(row['auc']):.3f}</span><span>ancestor step {int(row['birth_step']):,}</span><span>{escape(causal)}</span></div></a>"
        )
    rejected_cards = []
    for row in evaluated_rejected[:12]:
        feature_id = int(row["feature_id"])
        rejected_cards.append(
            f"<a class=\"compact-card\" href=\"feature_{feature_id:05d}/index.html\"><span>Feature {feature_id:05d}</span><strong>{escape(row['label'])}</strong><small>AUC {float(row['auc']):.3f} · rejected or control</small></a>"
        )
    table_rows = []
    for row in rows:
        feature_id = int(row["feature_id"])
        status_label = {
            "accepted": "validated",
            "candidate_unlabeled": "unlabeled",
        }.get(row["status"], "rejected")
        label = (
            row["label"]
            if row["status"] != "candidate_unlabeled"
            else "Meaning not evaluated"
        )
        auc = "—" if row["auc"] is None else f"{float(row['auc']):.3f}"
        search_value = escape(f"{feature_id} {label} {status_label}")
        table_rows.append(
            f"<tr data-search=\"{search_value}\" data-status=\"{status_label}\"><td><a href=\"feature_{feature_id:05d}/index.html\">{feature_id:05d}</a></td><td>{escape(label)}</td><td><span class=\"status {status_label}\">{status_label}</span></td><td>{float(row['candidate_score']):.3f}</td><td>{float(row['active_image_frequency']):.1%}</td><td>{auc}</td><td>{int(row['birth_step']):,}</td></tr>"
        )
    body = f"""
<style>.compact-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.compact-card{{display:flex;flex-direction:column;gap:7px;padding:16px;border:1px solid var(--line);border-radius:11px;color:var(--ink);text-decoration:none}}.compact-card span,.compact-card small{{color:var(--muted);font-size:.75rem}}.atlas-controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:20px 0}}.atlas-controls input,.atlas-controls select{{border:1px solid var(--line);border-radius:9px;padding:11px 13px;background:var(--paper);color:var(--ink);font:inherit}}.atlas-controls input{{min-width:260px;flex:1}}.atlas-controls span{{color:var(--muted);font-size:.82rem}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;max-height:660px}}table{{border-collapse:collapse;width:100%;font-size:.82rem}}th{{position:sticky;top:0;background:var(--paper);z-index:1;text-align:left;color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.06em}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap}}td:nth-child(2){{white-space:normal;min-width:220px}}td a{{color:var(--blue);font-family:ui-monospace,monospace;font-weight:700}}.status{{display:inline-block;padding:3px 7px;border:1px solid var(--line);border-radius:999px;font-size:.66rem;text-transform:uppercase}}.status.validated{{color:var(--blue);background:var(--blue-soft)}}.status.rejected{{color:var(--gold);background:var(--gold-soft)}}tr[hidden]{{display:none}}@media(max-width:800px){{.compact-grid{{grid-template-columns:1fr}}}}</style>
<header class="hero index-hero"><div class="eyebrow">QuickDraw · ResNet-34 GN · v6 TopK SAE</div><h1>The complete feature atlas</h1><p class="lede">The final dictionary has {inventory['dictionary_slots']:,} slots. {inventory['validated']:,} human-readable concepts are validated; {inventory['unlabeled_candidates']:,} explorable candidates remain semantically unlabeled.</p></header>
<main><section><h2>What was actually trained</h2><div class="metrics">{metric('Final dictionary', f"{inventory['dictionary_slots']:,}", 'feature slots at step 21,975')}{metric('Live features', f"{inventory['live_features']:,}", f"{inventory['dead_features']} dead slots")}{metric('Explorable candidates', f"{inventory['explorable_candidates']:,}", 'passed frequency, diversity, and contrast filters')}{metric('Gemini evaluated', f"{inventory['gemini_evaluated']:,}", f"{inventory['validated']} validated")}{metric('Developmental instances', f"{inventory['developmental_feature_instances']:,}", '512 slots × 18 checkpoints × 3 seeds')}</div><div class="callout"><strong>Why the earlier page showed only three.</strong> It was filtered to semantically validated features. That hid the distinction between trained SAE slots, explorable candidates, Gemini-evaluated hypotheses, and accepted concepts. This atlas keeps those levels separate.</div></section>
<section><h2>{len(validated):,} validated feature worlds</h2><p class="note">These passed localized visual review and held-out positive-versus-control scoring. Up to twelve are highlighted here; the complete set is searchable below.</p><div class="feature-grid">{''.join(cards)}</div></section>
<section><h2>{len(evaluated_rejected):,} evaluated hypotheses that did not pass</h2><p class="note">They remain inspectable, but the proposed wording is not treated as a discovered concept. Up to twelve are highlighted here.</p><div class="compact-grid">{''.join(rejected_cards)}</div></section>
<section><h2>All {inventory['explorable_candidates']:,} explorable candidates</h2><p class="note">Every row links to an automatic structural-genesis and response-evolution report. Unlabeled rows have not consumed Gemini tokens and make no semantic claim.</p><div class="atlas-controls"><input id="feature-search" type="search" placeholder="Search feature id or label"><select id="status-filter"><option value="all">All statuses</option><option value="validated">Validated</option><option value="rejected">Rejected/control</option><option value="unlabeled">Unlabeled</option></select><span id="visible-count">{inventory['explorable_candidates']:,} shown</span></div><div class="table-wrap"><table><thead><tr><th>Feature</th><th>Meaning</th><th>Status</th><th>Candidate score</th><th>Image frequency</th><th>AUC</th><th>Ancestor step</th></tr></thead><tbody id="atlas-body">{''.join(table_rows)}</tbody></table></div></section>
<section><h2>What remains outside the candidate atlas</h2><p><strong>{inventory['other_live_features']} other live slots</strong> did not pass the current exploration filters, usually because they were too rare, too ubiquitous, insufficiently contrastive, or less stable. <strong>{inventory['dead_features']} slots are dead.</strong> They were trained and remain part of the 512-slot dictionary, but presenting them as human concepts would be misleading.</p></section>
<section><h2>Evidence ladder</h2><ol class="ladder"><li><strong>Trained slot:</strong> one coordinate in a checkpoint-specific SAE dictionary.</li><li><strong>Live feature:</strong> activates in the final evaluation cache.</li><li><strong>Explorable candidate:</strong> passes quantitative frequency, contrast, and class-diversity filters.</li><li><strong>Evaluated hypothesis:</strong> receives visual evidence and a Gemini label proposal.</li><li><strong>Validated concept:</strong> predicts unseen active images over matched controls.</li><li><strong>Causally studied feature:</strong> training examples survive exact masking and replay.</li></ol></section></main>
<script>const search=document.getElementById('feature-search');const filter=document.getElementById('status-filter');const atlasRows=[...document.querySelectorAll('#atlas-body tr')];const count=document.getElementById('visible-count');function applyAtlasFilter(){{const query=search.value.trim().toLowerCase();const status=filter.value;let visible=0;atlasRows.forEach(row=>{{const matchesText=!query||row.dataset.search.toLowerCase().includes(query);const matchesStatus=status==='all'||row.dataset.status===status;row.hidden=!(matchesText&&matchesStatus);if(!row.hidden)visible++}});count.textContent=`${{visible}} shown`}}search.addEventListener('input',applyAtlasFilter);filter.addEventListener('change',applyAtlasFilter);</script>"""
    return document("Complete Feature Atlas", body)


def render_emergence_assessment(assessment):
    onset = assessment["semantic_onset_interval"]
    consolidation = assessment.get("consolidation_interval")
    consolidation_text = (
        f"{int(consolidation[0]):,}–{int(consolidation[1]):,}"
        if consolidation is not None
        else "not resolved"
    )
    cards = "".join(
        [
            metric("Earliest structural trace", f"step {int(onset[0]):,}", "connected precursor, not guaranteed semantics"),
            metric("Concept present by", f"step {int(onset[1]):,}", "first reliable aligned measurement"),
            metric("Semantic sureness", format_decimal(assessment["semantic_validation_auc"]), "held-out label AUC"),
            metric("Weakest lineage edge", format_decimal(assessment["lineage_weakest_edge_score"]), "minimum strongest-path similarity"),
            metric("Mean lineage continuity", format_decimal(assessment["lineage_mean_edge_score"]), f"{int(assessment['lineage_merges'])} merge transitions"),
            metric("Consolidation window", consolidation_text, "sustained ≥60% of peak response"),
        ]
    )
    return f"<section><h2>When did this concept emerge—and how sure are we?</h2><div class=\"metrics\">{cards}</div><p>{escape(assessment['interpretation'])}</p><div class=\"callout warning\"><strong>Onset interval, not a point estimate.</strong> The experiment can bound semantic emergence to steps {int(onset[0]):,}–{int(onset[1]):,}. It cannot prove which update inside that interval created the human-readable concept.</div></section>"


def render_causal(dossier, report_root):
    records = dossier["causal_records"]
    if not records:
        if dossier["evaluation"]["validated_status"] != "accepted":
            return "<section><h2>Causal replay: not scheduled</h2><p>This proposed interpretation failed held-out semantic validation, so it was not promoted into the expensive exact-replay experiment. No training image is claimed as a builder or breaker for this rejected hypothesis.</p></section>"
        return "<section><h2>Causal replay: processing</h2><p>This concept passed semantic validation, but its exact builder and breaker replays have not both completed yet. The final report will populate this section automatically when the causal pipeline finishes.</p></section>"
    cards = []
    for record in sorted(
        records,
        key=lambda row: (row["candidate_step"], row["predicted_direction"]),
    ):
        confirmed = bool(record["predicted_direction_confirmed"])
        verdict = "confirmed" if confirmed else "disconfirmed"
        tone = "accepted" if confirmed else "warning"
        cards.append(
            f"<article class=\"causal-card\"><span class=\"badge {tone}\">{escape(record['predicted_direction'].removesuffix('s'))} prediction {verdict}</span><h3>Step {int(record['candidate_step']):,} is a causal {escape(record['causal_role'])}</h3><p>Presence effect <strong>{float(record['causal_presence_effect']):+.4f}</strong>; removing its eight selected examples changed the birth-node response by <strong>{float(record['relative_removal_change']):+.1%}</strong>.</p><p class=\"fine\">Predicted influence {float(record['predicted_influence']):+.4f} · baseline hash verified · masked only at step {int(record['candidate_step']):,}</p></article>"
        )
    groups = {}
    for row in dossier["training_influence"]:
        if row.get("causal_status") != "validated":
            continue
        key = (row["predicted_direction"], int(row["step"]))
        groups.setdefault(key, []).append(row)
    galleries = []
    for (direction, step), group in sorted(groups.items()):
        record = next(
            row
            for row in records
            if row["predicted_direction"] == direction
            and int(row["candidate_step"]) == step
        )
        figures = []
        for row in group[:8]:
            uri = relative_uri(Path(row["image"]), report_root)
            figures.append(
                f"<figure><button class=\"image-button\" data-full=\"{uri}\"><img src=\"{uri}\" alt=\"{escape(row['class_name'])} training drawing\"></button><figcaption>{escape(row['class_name'])}<br><span>index {int(row['dataset_index']):,} · proxy {float(row['example_influence']):+.3g}</span></figcaption></figure>"
            )
        galleries.append(
            f"<h3>Step {step:,}: predicted {escape(direction.removesuffix('s'))}, causally {escape(record['causal_role'])}</h3><div class=\"gallery training\">{''.join(figures)}</div>"
        )
    builders = sum(record["causal_role"] == "builder" for record in records)
    breakers = sum(record["causal_role"] == "breaker" for record in records)
    neutral = len(records) - builders - breakers
    return f"<section><h2>Which training drawings helped, confused, or failed to matter?</h2><p>Gradient alignment selected one likely-helpful and one likely-harmful batch for exact replay. Each gallery is a jointly masked group: the group receives a causal role, while the per-image proxy shown below is ranking evidence rather than individual causality.</p><div class=\"causal-grid\">{''.join(cards)}</div>{''.join(galleries)}<div class=\"callout neutral\"><strong>Measured outcome.</strong> {builders} tested groups helped build the response, {breakers} suppressed or confused it, and {neutral} were neutral. A disconfirmed prediction remains useful negative evidence about the gradient proxy.</div></section>"


def render_interventions(dossier):
    interventions = dossier.get("interventions", {})
    if not interventions:
        if dossier["evaluation"]["validated_status"] != "accepted":
            return "<section><h2>Downstream interventions: not scheduled</h2><p>This semantic hypothesis was rejected before causal promotion, so feature removal and steering were intentionally not run for it.</p></section>"
        return "<section><h2>Downstream interventions: processing</h2><p>This accepted concept is queued for final-model removal and steering after exact replay completes. The report will populate this section automatically.</p></section>"
    blocks = []
    for mode in ["ablate", "steer"]:
        intervention = interventions.get(mode)
        if intervention is None:
            continue
        action = "Remove the encoded contribution" if mode == "ablate" else "Add one decoder-direction unit"
        blocks.append(render_intervention_block(intervention, action))
    return f"<section><h2>What changes when this feature is removed or amplified?</h2><p class=\"note\">Both interventions use the same 512 held-out validation drawings. Removal subtracts the feature's reconstructed contribution; steering adds its decoder direction.</p>{''.join(blocks)}</section>"


def render_intervention_block(intervention, action):
    accuracy_delta = float(
        intervention["intervened_accuracy"] - intervention["baseline_accuracy"]
    )
    flip_rate = f"{float(intervention['prediction_flip_rate']):.2%}"
    image_count = f"{int(intervention['images']):,} validation images"
    accuracy_change = f"{accuracy_delta:+.2%}"
    task_loss_change = f"{float(intervention['delta_task_loss']):+.4f}"
    logit_l2 = f"{float(intervention['mean_logit_l2']):.3f}"
    cards = "".join(
        [
            metric("Prediction flips", flip_rate, image_count),
            metric("Accuracy change", accuracy_change, "intervened minus baseline"),
            metric("Task-loss change", task_loss_change, "intervened minus baseline"),
            metric("Mean logit L2", logit_l2, "per-image output movement"),
        ]
    )
    return f"<h3>{escape(action)}</h3><div class=\"metrics\">{cards}</div>"


def render_ancestry(ancestry):
    edges = ancestry["edges"]
    if not edges:
        return "<p>No incoming lineage edges.</p>"
    nodes = [
        f"<div class=\"ancestor-node birth\"><span>step {int(ancestry['structural_birth_step']):,}</span><strong>feature {int(ancestry['structural_birth_feature_id'])}</strong><small>structural ancestor</small></div>"
    ]
    for edge in edges:
        step, feature = parse_node(edge["target"])
        nodes.append(
            f"<div class=\"ancestor-edge\"><span>{escape(edge['relation'])}</span><small>score {float(edge['score']):.3f}</small></div><div class=\"ancestor-node\"><span>step {step:,}</span><strong>feature {feature}</strong></div>"
        )
    return "".join(nodes)


def line_chart(
    rows,
    field,
    title,
    subtitle,
    birth_step,
    reliable_start,
    percent,
):
    width = 720
    height = 300
    left = 58
    right = 18
    top = 30
    bottom = 45
    values = [float(row[field]) for row in rows]
    maximum = max(values) * 1.12 if max(values) > 0 else 1.0
    max_step = max(int(row["step"]) for row in rows)

    def x_value(step):
        return left + (width - left - right) * step / max_step

    def y_value(value):
        return top + (height - top - bottom) * (1.0 - value / maximum)

    all_points = " ".join(
        f"{x_value(int(row['step'])):.1f},{y_value(float(row[field])):.1f}"
        for row in rows
    )
    reliable = [row for row in rows if row["alignment_reliable"]]
    reliable_points = " ".join(
        f"{x_value(int(row['step'])):.1f},{y_value(float(row[field])):.1f}"
        for row in reliable
    )
    circles = []
    for row in rows:
        value = float(row[field])
        display = f"{value:.2%}" if percent else f"{value:.3f}"
        style = "reliable-point" if row["alignment_reliable"] else "unreliable-point"
        circles.append(
            f"<circle class=\"{style}\" cx=\"{x_value(int(row['step'])):.1f}\" cy=\"{y_value(value):.1f}\" r=\"4\"><title>step {int(row['step']):,}: {display}; residual {float(row['alignment_residual_ratio']):.3f}</title></circle>"
        )
    grid = []
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = maximum * fraction
        y = y_value(value)
        label = f"{value:.1%}" if percent else f"{value:.1f}"
        grid.append(
            f"<line class=\"grid\" x1=\"{left}\" x2=\"{width-right}\" y1=\"{y:.1f}\" y2=\"{y:.1f}\"/><text class=\"axis-label\" x=\"{left-10}\" y=\"{y+4:.1f}\" text-anchor=\"end\">{label}</text>"
        )
    ticks = []
    for step in [0, 5000, 10000, 15000, max_step]:
        label = "0" if step == 0 else f"{step/1000:.0f}k"
        ticks.append(
            f"<text class=\"axis-label\" x=\"{x_value(step):.1f}\" y=\"{height-16}\" text-anchor=\"middle\">{label}</text>"
        )
    birth_x = x_value(birth_step)
    reliable_x = x_value(reliable_start)
    return f"<figure class=\"chart\"><figcaption><strong>{escape(title)}</strong><span>{escape(subtitle)}</span></figcaption><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{escape(title)} across training steps\">{''.join(grid)}<polyline class=\"unreliable-line\" points=\"{all_points}\"/><polyline class=\"reliable-line\" points=\"{reliable_points}\"/><line class=\"birth-marker\" x1=\"{birth_x:.1f}\" x2=\"{birth_x:.1f}\" y1=\"{top}\" y2=\"{height-bottom}\"/><text class=\"annotation birth-label\" x=\"{birth_x+6:.1f}\" y=\"{height-bottom-8}\">ancestor {birth_step:,}</text><line class=\"reliable-marker\" x1=\"{reliable_x:.1f}\" x2=\"{reliable_x:.1f}\" y1=\"{top}\" y2=\"{height-bottom}\"/><text class=\"annotation\" x=\"{reliable_x+6:.1f}\" y=\"{top+13}\">reliable from {reliable_start:,}</text>{''.join(circles)}{''.join(ticks)}</svg></figure>"


def gallery(rows, report_root, group, limit):
    selected = [row for row in rows if row["group"] == group][:limit]
    figures = []
    for row in selected:
        path = next(
            (
                Path(value)
                for value in row["paths"]
                if value.endswith("_evidence.png")
            ),
            Path(row["paths"][0]),
        )
        uri = relative_uri(path, report_root)
        figures.append(
            f"<figure><button class=\"image-button\" data-full=\"{uri}\"><img src=\"{uri}\" alt=\"{escape(row['class_name'])} evidence\"></button><figcaption>{escape(row['class_name'])}<br><span>activation {float(row['activation']):.2f}</span></figcaption></figure>"
        )
    return "".join(figures) if figures else "<p>No evidence images available.</p>"


def document(title, body):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>{escape(title)}</title>
<style>
:root{{color-scheme:light dark;--bg:#f4f2ed;--paper:#fffefa;--ink:#171815;--muted:#686b65;--line:#d9d7cf;--blue:#245a78;--blue-soft:#e6f0f4;--gold:#a87718;--gold-soft:#f6edd8;--shadow:0 16px 44px rgba(35,32,24,.07)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;line-height:1.55}}.hero,main{{width:min(1120px,calc(100% - 40px));margin:0 auto}}.hero{{padding:58px 0 28px}}.index-hero{{padding-bottom:12px}}.back{{color:var(--blue);text-decoration:none;font-size:.9rem;font-weight:650}}.eyebrow{{color:var(--muted);font-size:.74rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;margin:18px 0 10px}}h1{{font-family:Georgia,serif;font-size:clamp(2.45rem,6vw,5.1rem);line-height:.98;letter-spacing:-.045em;max-width:960px;margin:0;font-weight:500}}.lede{{color:var(--muted);font-family:Georgia,serif;font-size:clamp(1.12rem,2vw,1.48rem);max-width:850px;margin:20px 0}}.badges{{display:flex;flex-wrap:wrap;gap:8px}}.badge{{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--muted);background:var(--paper);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em}}.badge.accepted{{color:var(--blue);border-color:#a8c3d2;background:var(--blue-soft)}}.badge.warning{{color:#875b08;border-color:#dbc18b;background:var(--gold-soft)}}main{{padding-bottom:80px}}section,details.sources{{background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow);margin:18px 0;padding:clamp(22px,4vw,42px);border-radius:18px}}h2{{font-family:Georgia,serif;font-size:clamp(1.6rem,3vw,2.25rem);line-height:1.12;letter-spacing:-.02em;margin:0 0 14px;font-weight:500}}h3{{font-size:.9rem;letter-spacing:.04em;text-transform:uppercase;margin:30px 0 12px}}p{{max-width:850px}}.note,.fine{{color:var(--muted);font-size:.9rem}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:24px 0}}.metric{{min-height:120px;border:1px solid var(--line);border-radius:12px;padding:17px;background:color-mix(in srgb,var(--paper) 86%,var(--bg))}}.metric span{{display:block;color:var(--muted);font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em}}.metric strong{{display:block;font-family:ui-monospace,SFMono-Regular,monospace;font-size:1.45rem;margin:8px 0 5px;letter-spacing:-.04em}}.metric small{{display:block;color:var(--muted);line-height:1.35}}.callout{{border-left:4px solid var(--blue);padding:14px 17px;background:var(--blue-soft);border-radius:0 9px 9px 0;margin:24px 0 0}}.callout.warning{{border-color:var(--gold);background:var(--gold-soft)}}.callout.neutral{{border-color:#85877f;background:color-mix(in srgb,var(--paper) 82%,var(--bg))}}.two-col{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:42px}}.two-col.compact{{border-top:1px solid var(--line);margin-top:22px}}ul,ol{{padding-left:1.2rem}}li{{margin:.55rem 0}}dl{{display:grid;grid-template-columns:minmax(120px,.7fr) 1.3fr;gap:8px 15px;margin:0}}dt{{color:var(--muted);font-weight:650}}dd{{margin:0}}.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.gallery.controls,.gallery.training{{grid-template-columns:repeat(4,minmax(0,1fr))}}figure{{margin:0}}.gallery figure{{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#090b0d}}.image-button{{border:0;padding:0;display:block;width:100%;background:#090b0d;cursor:zoom-in}}.gallery img{{display:block;width:100%;aspect-ratio:1.75/1;object-fit:contain}}.gallery.training img{{aspect-ratio:1/1;image-rendering:pixelated}}.gallery figcaption{{color:#e8e8e3;padding:9px 11px;font-size:.78rem;text-transform:capitalize}}.gallery figcaption span{{color:#9fa4a6;font-family:ui-monospace,monospace;font-size:.7rem}}.ancestry{{display:flex;align-items:center;gap:8px;overflow-x:auto;padding:18px 0 5px}}.ancestor-node{{min-width:112px;border:1px solid var(--line);border-radius:10px;padding:10px;background:var(--paper)}}.ancestor-node.birth{{border-color:var(--blue);background:var(--blue-soft)}}.ancestor-node span,.ancestor-node small{{display:block;color:var(--muted);font-size:.68rem}}.ancestor-node strong{{display:block;font-size:.9rem}}.ancestor-edge{{min-width:62px;text-align:center;color:var(--muted);font-size:.68rem}}.ancestor-edge:after{{content:'→';display:block;font-size:1.2rem;color:var(--blue)}}.ancestor-edge small{{display:block}}.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.chart{{border:1px solid var(--line);border-radius:12px;padding:16px;overflow:hidden}}.chart figcaption strong,.chart figcaption span{{display:block}}.chart figcaption span{{color:var(--muted);font-size:.78rem;margin-top:2px}}.chart svg{{display:block;width:100%;height:auto;margin-top:8px;overflow:visible}}.grid{{stroke:var(--line);stroke-width:1}}.axis-label,.annotation{{fill:var(--muted);font:10px ui-monospace,monospace}}.unreliable-line{{fill:none;stroke:#a7a9a4;stroke-width:2;stroke-dasharray:5 5}}.reliable-line{{fill:none;stroke:var(--blue);stroke-width:3}}.unreliable-point{{fill:var(--paper);stroke:#858882;stroke-width:2}}.reliable-point{{fill:var(--blue);stroke:var(--paper);stroke-width:2}}.birth-marker{{stroke:var(--gold);stroke-width:2;stroke-dasharray:3 4}}.reliable-marker{{stroke:var(--blue);stroke-width:1.5;stroke-dasharray:3 4}}.birth-label{{fill:var(--gold)}}.causal-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:20px 0}}.causal-card{{border:1px solid var(--line);border-radius:12px;padding:18px}}.causal-card h3{{margin:12px 0 8px}}.feature-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.feature-card{{display:flex;flex-direction:column;min-height:280px;border:1px solid var(--line);border-radius:14px;padding:24px;color:var(--ink);text-decoration:none;background:color-mix(in srgb,var(--paper) 86%,var(--bg));transition:transform .16s ease,border-color .16s ease}}.feature-card:hover{{transform:translateY(-3px);border-color:var(--blue)}}.feature-card h2{{font-size:1.5rem}}.feature-card p{{color:var(--muted)}}.stats{{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto}}.stats span{{font-size:.68rem;border:1px solid var(--line);border-radius:999px;padding:4px 8px}}.ladder{{max-width:820px}}details.sources summary{{cursor:pointer;font-weight:700}}details.sources code{{font-size:.75rem;word-break:break-all}}dialog{{width:min(92vw,1100px);max-height:92vh;padding:12px;border:1px solid var(--line);border-radius:14px;background:#090b0d}}dialog::backdrop{{background:rgba(0,0,0,.82)}}dialog img{{display:block;max-width:100%;max-height:84vh;margin:auto}}#close{{position:absolute;right:18px;top:14px;z-index:2;width:38px;height:38px;border:0;border-radius:50%;font-size:1.6rem;background:#fff;color:#111;cursor:pointer}}
@media(prefers-color-scheme:dark){{:root{{--bg:#101210;--paper:#181b18;--ink:#f0eee6;--muted:#a9ada6;--line:#343a34;--blue:#7fb8d5;--blue-soft:#172d37;--gold:#dbb258;--gold-soft:#332916;--shadow:none}}}}@media(max-width:800px){{.two-col,.chart-grid,.causal-grid{{grid-template-columns:1fr;gap:16px}}.gallery,.gallery.controls,.gallery.training,.feature-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.hero,main{{width:min(100% - 24px,1120px)}}section,details.sources{{border-radius:12px}}}}@media(max-width:480px){{.gallery,.gallery.controls,.gallery.training,.feature-grid{{grid-template-columns:1fr}}dl{{grid-template-columns:1fr}}dd{{margin-bottom:8px}}}}@media print{{body{{background:#fff}}.hero,main{{width:100%}}section{{box-shadow:none;break-inside:avoid}}.back,#viewer{{display:none}}}}
</style></head><body>{body}<script>const viewer=document.getElementById('viewer');const image=document.getElementById('viewer-image');if(viewer&&image){{document.querySelectorAll('.image-button').forEach(button=>button.addEventListener('click',()=>{{image.src=button.dataset.full;viewer.showModal()}}));document.getElementById('close').addEventListener('click',()=>viewer.close());viewer.addEventListener('click',event=>{{if(event.target===viewer)viewer.close()}})}}</script></body></html>"""


def metric(label, value, note):
    return f"<div class=\"metric\"><span>{escape(label)}</span><strong>{value}</strong><small>{escape(note)}</small></div>"


def bullet_list(values):
    return "<ul>" + "".join(f"<li>{escape(value)}</li>" for value in values) + "</ul>"


def definition_list(values):
    return "<dl>" + "".join(
        f"<dt>{escape(key.replace('_', ' '))}</dt><dd>{escape(value)}</dd>"
        for key, value in values.items()
    ) + "</dl>"


def phase_text(phase):
    if phase["status"] == "left_censored":
        start, end = phase["measurement_window"]
        return f"The feature is already active when measurement becomes reliable at step {start:,} and remains in consolidation through step {end:,}. Its semantic onset occurred earlier than the measurable window, so no exact emergence interval is claimed."
    return f"Emergence: {phase.get('emergence')}; consolidation: {phase.get('consolidation')}; status: {phase['status']}."


def relative_uri(path, report_root):
    relative = os.path.relpath(
        Path(path).resolve(),
        Path(report_root).resolve(),
    ).replace(os.sep, "/")
    return quote(relative, safe="/._-")


def parse_node(value):
    step, feature = value.split(":")
    return int(step.removeprefix("s")), int(feature.removeprefix("f"))


def format_probability(value):
    if value is None:
        return "not estimable"
    value = float(value)
    if value == 0:
        return "0"
    if value < 0.001:
        exponent = math.floor(math.log10(value))
        coefficient = value / 10**exponent
        return f"{coefficient:.2f}e{exponent}"
    return f"{value:.4f}"


def format_decimal(value):
    if value is None:
        return "not estimable"
    value = float(value)
    if not math.isfinite(value):
        return "not estimable"
    return f"{value:.3f}"


def escape(value):
    return html.escape(str(value))
