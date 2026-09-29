from __future__ import annotations

import html
import json
from pathlib import Path

from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_lineage_path,
    circuit_root,
    circuit_steps,
    circuit_validation_path,
    feature_circuit_root,
)


def render_circuit_report(config) -> Path:
    root = circuit_root(config) / "report"
    root.mkdir(parents=True, exist_ok=True)
    labels_path = circuit_root(config) / "node_labels" / "labels.json"
    labels = _read(labels_path) if labels_path.exists() else {}
    rows = []
    for feature_id in accepted_target_feature_ids(config):
        lineage_path = circuit_lineage_path(config, feature_id)
        if not lineage_path.exists():
            continue
        lineage = _read(lineage_path)
        final_step = int(lineage["final_step"])
        final_graph = _read(circuit_graph_path(config, feature_id, final_step))
        validation_path = circuit_validation_path(config, feature_id, final_step)
        validation = _read(validation_path) if validation_path.exists() else None
        label = _target_label(config, feature_id)
        graphs = {
            step: _read(circuit_graph_path(config, feature_id, step))
            for step in circuit_steps(config)
            if circuit_graph_path(config, feature_id, step).exists()
        }
        for graph in graphs.values():
            graph_step = int(graph.get("model_step", -1))
            graph_validation_path = circuit_validation_path(
                config,
                feature_id,
                graph_step,
            )
            selected_ids = set()
            if graph_validation_path.exists():
                graph_validation = _read(graph_validation_path)
                selected_ids = set(graph_validation.get("selected_node_ids", []))
            for node in graph.get("nodes", []):
                node["validated_circuit"] = node["kind"] == "logit" or node["id"] in selected_ids
                if node["id"] in labels:
                    node["human_label"] = labels[node["id"]]["canonical_label"]
                    node["human_summary"] = labels[node["id"]]["plain_english_summary"]
            for edge in graph.get("edges", []):
                edge["validated_circuit"] = edge["source"] in selected_ids and (
                    edge["target"] in selected_ids
                    or edge["target"].startswith("logit:")
                )
        page_root = root / f"feature_{feature_id:05d}"
        page_root.mkdir(parents=True, exist_ok=True)
        (page_root / "index.html").write_text(
            _feature_page(
                config,
                feature_id,
                label,
                lineage,
                graphs,
                validation,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "feature_id": feature_id,
                "label": label,
                "emergence_step": lineage.get("emergence_step"),
                "confidence": max(
                    (row["emergence_confidence"] for row in lineage["trajectory"]),
                    default=0.0,
                ),
                "nodes": final_graph["graph_summary"]["nodes"],
                "edges": final_graph["graph_summary"]["edges"],
                "faithfulness": validation["scores"]["sufficiency_faithfulness"]
                if validation
                else None,
                "passed": validation.get("passed") if validation else False,
            }
        )
    (root / "index.html").write_text(_index_page(rows), encoding="utf-8")
    return root / "index.html"


def _feature_page(config, feature_id, label, lineage, graphs, validation):
    graph_json = json.dumps(graphs, separators=(",", ":")).replace("</", "<\\/")
    trajectory = lineage["trajectory"]
    replay_rows = []
    feature_root = feature_circuit_root(config, feature_id)
    for path in sorted(feature_root.glob("counterfactual_*_rank_*.json")):
        replay_rows.append(_read(path))
    replay_html = "".join(
        f"<article><span class='badge'>{_escape(row['causal_role'])}</span>"
        f"<h3>Training step {int(row['candidate_step']):,}</h3>"
        f"<p>Circuit presence effect <strong>{float(row['causal_presence_effect']):+.4f}</strong>. "
        f"Prediction {'confirmed' if row['predicted_direction_confirmed'] else 'disconfirmed'}.</p>"
        f"<p class='fine'>Masked examples: {_escape(', '.join(str(value) for value in row['masked_dataset_indices']))}</p></article>"
        for row in replay_rows
    ) or "<p class='muted'>Exact training-example replay has not been run yet.</p>"
    metrics = validation["scores"] if validation else {}
    emergence = lineage.get("emergence_step")
    return _document(
        f"Circuit · feature {feature_id:05d}",
        f"""
<main>
<nav><a href='../index.html'>← circuit atlas</a></nav>
<header><p class='eyebrow'>Sparse visual circuit · target feature {feature_id:05d}</p><h1>{_escape(label)}</h1><p>Follow the computation through depth and training time. Edges are attribution hypotheses until causal patching confirms them.</p></header>
<section class='metrics'>
<article><span>Emergence</span><strong>{f'step {emergence:,}' if emergence is not None else 'unresolved'}</strong></article>
<article><span>Final faithfulness</span><strong>{float(metrics.get('sufficiency_faithfulness', 0)):.1%}</strong></article>
<article><span>Edge confirmation</span><strong>{float(validation.get('edge_sign_confirmation_fraction', 0) if validation else 0):.1%}</strong></article>
<article><span>Scientific gate</span><strong>{'passed' if validation and validation.get('passed') else 'pending / failed'}</strong></article>
</section>
<section><div class='section-head'><div><p class='eyebrow'>Circuit explorer</p><h2>How the pathway changes</h2></div><label>Checkpoint <select id='step'></select></label></div><div id='graph'></div><aside id='detail'>Select a node or edge.</aside></section>
<section><p class='eyebrow'>Development</p><h2>When the circuit became reliable</h2>{_trajectory_table(trajectory)}</section>
<section><p class='eyebrow'>Training causality</p><h2>Examples that built or disrupted the circuit</h2><div class='cards'>{replay_html}</div></section>
</main>
<script>const graphs={graph_json};{_graph_script()}</script>
""",
    )


def _index_page(rows):
    cards = "".join(
        f"<a class='world' href='feature_{row['feature_id']:05d}/index.html'>"
        f"<p class='eyebrow'>Feature {row['feature_id']:05d}</p><h2>{_escape(row['label'])}</h2>"
        f"<p>{row['nodes']} nodes · {row['edges']} edges</p>"
        f"<div><span class='badge'>{'validated' if row['passed'] else 'pending'}</span> "
        f"<span class='badge'>birth {row['emergence_step'] if row['emergence_step'] is not None else '?'}</span> "
        f"<span class='badge'>faithfulness {float(row['faithfulness'] or 0):.0%}</span></div></a>"
        for row in rows
    )
    return _document(
        "Circuit Genesis",
        f"<main><header><p class='eyebrow'>Feature Genesis · second study</p><h1>Circuit Genesis</h1><p>From isolated sparse features to causally validated computations across model depth and training time.</p></header><section class='worlds'>{cards}</section></main>",
    )


def _trajectory_table(rows):
    body = "".join(
        f"<tr><td>{int(row['step']):,}</td><td>{_escape(row['phase'])}</td>"
        f"<td>{int(row['nodes'])}</td><td>{int(row['edges'])}</td>"
        f"<td>{float(row['final_edge_similarity']):.0%}</td>"
        f"<td>{float(row['sufficiency_faithfulness']):.0%}</td>"
        f"<td>{float(row['emergence_confidence']):.0%}</td></tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr><th>Step</th><th>Phase</th><th>Nodes</th><th>Edges</th><th>Final similarity</th><th>Faithfulness</th><th>Confidence</th></tr></thead><tbody>{body}</tbody></table></div>"


def _graph_script():
    return """
const select=document.getElementById('step'),host=document.getElementById('graph'),detail=document.getElementById('detail');
Object.keys(graphs).sort((a,b)=>+a-+b).forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=Number(s).toLocaleString();select.appendChild(o)});
select.value=Object.keys(graphs).sort((a,b)=>+a-+b).at(-1);select.onchange=render;
function render(){const g=graphs[select.value];host.innerHTML='';if(g.status){host.textContent='Target inactive at this checkpoint.';return}const layers=[...g.layers,'output'],by={};layers.forEach(x=>by[x]=[]);g.nodes.forEach(n=>by[n.kind==='logit'?'output':n.layer].push(n));const W=1100,H=Math.max(430,...Object.values(by).map(x=>x.length*72+100));const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);const pos={};layers.forEach((l,li)=>{const x=70+li*(W-140)/(layers.length-1);const arr=by[l];arr.forEach((n,i)=>pos[n.id]=[x,70+(i+1)*(H-120)/(arr.length+1)]);const t=document.createElementNS(svg.namespaceURI,'text');t.setAttribute('x',x);t.setAttribute('y',28);t.setAttribute('text-anchor','middle');t.setAttribute('class','layer-label');t.textContent=l;svg.appendChild(t)});g.edges.forEach(e=>{if(!pos[e.source]||!pos[e.target])return;const [x1,y1]=pos[e.source],[x2,y2]=pos[e.target],p=document.createElementNS(svg.namespaceURI,'path');p.setAttribute('d',`M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`);p.setAttribute('class',(e.attribution_mean>=0?'edge positive':'edge negative')+(e.validated_circuit?'':' candidate'));p.setAttribute('stroke-width',Math.min(7,1+Math.log1p(Math.abs(e.attribution_mean)*100)));p.onclick=()=>detail.innerHTML=`<strong>${e.source} → ${e.target}</strong><br>Attribution ${e.attribution_mean.toPrecision(4)}<br>Bootstrap CI ${e.bootstrap_ci.map(v=>v.toPrecision(3)).join(' — ')}<br>Sign agreement ${(100*e.sign_agreement).toFixed(0)}%`;svg.appendChild(p)});g.nodes.forEach(n=>{if(!pos[n.id])return;const [x,y]=pos[n.id],group=document.createElementNS(svg.namespaceURI,'g'),circle=document.createElementNS(svg.namespaceURI,'circle'),text=document.createElementNS(svg.namespaceURI,'text');circle.setAttribute('cx',x);circle.setAttribute('cy',y);circle.setAttribute('r',n.forced_target?18:12);circle.setAttribute('class',(n.forced_target?'node target':'node')+(n.validated_circuit?'':' candidate'));text.setAttribute('x',x);text.setAttribute('y',y+32);text.setAttribute('text-anchor','middle');text.textContent=n.kind==='logit'?n.label:(n.human_label||`f${String(n.feature_id).padStart(4,'0')}`);group.append(circle,text);group.onclick=()=>detail.innerHTML=`<strong>${n.human_label||n.id}</strong><br>${n.human_summary||n.id}<br>${n.forced_target?'Original validated target feature':'Sparse circuit node'}<br>Maximum attribution ${(n.maximum_absolute_attribution||0).toPrecision(4)}`;svg.appendChild(group)});host.appendChild(svg)}render();
"""


def _document(title, body):
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>{_escape(title)}</title><style>
:root{{--bg:#0f1210;--panel:#161a17;--line:#303832;--text:#f2f0e8;--muted:#aeb7b0;--accent:#9dc7b2;--red:#c88f8f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:17px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:38px 28px 80px}}header{{padding:42px 0 34px;max-width:850px}}h1{{font:56px/1.04 Georgia,serif;margin:8px 0 18px}}h2{{font:32px/1.15 Georgia,serif;margin:5px 0 14px}}h3{{margin:10px 0}}a{{color:inherit;text-decoration:none}}nav a,.muted,.fine{{color:var(--muted)}}section{{margin:24px 0;padding:28px;border:1px solid var(--line);border-radius:20px;background:var(--panel)}}.eyebrow{{text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:700;color:var(--accent)}}.metrics,.worlds,.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.metrics article,.cards article,.world{{padding:20px;border:1px solid var(--line);border-radius:14px;background:#121512}}.metrics span{{display:block;color:var(--muted);font-size:13px}}.metrics strong{{font:24px Georgia,serif}}.world:hover{{border-color:var(--accent)}}.badge{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 9px;font-size:12px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:20px}}select{{background:#101310;color:var(--text);border:1px solid var(--line);padding:8px;border-radius:8px}}#graph{{overflow:auto;background:#101310;border-radius:14px;margin-top:18px}}#graph svg{{width:100%;min-width:900px;display:block}}#detail{{margin-top:12px;padding:14px;border-left:3px solid var(--accent);background:#121512}}.edge{{fill:none;opacity:.72;cursor:pointer}}.positive{{stroke:var(--accent)}}.negative{{stroke:var(--red)}}.candidate{{opacity:.22}}.node{{fill:#d9ded9;stroke:#0c0f0d;stroke-width:3;cursor:pointer}}.target{{fill:#f0c879;stroke:#fff0b5;stroke-width:4}}svg text{{fill:var(--text);font-size:11px}}.layer-label{{fill:var(--muted);font-weight:700}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}@media(max-width:700px){{h1{{font-size:40px}}section{{padding:18px}}}}
</style></head><body>{body}</body></html>"""


def _target_label(config, feature_id):
    matches = list(Path(config.project.run_dir).glob(f"feature_evaluation/**/gemini/feature_{feature_id:05d}_evaluation.json"))
    if not matches:
        return f"Feature {feature_id:05d}"
    row = _read(matches[-1])
    return row.get("label", {}).get("canonical_label", f"Feature {feature_id:05d}")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _escape(value):
    return html.escape(str(value))
