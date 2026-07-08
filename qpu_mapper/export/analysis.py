"""
export.analysis — Per-qubit / per-gate ranking, trend, connectivity and
stable-element finders. Pure data + HTML; no widgets, no file I/O.
"""
import numpy as np

from ..constants import HIGHER_IS_BETTER_PROPS


# ── Qubit ranking table ──────────────────────────────────────────────

# Fixed columns the qubit ranking table always shows; if the sort
# property is one of these, the column gets bolded rather than appended
# as an extra one.
_QUBIT_RANKING_FIXED_KEYS = frozenset({'readout_error', 't1', 't2'})


def _fmt_or_dash(v, fmt='3E'):
    """Format a float, or render '—' on NaN. Shared by both ranking tables."""
    return f"{v:.{fmt}}" if not np.isnan(v) else "—"


def _per_qubit_gate_stats(edges_list, cz_arr, n_ids):
    """For each qubit index `i in range(n_ids)`, return the list of CZ
    errors of the edges incident on `i` (NaN entries skipped)."""
    g_errors: dict = {i: [] for i in range(n_ids)}
    for gi, edge in enumerate(edges_list or []):
        if gi >= len(cz_arr) or np.isnan(cz_arr[gi]):
            continue
        for eidx in edge:
            if eidx < n_ids:
                g_errors[eidx].append(float(cz_arr[gi]))
    return g_errors


def _collect_qubit_ranking_rows(state, prop_key, ascending, top_n, filter_text):
    """Pure data-prep pass for `get_qubit_ranking_html`.

    Pulls the per-qubit metrics needed to render the table from `state`,
    filters/sorts, and returns the list of row tuples
    `(qid, val, ro, t1, t2, avg_cz, worst_cz, n_gates)` plus the pre-filter
    total used by the rank-fraction heat-map colour. Kept separate from
    the HTML builder so the data side stays unit-testable without
    parsing a string.
    """
    arr = state.qubit_props_data[prop_key]
    ids = state.qubit_ids or [str(i) for i in range(len(arr))]

    re_arr = state.qubit_props_data.get('readout_error', np.array([]))
    t1_arr = state.qubit_props_data.get('t1', np.array([]))
    t2_arr = state.qubit_props_data.get('t2', np.array([]))
    cz_arr = (state.gate_props_data.get('cz_error', np.array([]))
              if state.gate_props_data else np.array([]))
    g_errors = _per_qubit_gate_stats(state.edges_list, cz_arr, len(ids))

    def _safe(arr_, i):
        return (float(arr_[i])
                if i < len(arr_) and not np.isnan(arr_[i])
                else float('nan'))

    rows = []
    for i in range(len(arr)):
        val = float(arr[i])
        if np.isnan(val):
            continue
        ge = g_errors.get(i, [])
        rows.append((
            ids[i], val,
            _safe(re_arr, i), _safe(t1_arr, i), _safe(t2_arr, i),
            float(np.mean(ge)) if ge else float('nan'),
            float(max(ge)) if ge else float('nan'),
            len(ge),
        ))

    rows.sort(key=lambda x: x[1], reverse=not ascending)
    total = len(rows)
    if filter_text:
        tokens = [t.strip()
                  for t in filter_text.replace(';', ',').split(',')
                  if t.strip()]
        if tokens:
            rows = [r for r in rows if any(t in r[0] for t in tokens)]
    if top_n > 0:
        rows = rows[:top_n]
    return rows, total, len(ids)


def _qubit_ranking_html_rows(rows, total, prop_key, extra_col, ascending):
    """HTML rendering pass over the prepared rows."""
    def _bold_if(key):
        return "font-weight:bold;" if prop_key == key else ""

    out = []
    for rank, (qid, val, ro, t1, t2, ga, gw, gc) in enumerate(rows, 1):
        frac = rank / max(total, 1)
        c = (f"hsl({120 - frac*120:.0f},60%,92%)" if ascending
             else f"hsl({frac*120:.0f},60%,92%)")
        extra_td = (
            f"<td style='padding:4px 6px;text-align:right;font-weight:bold;'>"
            f"{_fmt_or_dash(val)}</td>"
            if extra_col else "")
        out.append(
            f"<tr style='background:{c};'>"
            f"<td style='padding:4px 6px;text-align:center;'>{rank}</td>"
            f"<td style='padding:4px 6px;text-align:center;font-weight:bold;'>Q{qid}</td>"
            f"{extra_td}"
            f"<td style='padding:4px 6px;text-align:right;{_bold_if('readout_error')}'>"
            f"{_fmt_or_dash(ro)}</td>"
            f"<td style='padding:4px 6px;text-align:right;{_bold_if('t1')}'>"
            f"{_fmt_or_dash(t1,'0f')}</td>"
            f"<td style='padding:4px 6px;text-align:right;{_bold_if('t2')}'>"
            f"{_fmt_or_dash(t2,'0f')}</td>"
            f"<td style='padding:4px 6px;text-align:right;'>{_fmt_or_dash(ga)}</td>"
            f"<td style='padding:4px 6px;text-align:right;'>{_fmt_or_dash(gw)}</td>"
            f"<td style='padding:4px 6px;text-align:center;'>{gc}</td>"
            f"</tr>")
    return ''.join(out)


def get_qubit_ranking_html(state, prop_key='readout_error', top_n=0, ascending=True,
                           filter_text=''):
    """Generate HTML table ranking ALL qubits with gate context.

    Fixed columns: Readout, T1, T2, Avg CZ, Worst CZ, Links.
    If sort property is NOT a fixed column, adds it as extra column.
    If it IS a fixed column, that column gets bolded instead.

    Split into:
      • `_collect_qubit_ranking_rows` — pure data prep (testable),
      • `_qubit_ranking_html_rows`    — HTML row rendering,
      • the wrapper here — header + container, no logic.
    """
    if not state.qubit_props_data or prop_key not in state.qubit_props_data:
        return "<span style='color:#888;'>No calibration data available.</span>"

    rows, total, n_ids = _collect_qubit_ranking_rows(
        state, prop_key, ascending, top_n, filter_text)

    extra_col = prop_key not in _QUBIT_RANKING_FIXED_KEYS
    order = "↑" if ascending else "↓"
    row_html = _qubit_ranking_html_rows(rows, total, prop_key, extra_col, ascending)

    ro_hdr = f"<b>Readout {order}</b>" if prop_key == 'readout_error' else "Readout"
    t1_hdr = f"<b>T1 {order}</b>" if prop_key == 't1' else "T1 µs"
    t2_hdr = f"<b>T2 {order}</b>" if prop_key == 't2' else "T2 µs"
    extra_th = (f"<th style='padding:2px 3px;'><b>{prop_key} {order}</b></th>"
                if extra_col else "")

    return (
        f"<div style='max-height:400px;overflow-y:auto;'>"
        f"<table style='font-size:15px;border-collapse:collapse;width:100%;'>"
        f"<tr style='background:#2C3E50;color:white;'>"
        f"<th style='padding:2px 3px;'>#</th>"
        f"<th style='padding:2px 3px;'>Q</th>"
        f"{extra_th}"
        f"<th style='padding:2px 3px;'>{ro_hdr}</th>"
        f"<th style='padding:2px 3px;'>{t1_hdr}</th>"
        f"<th style='padding:2px 3px;'>{t2_hdr}</th>"
        f"<th style='padding:2px 3px;'>Avg CZ</th>"
        f"<th style='padding:2px 3px;'>Worst CZ</th>"
        f"<th style='padding:2px 3px;'>Links</th>"
        f"</tr>{row_html}</table>"
        f"<div style='font-size:13px;color:#888;margin:2px;'>"
        f"{len(rows)}/{n_ids} qubits</div></div>"
    )


def get_gate_ranking_html(state, prop_key='cz_error', top_n=0, ascending=True,
                          filter_text=''):
    """Generate HTML table ranking ALL gates by a property.

    Columns: #, Edge, sort_prop, Gate Length, Q_A readout, Q_B readout.
    top_n=0 means ALL gates.
    """
    if not state.gate_props_data or prop_key not in state.gate_props_data:
        return "<span style='color:#888;'>No gate data available.</span>"

    arr = state.gate_props_data[prop_key]
    edges = state.edges_list or []
    ids = state.qubit_ids or []

    re_arr = state.qubit_props_data.get('readout_error', np.array([])) if state.qubit_props_data else np.array([])
    gl_arr = state.gate_props_data.get('gate_length', np.array([]))

    data_rows = []
    for i, edge in enumerate(edges):
        if i >= len(arr):
            continue
        val = float(arr[i])
        if np.isnan(val):
            continue
        u, v = edge
        u_id = ids[u] if u < len(ids) else str(u)
        v_id = ids[v] if v < len(ids) else str(v)
        gl = float(gl_arr[i]) if i < len(gl_arr) and not np.isnan(gl_arr[i]) else float('nan')
        ro_u = float(re_arr[u]) if u < len(re_arr) and not np.isnan(re_arr[u]) else float('nan')
        ro_v = float(re_arr[v]) if v < len(re_arr) and not np.isnan(re_arr[v]) else float('nan')
        data_rows.append((u_id, v_id, val, gl, ro_u, ro_v))

    data_rows.sort(key=lambda x: x[2], reverse=not ascending)
    total = len(data_rows)
    if filter_text:
        tokens = [t.strip() for t in filter_text.replace(';', ',').split(',') if t.strip()]
        if tokens:
            data_rows = [r for r in data_rows if any(t in r[0] or t in r[1] for t in tokens)]
    if top_n > 0:
        data_rows = data_rows[:top_n]

    def _f(v, fmt='3E'):
        return f"{v:.{fmt}}" if not np.isnan(v) else "—"

    rows = []
    for rank, (uid, vid, val, gl, rou, rov) in enumerate(data_rows, 1):
        frac = rank / max(total, 1)
        c = f"hsl({120 - frac*120:.0f},60%,92%)" if ascending else f"hsl({frac*120:.0f},60%,92%)"
        rows.append(
            f"<tr style='background:{c};'>"
            f"<td style='padding:4px 6px;text-align:center;'>{rank}</td>"
            f"<td style='padding:4px 6px;text-align:center;font-weight:bold;'>({uid},{vid})</td>"
            f"<td style='padding:4px 6px;text-align:right;font-weight:bold;'>{_f(val)}</td>"
            f"<td style='padding:4px 6px;text-align:right;'>{_f(gl, '0f')} ns</td>"
            f"<td style='padding:4px 6px;text-align:right;'>{_f(rou)}</td>"
            f"<td style='padding:4px 6px;text-align:right;'>{_f(rov)}</td>"
            f"</tr>")

    order = "Best→Worst" if ascending else "Worst→Best"
    n_edges = len(edges)
    return (
        f"<div style='max-height:400px;overflow-y:auto;'>"
        f"<table style='font-size:15px;border-collapse:collapse;width:100%;'>"
        f"<tr style='background:#1a5276;color:white;'>"
        f"<th style='padding:2px 3px;'>#</th>"
        f"<th style='padding:2px 3px;'>Edge</th>"
        f"<th style='padding:2px 3px;'>{prop_key}<br>({order})</th>"
        f"<th style='padding:2px 3px;'>Length</th>"
        f"<th style='padding:2px 3px;'>RO Q_A</th>"
        f"<th style='padding:2px 3px;'>RO Q_B</th>"
        f"</tr>{''.join(rows)}</table>"
        f"<div style='font-size:13px;color:#888;margin:2px;'>"
        f"{len(data_rows)}/{n_edges} gates</div></div>"
    )


# ── Qubit trend data (time-series chart) ─────────────────────────────

def get_qubit_trend_data(state, qubit_ids, prop_key='readout_error'):
    """Extract time series data for specific qubits from calibration history.

    Args:
        state: SessionState with cal_history populated.
        qubit_ids: List of qubit ID strings to track.
        prop_key: Property to extract.

    Returns:
        dict: {'dates': [str], 'series': {qubit_id: [float]}}
    """
    if not state.cal_history_dates or not state.cal_history:
        return {'dates': [], 'series': {}}

    q_id_map = {q: i for i, q in enumerate(state.qubit_ids)} if state.qubit_ids else {}
    dates = sorted(state.cal_history_dates)
    series = {qid: [] for qid in qubit_ids}

    for date_str in dates:
        snap = state.cal_history.get(date_str, {})
        qp = snap.get('qubit_props', {})
        arr = qp.get(prop_key, np.array([]))
        for qid in qubit_ids:
            idx = q_id_map.get(qid, -1)
            if 0 <= idx < len(arr):
                series[qid].append(float(arr[idx]))
            else:
                series[qid].append(float('nan'))

    return {'dates': dates, 'series': series}


def get_gate_trend_data(state, edge_strs, prop_key='cz_error'):
    """Extract time-series data for specific gates across calibration history.

    Args:
        state: SessionState with cal_history populated.
        edge_strs: List of edge strings like "(1,2)", "1,2", "(10, 15)", etc.
        prop_key: Gate property to extract.

    Returns:
        dict: {'dates': [str], 'series': {'(1,2)': [float], ...}}
    """
    if not state.cal_history_dates or not state.cal_history:
        return {'dates': [], 'series': {}}

    # Parse edge strings into tuples
    import re as _re
    edge_map = {}  # edge_label → index in edges_list
    edges_list = state.edges_list or []
    for estr in edge_strs:
        nums = _re.findall(r'\d+', estr)
        if len(nums) >= 2:
            u, v = int(nums[0]), int(nums[1])
            label = f"({u},{v})"
            # Find index in edges_list (try both orderings)
            idx = None
            for i, e in enumerate(edges_list):
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    idx = i
                    break
            edge_map[label] = idx

    dates = sorted(state.cal_history_dates)
    series = {label: [] for label in edge_map}

    for date_str in dates:
        snap = state.cal_history.get(date_str, {})
        gp = snap.get('gate_props', {})
        arr = gp.get(prop_key, np.array([]))
        for label, idx in edge_map.items():
            if idx is not None and 0 <= idx < len(arr):
                series[label].append(float(arr[idx]))
            else:
                series[label].append(float('nan'))

    return {'dates': dates, 'series': series}


# ── Connectivity heatmap (per-qubit gate quality) ────────────────────

def get_connectivity_scores(state):
    """Compute per-qubit connectivity quality score.

    For each qubit: average CZ error of all its connected gates.
    Lower = better neighborhood. Also returns number of connections.

    Returns:
        list of dicts: [{'qubit': '0', 'avg_gate_error': float,
                         'n_connections': int, 'best_gate': float,
                         'worst_gate': float}, ...]
    """
    if not state.edges_list or not state.gate_props_data:
        return []

    cz_arr = state.gate_props_data.get('cz_error', np.array([]))
    q_ids = state.qubit_ids or []

    # Build adjacency: qubit_idx → list of neighbor gate errors.
    # Deduplicate by unordered pair: a coupling map may list a link as both
    # (u, v) and (v, u); counting both would double n_connections and skew the
    # per-qubit averages. A frozenset key collapses the two directions to one.
    # A pair is marked "seen" only when it actually contributes a valid
    # (non-NaN) value, so a NaN in one direction never blocks a valid value in
    # the other.
    adjacency = {i: [] for i in range(len(q_ids))}
    seen_pairs = set()
    for gi, edge in enumerate(state.edges_list):
        u, v = edge
        pair = frozenset((u, v))
        if pair in seen_pairs:
            continue
        if gi < len(cz_arr) and not np.isnan(cz_arr[gi]):
            seen_pairs.add(pair)
            adjacency[u].append(float(cz_arr[gi]))
            adjacency[v].append(float(cz_arr[gi]))

    results = []
    for i, qid in enumerate(q_ids):
        errors = adjacency.get(i, [])
        if errors:
            results.append({
                'qubit': qid,
                'avg_gate_error': float(np.mean(errors)),
                'n_connections': len(errors),
                'best_gate': float(min(errors)),
                'worst_gate': float(max(errors)),
            })
        else:
            results.append({
                'qubit': qid, 'avg_gate_error': float('nan'),
                'n_connections': 0, 'best_gate': float('nan'),
                'worst_gate': float('nan'),
            })

    return results


def get_connectivity_ranking_html(state, top_n=20, ascending=True):
    """HTML table ranking qubits by neighborhood gate quality."""
    scores = get_connectivity_scores(state)
    if not scores:
        return "<span style='color:#888;'>No data.</span>"

    # Filter out NaN, sort
    valid = [s for s in scores if not np.isnan(s['avg_gate_error'])]
    valid.sort(key=lambda x: x['avg_gate_error'], reverse=not ascending)
    if top_n > 0:
        valid = valid[:top_n]

    # Guard None qubit_props_data (fresh session / offline) to avoid
    # AttributeError — mirrors the other helpers in this module.
    re_arr = (state.qubit_props_data.get('readout_error', np.array([]))
              if state.qubit_props_data else np.array([]))
    q_id_map = {q: i for i, q in enumerate(state.qubit_ids)} if state.qubit_ids else {}

    rows = []
    for rank, s in enumerate(valid, 1):
        idx = q_id_map.get(s['qubit'], -1)
        ro = f"{re_arr[idx]:.3E}" if 0 <= idx < len(re_arr) and not np.isnan(re_arr[idx]) else "—"
        frac = rank / len(valid)
        color = f"hsl({120 - frac * 120:.0f}, 70%, 90%)" if ascending else \
                f"hsl({frac * 120:.0f}, 70%, 90%)"
        rows.append(
            f"<tr style='background:{color};'>"
            f"<td style='padding:2px 6px;text-align:center;'>{rank}</td>"
            f"<td style='padding:2px 6px;text-align:center;font-weight:bold;'>Q{s['qubit']}</td>"
            f"<td style='padding:2px 6px;text-align:right;'>{s['avg_gate_error']:.4E}</td>"
            f"<td style='padding:2px 6px;text-align:center;'>{s['n_connections']}</td>"
            f"<td style='padding:2px 6px;text-align:right;'>{ro}</td>"
            f"</tr>"
        )

    order = "Best → Worst" if ascending else "Worst → Best"
    return (
        f"<div style='max-height:300px;overflow-y:auto;'>"
        f"<table style='font-size:15px;border-collapse:collapse;width:100%;'>"
        f"<tr style='background:#2C3E50;color:white;'>"
        f"<th style='padding:3px 6px;'>#</th>"
        f"<th style='padding:3px 6px;'>Qubit</th>"
        f"<th style='padding:3px 6px;'>Avg CZ Err ({order})</th>"
        f"<th style='padding:3px 6px;'>Links</th>"
        f"<th style='padding:3px 6px;'>Readout</th>"
        f"</tr>"
        f"{''.join(rows)}"
        f"</table></div>"
    )


# ── Stable qubit finder ──────────────────────────────────────────────

def find_stable_qubits(state, prop_key='readout_error', max_error=0.05,
                        min_days_ratio=0.8):
    """Find qubits that stayed within an acceptable range across calibration history.

    The comparison direction depends on the property:
      - For error rates and durations (default): qubits must satisfy
        ``value <= max_error`` to count as good. Lower is better.
      - For coherence times (T1, T2; see ``HIGHER_IS_BETTER_PROPS``): qubits
        must satisfy ``value >= max_error`` to count as good. Higher is better.
        In this case, ``max_error`` is reinterpreted as a *minimum* threshold
        despite the parameter name (kept as-is for API back-compat).

    Args:
        state: SessionState with cal_history populated.
        prop_key: Property to check.
        max_error: Threshold. Interpretation depends on prop_key (see above).
        min_days_ratio: Fraction of days qubit must satisfy the threshold (0-1).
                        1.0 = every single day. 0.8 = 80% of days.

    Returns:
        list of dicts sorted by stability:
        [{'qubit': '5', 'good_days': 9, 'total_days': 10, 'ratio': 0.9,
          'avg_value': 0.008, 'max_value': 0.012}, ...]
    """
    higher_is_better = prop_key in HIGHER_IS_BETTER_PROPS

    if not state.cal_history_dates or not state.cal_history:
        return []

    q_ids = state.qubit_ids or []
    q_id_map = {q: i for i, q in enumerate(q_ids)}

    # For each qubit, count how many days it satisfies the threshold
    stats = {qid: {'good': 0, 'total': 0, 'values': []} for qid in q_ids}

    for date_str in state.cal_history_dates:
        snap = state.cal_history.get(date_str, {})
        arr = snap.get('qubit_props', {}).get(prop_key, np.array([]))
        for qid in q_ids:
            idx = q_id_map.get(qid, -1)
            if 0 <= idx < len(arr):
                val = arr[idx]
                if not np.isnan(val):
                    stats[qid]['total'] += 1
                    stats[qid]['values'].append(val)
                    if higher_is_better:
                        if val >= max_error:
                            stats[qid]['good'] += 1
                    else:
                        if val <= max_error:
                            stats[qid]['good'] += 1

    results = []
    for qid, s in stats.items():
        if s['total'] == 0:
            continue
        ratio = s['good'] / s['total']
        if ratio >= min_days_ratio:
            results.append({
                'qubit': qid,
                'good_days': s['good'],
                'total_days': s['total'],
                'ratio': ratio,
                'avg_value': float(np.mean(s['values'])),
                'max_value': float(np.max(s['values'])),
            })

    # Sort: most-stable first; tiebreak by avg_value direction.
    # Higher-is-better → tiebreak descending (higher avg = better).
    # Lower-is-better  → tiebreak ascending (lower avg = better).
    if higher_is_better:
        results.sort(key=lambda x: (-x['ratio'], -x['avg_value']))
    else:
        results.sort(key=lambda x: (-x['ratio'], x['avg_value']))
    return results


def _build_stable_html(results, label_col, empty_msg, top_n,
                       row_label_fn):
    """Generic HTML renderer shared between get_stable_qubits_html and
    get_stable_gates_html. The two callers differ only in:
      • the result-row label column header (`label_col`, e.g. 'Qubit'/'Gate'),
      • how the per-row label is rendered (`row_label_fn(r)` returns the
        cell HTML — e.g. `Q5` for qubits, `(0,1)` for gates),
      • the empty-state message (`empty_msg`).

    Keeping the rest of the table chrome here removes the ~30-line
    near-duplicate that lived in both callers and was already drifting
    (different padding values had appeared in one but not the other).
    """
    if not results:
        return f"<span style='color:#888;font-size:13px;'>{empty_msg}</span>"

    if top_n > 0:
        results = results[:top_n]

    rows = []
    for rank, r in enumerate(results, 1):
        pct = r['ratio'] * 100
        bar_color = ('#27ae60' if pct == 100
                     else '#e67e22' if pct >= 90
                     else '#e74c3c')
        rows.append(
            f"<tr>"
            f"<td style='padding:2px 6px;text-align:center;'>{rank}</td>"
            f"<td style='padding:2px 6px;text-align:center;font-weight:bold;'>"
            f"{row_label_fn(r)}</td>"
            f"<td style='padding:2px 6px;text-align:center;'>"
            f"<span style='color:{bar_color};font-weight:bold;'>{pct:.0f}%</span> "
            f"({r['good_days']}/{r['total_days']})</td>"
            f"<td style='padding:2px 6px;text-align:right;'>{r['avg_value']:.4E}</td>"
            f"<td style='padding:2px 6px;text-align:right;'>{r['max_value']:.4E}</td>"
            f"</tr>"
        )

    return (
        f"<div style='max-height:300px;overflow-y:auto;'>"
        f"<table style='font-size:15px;border-collapse:collapse;width:100%;'>"
        f"<tr style='background:#1a5276;color:white;'>"
        f"<th style='padding:3px 6px;'>#</th>"
        f"<th style='padding:3px 6px;'>{label_col}</th>"
        f"<th style='padding:3px 6px;'>Stability</th>"
        f"<th style='padding:3px 6px;'>Avg</th>"
        f"<th style='padding:3px 6px;'>Max</th>"
        f"</tr>"
        f"{''.join(rows)}"
        f"</table></div>"
    )


def get_stable_qubits_html(state, prop_key='readout_error', max_error=0.05,
                            min_days_ratio=0.8, top_n=30):
    """HTML table of stable qubits."""
    results = find_stable_qubits(state, prop_key, max_error, min_days_ratio)
    return _build_stable_html(
        results,
        label_col='Qubit',
        empty_msg=("No stable qubits found "
                   "(try adjusting threshold or fetching more history)."),
        top_n=top_n,
        row_label_fn=lambda r: f"Q{r['qubit']}",
    )

# ── Stable gate finder ───────────────────────────────────────────────

def find_stable_gates(state, prop_key='cz_error', max_error=0.01,
                       min_days_ratio=0.8):
    """Find gates that stayed within an acceptable range across calibration history.

    Direction is controlled by ``HIGHER_IS_BETTER_PROPS`` (currently empty
    for gate-level properties — all gate properties are "lower is better").
    The branch is kept for symmetry with ``find_stable_qubits`` so that
    future gate properties can be added to the higher-is-better set without
    a logic refactor.
    """
    higher_is_better = prop_key in HIGHER_IS_BETTER_PROPS

    if not state.cal_history_dates or not state.cal_history:
        return []

    edges = state.edges_list or []
    stats = {i: {'good': 0, 'total': 0, 'values': []} for i in range(len(edges))}

    for date_str in state.cal_history_dates:
        snap = state.cal_history.get(date_str, {})
        arr = snap.get('gate_props', {}).get(prop_key, np.array([]))
        for i in range(len(edges)):
            if i < len(arr):
                val = arr[i]
                if not np.isnan(val):
                    stats[i]['total'] += 1
                    stats[i]['values'].append(val)
                    if higher_is_better:
                        if val >= max_error:
                            stats[i]['good'] += 1
                    else:
                        if val <= max_error:
                            stats[i]['good'] += 1

    results = []
    for i, s in stats.items():
        if s['total'] == 0:
            continue
        ratio = s['good'] / s['total']
        if ratio >= min_days_ratio:
            e = edges[i]
            results.append({
                'edge': f"({e[0]},{e[1]})",
                'edge_idx': i,
                'good_days': s['good'],
                'total_days': s['total'],
                'ratio': ratio,
                'avg_value': float(np.mean(s['values'])),
                'max_value': float(np.max(s['values'])),
            })

    if higher_is_better:
        results.sort(key=lambda x: (-x['ratio'], -x['avg_value']))
    else:
        results.sort(key=lambda x: (-x['ratio'], x['avg_value']))
    return results


def get_stable_gates_html(state, prop_key='cz_error', max_error=0.01,
                           min_days_ratio=0.8, top_n=0):
    """HTML table of stable gates."""
    results = find_stable_gates(state, prop_key, max_error, min_days_ratio)
    return _build_stable_html(
        results,
        label_col='Gate',
        empty_msg=("No stable gates found "
                   "(try adjusting threshold or fetching more history)."),
        top_n=top_n,
        row_label_fn=lambda r: r['edge'],
    )
