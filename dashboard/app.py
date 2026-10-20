from __future__ import annotations

import os
import sys
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings, DEFAULT_PRICING
from app.database.repository import repository
from app.cache.embeddings import EmbeddingFactory
from app.schemas import RoutingStrategy

# Page configuration
st.set_page_config(
    page_title="SmartRoute AI Gateway",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-themed Developer Tool CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #58a6ff;
        font-family: monospace;
    }
    .metric-sub {
        font-size: 13px;
        color: #8b949e;
        margin-top: 4px;
    }
    .badge-exact {
        background-color: #238636;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-semantic {
        background-color: #1f6feb;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-miss {
        background-color: #da3633;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        color: #c9d1d9;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        border-bottom: 2px solid #58a6ff !important;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{settings.PORT}")


def check_api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def query_gateway(
    query_text: str,
    namespace: str = "default",
    force_refresh: bool = False,
    strategy: str = "rule_based"
) -> dict:
    # Try calling via REST API first
    try:
        payload = {
            "query": query_text,
            "namespace": namespace,
            "force_refresh": force_refresh,
            "routing_strategy": strategy if strategy != "default" else None
        }
        r = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=settings.REQUEST_TIMEOUT_SECONDS)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # Direct in-process fallback if standalone dashboard is running
    from app.cache.semantic_cache import semantic_cache
    from app.routing.router import RouterFactory
    from app.routing.pricing import calculate_savings_vs_baseline
    from app.providers.factory import ProviderFactory
    import uuid
    import asyncio
    from datetime import datetime, timezone

    start = time.time()
    req_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    # Cache check
    if not force_refresh and settings.CACHE_ENABLED:
        hit = semantic_cache.lookup(query_text, namespace)
        if hit and hit.hit:
            lat = round((time.time() - start) * 1000.0, 2)
            saved = calculate_savings_vs_baseline(hit.prompt_tokens, hit.completion_tokens, hit.model_name, is_cache_hit=True)
            res = {
                "request_id": req_id, "query": query_text, "answer": hit.answer,
                "cache_hit": True, "cache_type": hit.cache_type.value, "similarity_score": hit.similarity_score,
                "matched_query": hit.cached_query, "selected_model": hit.model_name, "model_tier": hit.model_tier,
                "routing_reason": f"Cached answer ({hit.cache_type.value})", "routing_strategy": "cache",
                "prompt_tokens": hit.prompt_tokens, "completion_tokens": hit.completion_tokens,
                "total_tokens": hit.prompt_tokens + hit.completion_tokens, "estimated_cost": 0.0,
                "actual_cost": 0.0, "cost_saved": saved, "latency_ms": lat, "provider": "cache",
                "is_mock": False, "namespace": namespace, "timestamp": ts
            }
            repository.log_request({
                "request_id": req_id, "timestamp": ts, "query_text": query_text, "answer": hit.answer,
                "cache_hit": 1, "cache_type": hit.cache_type.value, "similarity_score": hit.similarity_score,
                "selected_model": hit.model_name, "model_tier": hit.model_tier, "routing_strategy": "cache",
                "routing_reason": res["routing_reason"], "prompt_tokens": hit.prompt_tokens,
                "completion_tokens": hit.completion_tokens, "total_tokens": res["total_tokens"],
                "estimated_cost": 0.0, "actual_cost": 0.0, "cost_saved": saved, "latency_ms": lat,
                "namespace": namespace, "provider": "cache", "is_mock": 0, "error": None
            })
            return res

    # Route & Call
    ov_strat = None
    if strategy in [e.value for e in RoutingStrategy]:
        ov_strat = RoutingStrategy(strategy)
    decision = RouterFactory.route_query(query_text, ov_strat)
    provider = ProviderFactory.get_provider()
    
    # In asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llm_res = loop.run_until_complete(provider.generate(query_text, decision.model_name))
    loop.close()

    lat = round((time.time() - start) * 1000.0, 2)
    saved = calculate_savings_vs_baseline(llm_res.prompt_tokens, llm_res.completion_tokens, llm_res.model)

    if settings.CACHE_ENABLED:
        semantic_cache.store(
            namespace=namespace, query=query_text, answer=llm_res.content,
            model_name=llm_res.model, model_tier=decision.tier.value,
            prompt_tokens=llm_res.prompt_tokens, completion_tokens=llm_res.completion_tokens,
            estimated_cost=llm_res.estimated_cost
        )

    res = {
        "request_id": req_id, "query": query_text, "answer": llm_res.content,
        "cache_hit": False, "cache_type": "none", "similarity_score": None,
        "matched_query": None, "selected_model": llm_res.model, "model_tier": decision.tier.value,
        "routing_reason": decision.reason, "routing_strategy": decision.strategy,
        "prompt_tokens": llm_res.prompt_tokens, "completion_tokens": llm_res.completion_tokens,
        "total_tokens": llm_res.total_tokens, "estimated_cost": llm_res.estimated_cost,
        "actual_cost": llm_res.actual_cost, "cost_saved": saved, "latency_ms": lat,
        "provider": llm_res.provider, "is_mock": llm_res.is_mock, "namespace": namespace,
        "timestamp": ts
    }
    repository.log_request({
        "request_id": req_id, "timestamp": ts, "query_text": query_text, "answer": llm_res.content,
        "cache_hit": 0, "cache_type": "none", "similarity_score": None,
        "selected_model": llm_res.model, "model_tier": decision.tier.value, "routing_strategy": decision.strategy,
        "routing_reason": decision.reason, "prompt_tokens": llm_res.prompt_tokens,
        "completion_tokens": llm_res.completion_tokens, "total_tokens": llm_res.total_tokens,
        "estimated_cost": llm_res.estimated_cost, "actual_cost": llm_res.actual_cost, "cost_saved": saved,
        "latency_ms": lat, "namespace": namespace, "provider": llm_res.provider, "is_mock": 1 if llm_res.is_mock else 0,
        "error": None
    })
    return res


# Sidebar: System Status & Controls
with st.sidebar:
    st.title("⚡ SmartRoute")
    st.caption("Cost-Aware AI Gateway & Cache")
    st.divider()

    health = check_api_health()
    is_online = bool(health)
    provider_mode = health.get("provider", settings.LLM_PROVIDER).upper()
    is_mock = health.get("is_mock", (settings.LLM_PROVIDER == "mock"))

    # Provider Mode Status Pill
    if is_mock:
        st.warning("⚠️ Mode: MOCK SIMULATION\n\n(No external API calls; zero cost)")
    else:
        st.success(f"🟢 Mode: LIVE API ({provider_mode})")

    st.markdown("### Gateway Configuration")
    st.write(f"**Embedding Model:** `{settings.EMBEDDING_MODEL_NAME}`")
    st.write(f"**Similarity Threshold:** `{settings.CACHE_SIMILARITY_THRESHOLD}`")
    st.write(f"**Tier 1 (Small):** `{settings.SMALL_MODEL_NAME}`")
    st.write(f"**Tier 2 (Large):** `{settings.LARGE_MODEL_NAME}`")

    st.divider()
    if st.button("🔄 Refresh Dashboard Data", use_container_width=True):
        st.rerun()

# Fetch latest metrics directly from persistent SQLite repository
metrics = repository.get_aggregated_metrics()
cache_stats = repository.get_cache_stats(time.time())

# Main Navigation Tabs
tab_overview, tab_playground, tab_logs, tab_cache, tab_arch = st.tabs([
    "📊 Performance & Analytics",
    "🎮 Query Playground",
    "📜 Request Telemetry",
    "🧠 Cache Explorer",
    "🏗️ Architecture & Pricing"
])

# ----------------- TAB 1: OVERVIEW & ANALYTICS -----------------
with tab_overview:
    st.subheader("Gateway KPI Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Total Requests</div>
            <div class="metric-val">{metrics['total_requests']:,}</div>
            <div class="metric-sub">Processed by Gateway</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Cache Hit Rate</div>
            <div class="metric-val" style="color: #3fb950;">{metrics['cache_hit_rate_pct']:.1f}%</div>
            <div class="metric-sub">{metrics['cache_hits']} hits / {metrics['cache_misses']} misses</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Total LLM Cost</div>
            <div class="metric-val" style="color: #f0883e;">${metrics['total_estimated_cost_usd']:.4f}</div>
            <div class="metric-sub">Estimated API Spend</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Cost Saved</div>
            <div class="metric-val" style="color: #2ea043;">${metrics['total_cost_saved_usd']:.4f}</div>
            <div class="metric-sub">Savings: {metrics['cost_reduction_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Avg / p95 Latency</div>
            <div class="metric-val" style="color: #a371f7;">{metrics['avg_latency_ms']:.0f}ms</div>
            <div class="metric-sub">p95: {metrics['p95_latency_ms']:.0f}ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Cache Hits Distribution")
        exact_h = metrics["exact_cache_hits"]
        sem_h = metrics["semantic_cache_hits"]
        misses = metrics["cache_misses"]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Exact Hits", "Semantic Hits", "Cache Misses"],
            values=[exact_h, sem_h, misses],
            hole=0.55,
            marker=dict(colors=["#238636", "#1f6feb", "#da3633"])
        )])
        fig_pie.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=280,
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("##### Model Tier Selection Distribution")
        model_dist = metrics.get("model_distribution", {})
        if model_dist:
            fig_bar = px.bar(
                x=list(model_dist.keys()),
                y=list(model_dist.values()),
                labels={"x": "Model", "y": "Request Count"},
                color=list(model_dist.keys()),
                color_discrete_sequence=["#58a6ff", "#bc8cff", "#f0883e"]
            )
            fig_bar.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                height=280,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No requests recorded yet. Submit queries in the Playground to generate telemetry.")

    # Cost Savings Analysis Box
    st.markdown("##### 💡 Cost Savings Methodology")
    st.info(
        f"**Baseline Assumption:** Without SmartRoute, all incoming traffic is dispatched to the un-cached, premium Tier 2 model (`{settings.LARGE_MODEL_NAME}`).\n\n"
        f"• **Semantic Cache Savings:** A cache hit incurs \$0.00 API cost, saving 100% of the baseline cost.\n\n"
        f"• **Cost-Aware Routing Savings:** Straightforward queries are routed to Tier 1 (`{settings.SMALL_MODEL_NAME}`), saving the price differential between Large and Small tiers.\n\n"
        f"• **Net Savings:** SmartRoute has saved **\${metrics['total_cost_saved_usd']:.4f}** out of a gross baseline of **\${(metrics['total_estimated_cost_usd'] + metrics['total_cost_saved_usd']):.4f}** ({metrics['cost_reduction_pct']:.1f}% net reduction)."
    )


# ----------------- TAB 2: QUERY PLAYGROUND -----------------
with tab_playground:
    st.subheader("🎮 Interactive Gateway Playground")
    st.caption("Submit queries and inspect real-time semantic caching, routing decisions, and cost metrics.")

    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        user_query = st.text_area(
            "Enter your prompt / question:",
            value="What is gradient descent and how does the learning rate affect optimization?",
            height=100
        )
    with p_col2:
        namespace_input = st.text_input("Namespace Scope", value="default")
        routing_choice = st.selectbox(
            "Routing Strategy",
            options=["rule_based", "classifier", "direct_small", "direct_large"],
            index=0
        )
        force_bypass = st.checkbox("Bypass Cache (Force Refresh)", value=False)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        submit_btn = st.button("🚀 Send to Gateway", type="primary", use_container_width=True)
    with col_btn2:
        test_paraphrase_btn = st.button("🔁 Test Paraphrase", use_container_width=True)

    if test_paraphrase_btn:
        user_query = "Can you explain how gradient descent works in machine learning?"
        st.info(f"Loaded paraphrased query: *'{user_query}'*")

    if submit_btn or test_paraphrase_btn:
        if not user_query.strip():
            st.error("Please enter a valid prompt.")
        else:
            with st.spinner("Processing through SmartRoute Gateway..."):
                response = query_gateway(
                    query_text=user_query,
                    namespace=namespace_input,
                    force_refresh=force_bypass,
                    strategy=routing_choice
                )

            # Display Results Cards
            st.markdown("---")
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                hit_badge = "🟢 HIT (EXACT)" if response["cache_type"] == "exact" else ("🔵 HIT (SEMANTIC)" if response["cache_type"] == "semantic" else "🔴 CACHE MISS")
                st.metric("Cache Status", hit_badge)
            with r_col2:
                st.metric("Selected Model", f"{response['selected_model']} ({response['model_tier'].upper()})")
            with r_col3:
                st.metric("Latency", f"{response['latency_ms']:.1f} ms")
            with r_col4:
                st.metric("Estimated Cost", f"${response['estimated_cost']:.6f}", delta=f"Saved ${response['cost_saved']:.6f}")

            # Routing Explanation Callout
            st.markdown(f"**Routing Reason:** `{response['routing_reason']}`")
            if response.get("similarity_score"):
                st.markdown(f"**Cosine Similarity Score:** `{response['similarity_score']:.4f}` (Matched with: *'{response.get('matched_query')}'*)")

            # Generated Answer Box
            st.markdown("##### Generated Answer:")
            st.markdown(f"> {response['answer']}")

            # Detailed JSON Inspector
            with st.expander("🔍 Inspect Full Gateway Telemetry Payload (JSON)"):
                st.json(response)


# ----------------- TAB 3: REQUEST LOGS -----------------
with tab_logs:
    st.subheader("📜 Request Telemetry Logs")
    st.caption("Inspect live immutable request events recorded by the gateway.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        f_model = st.selectbox("Filter by Model", options=["All"] + list(DEFAULT_PRICING.keys()), index=0)
    with filter_col2:
        f_cache = st.selectbox("Filter by Cache Status", options=["All", "Cache Hits Only", "Cache Misses Only"], index=0)
    with filter_col3:
        search_query = st.text_input("Search Prompt Keywords", value="")

    # Fetch logs from DB
    cache_hit_filter = True if f_cache == "Cache Hits Only" else (False if f_cache == "Cache Misses Only" else None)
    model_filter = f_model if f_model != "All" else None

    logs, total_logs = repository.get_request_history(
        limit=100,
        model=model_filter,
        cache_hit=cache_hit_filter
    )

    if logs:
        df_logs = pd.DataFrame(logs)
        if search_query:
            df_logs = df_logs[df_logs["query_text"].str.contains(search_query, case=False, na=False)]

        # Display concise columns in main table
        display_cols = ["timestamp", "query_text", "cache_hit", "cache_type", "selected_model", "latency_ms", "estimated_cost", "cost_saved"]
        st.dataframe(
            df_logs[display_cols],
            use_container_width=True,
            column_config={
                "timestamp": "Timestamp (UTC)",
                "query_text": "User Prompt",
                "cache_hit": st.column_config.CheckboxColumn("Hit?"),
                "cache_type": "Match Type",
                "selected_model": "Model",
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
                "estimated_cost": st.column_config.NumberColumn("Cost ($)", format="$%.6f"),
                "cost_saved": st.column_config.NumberColumn("Saved ($)", format="$%.6f")
            }
        )

        with st.expander("🔎 Expand to Inspect Individual Request Details"):
            selected_idx = st.selectbox("Select Request Index to Inspect", options=range(len(df_logs)), format_func=lambda i: f"#{i+1}: {df_logs.iloc[i]['query_text'][:60]}...")
            st.json(df_logs.iloc[selected_idx].to_dict())
    else:
        st.info("No matching request logs found.")


# ----------------- TAB 4: CACHE EXPLORER -----------------
with tab_cache:
    st.subheader("🧠 Semantic Cache Explorer & Vector Analyzer")
    st.caption("Inspect persisted embedding vectors, test query similarity, and manage cache entries.")

    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    with c_col1:
        st.metric("Total Entries", cache_stats["total_entries"])
    with c_col2:
        st.metric("Active Entries", cache_stats["active_entries"])
    with c_col3:
        st.metric("Expired Entries", cache_stats["expired_entries"])
    with c_col4:
        st.metric("Namespaces", len(cache_stats["namespaces"]))

    st.markdown("---")
    st.markdown("##### 🧪 Test Cosine Similarity Between Two Queries")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        sim_q1 = st.text_input("Query 1:", value="How do I train a linear regression model in scikit-learn?")
    with t_col2:
        sim_q2 = st.text_input("Query 2:", value="Guide for fitting linear regression using sklearn in Python")

    if st.button("Calculate Vector Similarity", use_container_width=False):
        embedder = EmbeddingFactory.get_embedding_model()
        v1 = embedder.embed_text(sim_q1)
        v2 = embedder.embed_text(sim_q2)
        score = float(px.pd.np.dot(v1, v2)) if hasattr(px.pd, "np") else float(sum(a*b for a, b in zip(v1, v2)))
        meets = score >= settings.CACHE_SIMILARITY_THRESHOLD

        st.write(f"**Cosine Similarity:** `{score:.4f}`")
        if meets:
            st.success(f"✅ Exceeds threshold ({settings.CACHE_SIMILARITY_THRESHOLD}) — Eligible for Semantic Cache Hit!")
        else:
            st.warning(f"❌ Below threshold ({settings.CACHE_SIMILARITY_THRESHOLD}) — Would trigger Cache Miss.")

    st.markdown("---")
    st.markdown("##### 🗑️ Cache Maintenance")
    del_col1, del_col2 = st.columns(2)
    with del_col1:
        clear_ns = st.text_input("Clear specific namespace (leave empty for global):", value="")
    with del_col2:
        st.write("")
        st.write("")
        if st.button("⚠️ Clear Cache Entries", type="secondary"):
            cleared = repository.clear_cache(clear_ns if clear_ns else None)
            st.success(f"Cleared {cleared} cache entries.")
            st.rerun()


# ----------------- TAB 5: ARCHITECTURE & PRICING -----------------
with tab_arch:
    st.subheader("🏗️ System Architecture & Model Pricing Configuration")

    st.markdown("""
    #### SmartRoute AI Gateway Pipeline Flow:
    ```
    [User Client] ──► POST /query
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [Cache Lookup]             [Query Embedding]
               │                         │
      (Cosine Sim >= 0.85?)              │
        ├── YES ──► Return Cached Answer (0 Cost, <20ms)
        │
        └── NO ──► [Cost-Aware Router]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   [Tier 1: Small]                 [Tier 2: Large]
   (e.g., gpt-4o-mini)             (e.g., gpt-4o)
   • Short queries                 • Complex reasoning
   • Definitions / Lookups         • Code generation / debugging
   • Fact retrieval                • Multi-step system design
        │                                 │
        └────────────────┬────────────────┘
                         ▼
               [LLM Provider Call]
                         │
                         ▼
        [Store in Cache & Record Telemetry]
                         │
                         ▼
              [Client JSON Response]
    ```
    """)

    st.markdown("#### Centralized Model Pricing Card (per 1M tokens)")
    pricing_rows = []
    for model_name, p in DEFAULT_PRICING.items():
        pricing_rows.append({
            "Model Identifier": model_name,
            "Input Cost ($ / 1M tokens)": f"${p['input_cost_per_million']:.2f}",
            "Output Cost ($ / 1M tokens)": f"${p['output_cost_per_million']:.2f}",
            "Tier Classification": "Tier 1 (Small)" if "mini" in model_name or "haiku" in model_name or "8b" in model_name else "Tier 2 (Large)"
        })
    st.table(pd.DataFrame(pricing_rows))
