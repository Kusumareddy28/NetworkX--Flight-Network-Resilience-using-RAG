
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import os

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    langchain_available = True
except ImportError:
    langchain_available = False

st.set_page_config(page_title="Flight Network Resilience", layout="wide")
st.title("Flight Network Resilience Dashboard")

uploaded_airports = st.sidebar.file_uploader("Upload airports.csv", type="csv")
uploaded_flights  = st.sidebar.file_uploader("Upload flights.csv", type="csv")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not uploaded_airports or not uploaded_flights:
    st.info("Please upload both airports.csv and flights.csv in the sidebar.")
    st.stop()

airports = pd.read_csv(uploaded_airports)
flights  = pd.read_csv(uploaded_flights)
airports.columns = airports.columns.str.strip()
flights.columns  = flights.columns.str.strip()

if "AiportID" not in airports.columns:
    st.error("Expected 'AiportID' column in airports.csv")
    st.stop()
airports = airports[airports['AiportID'].notna() & (airports['AiportID'] != "\\N")]


G = nx.Graph()
for _, r in airports.iterrows():
    G.add_node(r['AiportID'])
for _, r in flights.iterrows():
    s,d = r['source airport'], r['destination airport']
    if s in G and d in G:
        G.add_edge(s, d)


pos = nx.spring_layout(G, k=0.1, seed=42)


col1, col2 = st.columns([3,1])

with col1:
    st.subheader("Initial Flight Network")
    fig, ax = plt.subplots(figsize=(6,4))
    nx.draw(G, pos, node_size=10, node_color='skyblue', edge_color='gray', width=0.2, alpha=0.5, ax=ax, with_labels=False)
    st.pyplot(fig)

    st.subheader("Centrality Metrics")
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, normalized=True)
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    top_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top 5 by Degree**")
        st.table(pd.DataFrame(top_deg, columns=["Airport","Degree Centrality"]))
    with c2:
        st.write("**Top 5 by Betweenness**")
        st.table(pd.DataFrame(top_bet, columns=["Airport","Betweenness Centrality"]))

    st.subheader("Shortest Path Example")
    try:
        orig = nx.shortest_path(G, 'LAX', 'JFK')
        st.write(f"Original LAX → JFK: {' → '.join(orig)}")
        G_no = G.copy(); G_no.remove_node('ATL')
        new = nx.shortest_path(G_no,'LAX','JFK')
        st.write(f"Without ATL: {' → '.join(new)}")
    except Exception as e:
        st.write(f"Error computing path: {e}")

with col2:
    st.subheader("Resilience Simulation")
    scenario = st.selectbox("Scenario", ['Random Failure','Targeted Attack'])
    k = st.slider("Nodes removed", 0, min(1000, G.number_of_nodes()), 100)
    if st.button("Run Simulation"):
        order = list(G.nodes())
        if scenario=='Targeted Attack':
            order = [n for n,_ in sorted(deg_cent.items(), key=lambda x:x[1], reverse=True)]
        else:
            random.shuffle(order)
        H = G.copy(); sizes=[]
        for node in order[:k]:
            H.remove_node(node)
            comps=[len(c) for c in nx.connected_components(H)]
            sizes.append(max(comps) if comps else 0)
        fig2, ax2 = plt.subplots()
        ax2.plot(sizes, label=scenario)
        ax2.set_xlabel("Removals"); ax2.set_ylabel("LCC Size")
        ax2.set_title(f"{scenario} (first {k})"); ax2.legend()
        st.pyplot(fig2)


with st.expander("Cancellation Risk Analysis"):
    for u,v in G.edges():
        G[u][v]['cancel_rate'] = np.random.beta(2,8)
    top_edges = sorted(G.edges(data='cancel_rate'), key=lambda x:x[2], reverse=True)[:30]
    fig3, ax3 = plt.subplots(figsize=(6,4))
    nx.draw(G, pos, node_size=5, node_color='lightgray', edge_color='lightgray', alpha=0.2, ax=ax3)
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v,_ in top_edges], width=2, edge_color='red', ax=ax3)
    st.pyplot(fig3)

with st.expander("Delay-Weighted Analysis"):
    for u,v in G.edges():
        G[u][v]['avg_delay'] = max(0, np.random.normal(15,5))
    delay_path = nx.shortest_path(G, 'LAX','JFK', weight='avg_delay')
    st.write(f"Min-delay LAX → JFK: {' → '.join(delay_path)}")

with st.expander("Weather Risk Analysis"):
    for n in G.nodes():
        G.nodes[n]['weather_risk'] = np.random.beta(2,5)
    topw = sorted(G.nodes(data='weather_risk'), key=lambda x:x[1], reverse=True)[:10]
    st.write("Top weather-risk airports:", [n for n,_ in topw])

with st.expander("Ask the Network (Q&A)"):
    question = st.text_input("Your question:")
    if st.button("Ask") and question:
        docs = [f"Top 5 hubs: {[a for a,_ in top_deg]}", f"Nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"]
        if langchain_available and api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_texts(docs, embeddings)
            qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type='stuff', retriever=db.as_retriever())
            ans = qa.run(question)
        else:
            tokens = set(question.lower().split())
            ans = max(docs, key=lambda d: len(tokens & set(d.lower().split())))
        st.write("**Answer:**", ans)
