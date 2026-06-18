import streamlit as st
import joblib
import requests
import spacy

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

.stApp{
    background:#050816;
    color:white;
}

/* Animated background */

.stApp::before{
    content:"";
    position:fixed;
    width:600px;
    height:600px;
    background:radial-gradient(
        circle,
        rgba(0,229,255,.15),
        transparent 70%
    );
    top:-200px;
    right:-150px;
    animation:float1 10s ease-in-out infinite;
    z-index:-1;
}

.stApp::after{
    content:"";
    position:fixed;
    width:500px;
    height:500px;
    background:radial-gradient(
        circle,
        rgba(123,97,255,.15),
        transparent 70%
    );
    bottom:-200px;
    left:-150px;
    animation:float2 12s ease-in-out infinite;
    z-index:-1;
}

@keyframes float1{
    0%,100%{transform:translateY(0px);}
    50%{transform:translateY(40px);}
}

@keyframes float2{
    0%,100%{transform:translateY(0px);}
    50%{transform:translateY(-40px);}
}

.hero{
    padding:70px;
    border-radius:30px;

    background:rgba(255,255,255,0.03);

    backdrop-filter:blur(25px);

    border:1px solid rgba(255,255,255,.08);

    text-align:center;

    box-shadow:
    0 0 50px rgba(0,229,255,.15);
}

.hero-title{
    font-size:64px;
    font-weight:900;

    background:
    linear-gradient(
        90deg,
        #00E5FF,
        #7B61FF
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

.hero-subtitle{
    color:#B0B8D1;
    font-size:20px;
}

.badge{
    display:inline-block;
    margin:8px;
    padding:10px 18px;
    border-radius:999px;

    background:
    rgba(0,229,255,.1);

    border:
    1px solid rgba(0,229,255,.3);

    color:#00E5FF;

    font-size:14px;
    font-weight:600;
}

.glass-card{
    background:
    rgba(255,255,255,.03);

    border:
    1px solid rgba(255,255,255,.08);

    border-radius:25px;

    padding:25px;

    backdrop-filter:blur(25px);

    transition:.3s;
}

.glass-card:hover{
    transform:translateY(-5px);
}

.metric-card{
    text-align:center;

    padding:25px;

    border-radius:20px;

    background:
    rgba(255,255,255,.03);

    border:
    1px solid rgba(255,255,255,.08);
}

.metric-value{
    font-size:40px;
    font-weight:800;
    color:#00E5FF;
}

.metric-label{
    color:#B0B8D1;
}

.verdict-card{
    text-align:center;

    padding:40px;

    border-radius:30px;

    margin-top:20px;

    backdrop-filter:blur(20px);
}

.supported{
    background:
    rgba(0,255,149,.1);

    border:
    1px solid rgba(0,255,149,.4);
}

.false{
    background:
    rgba(255,77,109,.1);

    border:
    1px solid rgba(255,77,109,.4);
}

.insufficient{
    background:
    rgba(255,204,0,.1);

    border:
    1px solid rgba(255,204,0,.4);
}

.pipeline-step{
    display:inline-block;

    padding:14px 22px;

    margin:10px;

    border-radius:14px;

    background:
    rgba(255,255,255,.03);

    border:
    1px solid rgba(0,229,255,.25);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">

<div class="hero-title">
TruthLens AI
</div>

<br>

<div class="hero-subtitle">
Fact Verification Powered By Machine Learning,
Semantic Search & Natural Language Inference
</div>

<br>

<span class="badge">AI Powered</span>
<span class="badge">Real-Time Evidence</span>
<span class="badge">Explainable Verdicts</span>

</div>
""", unsafe_allow_html=True)
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")


@st.cache_resource
def load_resources():

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    model = joblib.load("models/logistic_model.pkl")

    nlp = spacy.load("en_core_web_sm")

    similarity_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    nli = pipeline(
        "text-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=-1
    )

    return vectorizer, model, nlp, similarity_model, nli


loaded_vectorizer, loaded_model, nlp, similarity_model, nli = load_resources()

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class='glass-card'>
    <h3 style='color:#FFD700;'>📝 Verify a Claim</h3>
</div>
""", unsafe_allow_html=True)

default_claim = ""

examples = {
    
    "🚀 NASA confirms aliens landed in Chennai":
        "NASA confirms aliens landed in Chennai.",
    "🚗 Elon Musk is not the CEO of Tesla":
        "Elon Musk is not the CEO of Tesla.",
    "🌍 The Earth is flat":
        "The Earth is flat."
}

cols = st.columns(4)

for i, example in enumerate(examples.keys()):
    with cols[i]:
        if st.button(example):
            st.session_state.claim = examples[example]
            st.rerun()

if "claim" not in st.session_state:
    st.session_state.claim = ""

claim = st.text_area(
    "Enter a claim",
    value=st.session_state.claim,
    placeholder="e.g., NASA confirms aliens landed in Chennai.",
    height=120
)

def predict_fake_news(claim):

    x = loaded_vectorizer.transform([claim])

    pred = loaded_model.predict(x)[0]

    probs = loaded_model.predict_proba(x)[0]

    return {
        "prediction": "Fake" if pred == 1 else "Real",
        "fake_probability": float(probs[1])
    }

def generate_query(claim):

    doc = nlp(claim)

    keywords = []

    # Keep entities
    for ent in doc.ents:

        if ent.label_ in {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "DATE"
        }:

            keywords.append(ent.text)

    # Keep important nouns and verbs
    for token in doc:

        if token.pos_ in {
            "NOUN",
            "PROPN",
            "VERB"
        }:

            if not token.is_stop:

                keywords.append(token.text)

    # Remove duplicates
    seen = set()

    final_keywords = []

    for word in keywords:

        word_lower = word.lower()

        if word_lower not in seen:

            seen.add(word_lower)

            final_keywords.append(word)

    return " ".join(final_keywords)

def retrieve_articles(query, page_size=10):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return []

    data = response.json()

    articles = []

    for article in data.get("articles", []):

        articles.append({
            "title": article.get("title"),
            "description": article.get("description"),
            "source": article.get("source", {}).get("name"),
            "publishedAt": article.get("publishedAt"),
            "url": article.get("url")
        })

    return articles

def rank_articles(claim, articles, top_k=3):

    claim_lower = claim.lower()

    filtered_articles = []

    for article in articles:

        text = (
            (article["title"] or "")
            + " "
            + (article["description"] or "")
        ).lower()

        # If claim mentions covid
        if "covid" in claim_lower:

            if "covid" not in text:
                continue

        filtered_articles.append(article)

    articles = filtered_articles

    claim_emb = similarity_model.encode(
        claim,
        convert_to_tensor=True
    )

    ranked = []

    for article in articles:

        text = (
            (article["title"] or "")
            + " "
            + (article["description"] or "")
        )

        article_emb = similarity_model.encode(
            text,
            convert_to_tensor=True
        )

        score = util.cos_sim(
            claim_emb,
            article_emb
        ).item()

        article["similarity"] = score

        ranked.append(article)

    ranked.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return ranked[:top_k]

def verify_evidence(claim, evidence_text):

    result = nli({
        "text": evidence_text,
        "text_pair": claim
    })

    if isinstance(result, list):
        result = result[0]

    return {
        "label": result["label"].upper(),
        "score": float(result["score"])
    }

verify = st.button(
    "🔍 VERIFY CLAIM",
    use_container_width=True
)

if verify and claim.strip():

    with st.spinner("TruthLens AI is analyzing the claim..."):

        historical = predict_fake_news(claim)
        # Search using full claim first
        articles_full = retrieve_articles(claim)

        # Search using generated query
        query = generate_query(claim)

        articles_query = retrieve_articles(query)

        # Merge results
        all_articles = articles_full + articles_query

        # Remove duplicates
        unique_articles = {}

        for article in all_articles:

            url = article.get("url")

            if url:
                unique_articles[url] = article

        articles = list(unique_articles.values())
        # print("Full Claim Articles:", len(articles_full))
        # print("Query Articles:", len(articles_query))
        # print("Merged Articles:", len(articles))

        # print("\n==============================")
        # print("ALL RETRIEVED ARTICLES")
        # print("==============================")

        # for i, article in enumerate(articles):

        #     print(f"\nArticle {i+1}")

        #     print("Title:", article["title"])

        #     print("Source:", article["source"])


        if not articles:
            st.warning("⚠️ No relevant articles were found. Verdict may be inconclusive.")

        top_articles = rank_articles(
            claim,
            articles,
            top_k=3
        ) if articles else []

        evidence_results = []

        final_verdict = "INSUFFICIENT EVIDENCE"

        for article in top_articles:

            evidence = (
                (article["title"] or "")
                + ". "
                + (article["description"] or "")
            )

            nli_result = verify_evidence(
                claim,
                evidence
            )

            evidence_results.append({
                **article,
                **nli_result
            })

            if (
                nli_result["label"] == "CONTRADICTION"
                and nli_result["score"] >= 0.90
            ):
                final_verdict = "FALSE"

            elif (
                nli_result["label"] == "ENTAILMENT"
                and nli_result["score"] >= 0.90
                and final_verdict != "FALSE"
            ):
                final_verdict = "SUPPORTED"
        st.markdown("<br>", unsafe_allow_html=True)

        if final_verdict == "SUPPORTED":

            st.markdown(f"""
            <div class="verdict-card supported">
                <h1>✅ SUPPORTED</h1>
                <h3>Evidence supports the claim</h3>
            </div>
            """, unsafe_allow_html=True)

        elif final_verdict == "FALSE":

            st.markdown(f"""
            <div class="verdict-card false">
                <h1>❌ FALSE</h1>
                <h3>Evidence contradicts the claim</h3>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="verdict-card insufficient">
                <h1>⚠ INSUFFICIENT EVIDENCE</h1>
                <h3>Unable to verify claim confidently</h3>
            </div>
            """, unsafe_allow_html=True)

        col1,col2,col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{len(articles)}</div>
            <div class="metric-label">Articles Retrieved</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{len(evidence_results)}</div>
            <div class="metric-label">Evidence Sources</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{historical['fake_probability']:.0%}</div>
            <div class="metric-label">AI Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("## 📰 Evidence Explorer")
        for evidence in evidence_results:

            with st.expander(evidence["title"]):

                st.write(f"**Source:** {evidence['source']}")
                st.write(f"**Similarity:** {evidence['similarity']:.3f}")
                st.write(f"**NLI Label:** {evidence['label']}")
                st.write(f"**Confidence:** {evidence['score']:.2%}")

                st.link_button(
                    "Open Source",
                    evidence["url"]
                )


        with st.expander("🧠 How did TruthLens AI decide this?"):

            st.markdown("""
<h2>⚙ Verification Pipeline</h2>

<div style="text-align:center">

<span class="pipeline-step">Claim</span>

➡

<span class="pipeline-step">ML Detection</span>

➡

<span class="pipeline-step">News Retrieval</span>

➡

<span class="pipeline-step">SBERT Ranking</span>

➡

<span class="pipeline-step">NLI Verification</span>

➡

<span class="pipeline-step">Final Verdict</span>

</div>
""", unsafe_allow_html=True)
st.markdown("""
<hr>

<div style="text-align:center">

<h3>TruthLens AI</h3>

Machine Learning • Semantic Search • Fact Verification

<br><br>

Developed by Aravind Murugesan

<br>

MCA | AI & ML Engineer

</div>
""", unsafe_allow_html=True)
