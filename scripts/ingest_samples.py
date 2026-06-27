"""
scripts/ingest_samples.py - Seeds sample AI research documents
Run: python scripts/ingest_samples.py
"""
import urllib.request
import urllib.error
import json
import time

API_BASE = "http://127.0.0.1:8000/api/v1"

SAMPLE_DOCS = [
    {
        "title": "Attention Is All You Need",
        "source": "https://arxiv.org/abs/1706.03762",
        "text": "The Transformer architecture, introduced by Vaswani et al. at Google Brain in 2017, revolutionised natural language processing. Unlike previous sequence-to-sequence models that relied on recurrent neural networks (RNNs), the Transformer uses self-attention to draw global dependencies between input and output. The key innovation is the Multi-Head Attention layer. Google Brain researchers including Ashish Vaswani, Noam Shazeer, and Jakob Uszkoreit published this work. The architecture enabled models like BERT, GPT, and T5.",
    },
    {
        "title": "GPT-4 Technical Report",
        "source": "https://arxiv.org/abs/2303.08774",
        "text": "GPT-4 is a large multimodal model developed by OpenAI that accepts image and text inputs and produces text outputs. It exhibits human-level performance on various professional and academic benchmarks. OpenAI trained GPT-4 using Reinforcement Learning from Human Feedback (RLHF). GPT-4 was evaluated on standardised tests including the SAT, GRE, and AP exams. OpenAI CEO Sam Altman announced GPT-4 in March 2023.",
    },
    {
        "title": "Gemini: A Family of Highly Capable Multimodal Models",
        "source": "https://arxiv.org/abs/2312.11805",
        "text": "Gemini is Google DeepMind's most capable and general model family, built to be natively multimodal. Gemini was jointly trained on text, images, audio, and video. The Gemini family includes three sizes: Ultra, Pro, and Nano. Gemini Ultra became the first model to surpass human expert performance on MMLU achieving 90.0% accuracy. Google DeepMind, led by CEO Demis Hassabis, developed Gemini. The model was trained using Google's custom Tensor Processing Units (TPUs).",
    },
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "source": "https://arxiv.org/abs/2212.08073",
        "text": "Constitutional AI (CAI) is a training methodology developed by Anthropic that uses a set of principles to guide AI systems toward being helpful, harmless, and honest. CAI uses the AI itself to critique and revise outputs. The process has two phases: supervised learning from AI feedback and reinforcement learning from AI feedback. Anthropic's Claude models are trained using Constitutional AI. Researchers Jared Kaplan and Amanda Askell at Anthropic led this work.",
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "source": "https://arxiv.org/abs/2005.11401",
        "text": "Retrieval-Augmented Generation (RAG) combines parametric memory with non-parametric memory via a dense vector retrieval system. The RAG model retrieves relevant documents using a dense passage retriever, then passes both the query and retrieved documents to a language model to generate the final answer. Patrick Lewis and Ethan Perez at Facebook AI Research published this work in 2020. RAG substantially outperformed purely parametric models on open-domain QA benchmarks.",
    },
]


def post(path, data):
    url = f"{API_BASE}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def get(path):
    url = f"{API_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


print("Seeding AI Research Agent with sample documents...\n")

health = get("/health")
print(f"Health: {health}\n")

for i, doc in enumerate(SAMPLE_DOCS, 1):
    print(f"[{i}/{len(SAMPLE_DOCS)}] Ingesting: {doc['title'][:50]}...")
    status, result = post("/ingest", doc)
    if status == 200:
        print(f"  OK doc_id={result['doc_id']} chunks={result['chunks']} entities={result['entities']}")
    else:
        print(f"  ERROR {status}: {str(result)[:200]}")
    time.sleep(2)  # small pause between docs

print("\nDone! Check health:")
print(get("/health"))
