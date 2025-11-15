# Member Question-Answering Service

A robust, deterministic microservice that answers natural-language questions about members using only the messages returned from an external `/messages` API.  

## Live Demo

**SWAGER UI:**

https://qa-system-gstx.onrender.com/docs

Example query:

https://qa-system-gstx.onrender.com/ask?q=Where%20does%20Layla%20wants%20to%20book%20a%20villa%3F

https://qa-system-gstx.onrender.com/ask?q=How%20many%20cars%20does%20Vikram%20Desai%20have%3F


The service is intentionally **non-hallucinatory**, **conservative**, and **rule-driven**. It returns an answer **only when explicit evidence exists in the dataset**. Otherwise, it safely responds:

```
"I don't know based on the available messages."
```

This approach ensures correctness, safety, and determinism—exactly what is required in production AI systems and for this assessment.

---

# Project Structure

```
member_qa_service_full/
│
├── app/
│   ├── main.py                 # Core FastAPI application (final logic lives here)
│
├── scripts/
│   └── inspect_data.py         # Helper script for dataset analysis
│
├── Dockerfile
├── requirements.txt
└── README.md                   # This file
```

---

# Overview

The service answers questions such as:

- “Where is Ayesha traveling next?”
- “What does Hans Müller prefer for hotel rooms?”
- “How many cars does Vikram Desai have?”
- “When is Layla planning her trip to London?”

It does this by:

1. Fetching messages from the upstream API  
2. Matching the member mentioned in the question  
3. Ranking relevant messages using TF-IDF  
4. Extracting factual answers using deterministic rules and regex  
5. Falling back to a snippet only for open-ended questions

No part of the pipeline uses generative AI; answers are always extracted from the data provided.

---

# Key Features

### 1. Zero-hallucination guarantee
Factual questions (numeric, when, where, favorite) produce an answer **only** if evidence is found in a message.  
Otherwise → `"I don't know based on the available messages."`

---

### 2. Accurate member identification
- Extracts names from the question  
- Normalizes them (“Amira’s” → “Amira”)  
- Matches them to real members  
- If a named member is not found → refuse to answer

---

### 3. Destination-aware WHEN logic
A question like:

> “When is Layla planning her trip to London?”

requires BOTH:
- Layla  
- London  

in the **same** message.  
If not → `"I don't know..."`

This prevents cross-context contamination and accidental fabrications.

---

### 4. TF-IDF ranking for relevance
The service performs lightweight semantic ranking using Scikit-Learn’s TF-IDF vectorizer before extraction.

---

### 5. Deterministic extraction
Regex-based extractors handle:

- Dates  
- Locations  
- Numbers  
- Preferences  

This ensures stability and clear behavior with no generative guesses.

---

# Installation

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

# Running with Docker

### Build:
```bash
docker build -t member-qa-service .
```

### Run:
```bash
docker run -p 8080:8080 member-qa-service
```

---

# API Endpoints

### Health Check
```
GET /health
```

### Ask a Question
```
GET /ask?q=Your+question+here
```

---

# Test Cases (with Expected Outputs)

## 1. Correct factual extraction
**Request:**  
`Where is Ayesha traveling next?`  
**Expected:**  
```json
{ "answer": "Dubai" }
```

---

## 2. Preference extraction
**Request:**  
`What does Hans Müller prefer for hotel rooms?`  
**Expected:**  
```json
{
  "answer": "Remember that I have a preference for quiet hotel rooms."
}
```

---

## 3. Known member, missing explicit fact
**Request:**  
`How many cars does Vikram Desai have?`  
**Expected:**  
```json
{
  "answer": "I don't know based on the available messages."
}
```

---

## 4. Destination-sensitive WHEN logic
**Request:**  
`When is Layla planning her trip to London?`  
**Expected:**  
```json
{
  "answer": "I don't know based on the available messages."
}
```

---

## 5. Unknown member
**Request:**  
`When is Michael’s hotel reservation scheduled for?`  
**Expected:**  
```json
{
  "answer": "I don't know based on the available messages."
}
```

---

## 6. Open-ended snippet fallback
**Request:**  
`Tell me about Ayesha’s recent travel plans.`  
**Expected:**  
A real travel snippet from Ayesha's messages.

---

# Design Decisions

1. **Zero-Hallucination Policy**
2. **Rule-based Extraction**
3. **Semantic Ranking (TF-IDF)**
4. **Destination + Member Co-presence Requirement**
5. **Safe Snippet Fallbacks**

---

## Design Notes – Alternative Approaches Considered

While implementing this service, I considered several different approaches to the question-answering problem:

### 1. Retrieval + Large Language Model (LLM) Generation

**Idea:**  
Use embeddings (e.g., OpenAI / other vector models) to retrieve the most relevant messages for a member, then ask an LLM to generate a natural-language answer.

**Pros:**
- Very flexible; can handle complex, fuzzy questions.
- Less manual rule-writing.

**Cons (for this assignment):**
- Harder to make truly deterministic and non-hallucinatory.
- Requires external model dependencies, API keys, and cost.
- Explaining exactly *why* a specific answer was produced is more difficult.

**Reason not chosen:**  
The assignment emphasizes reliability, explainability, and not hallucinating. A pure LLM approach would make it harder to guarantee those properties.

---

### 2. Embedding-based Semantic Search (without generation)

**Idea:**  
Instead of TF–IDF, use sentence embeddings (e.g., sentence-transformers) to rank member messages by semantic similarity, then still do rule-based extraction for numbers/dates/locations.

**Pros:**
- Better semantic understanding than TF–IDF.
- More robust to paraphrasing.

**Cons:**
- Heavier dependency footprint (models, weights).
- Slower and more complex to deploy.
- Overkill for a relatively small dataset and straightforward questions.

**Reason not chosen:**  
TF–IDF is simpler, fast, and good enough for the size and style of the dataset, while still being deterministic.

---

### 3. Precomputed Structured Profiles per Member

**Idea:**  
Pre-scan all messages to build a “profile” per member (e.g., number of kids, pets, preferred cities, hotel preferences, etc.), store this in a small in-memory index or database, and answer questions from that structure.

**Pros:**
- Very fast at query time.
- Easy to enforce constraints (e.g., numeric consistency).

**Cons:**
- More complex preprocessing pipeline.
- Easy to miss edge cases when manually designing all fields.
- Any new question type might require updating the preprocessing logic.

**Reason not chosen:**  
For a relatively small dataset and limited question set, on-the-fly analysis using TF–IDF + extraction strikes a better balance between flexibility and complexity.

---

### 4. Pure Keyword / Regex Matching (No TF–IDF)

**Idea:**  
Skip TF–IDF entirely and just scan all messages with keyword rules like:
- Must contain the member’s name
- Must contain “car(s)” / “trip” / “hotel”, etc.

**Pros:**
- Very simple implementation.
- No vectorization step.

**Cons:**
- Ranking is brittle; multiple messages can match with no good way to pick the best one.
- Harder to generalize to more open-ended questions.

**Reason not chosen:**  
I wanted something more robust than pure keyword matching, but still deterministic and easy to reason about. TF–IDF is a good middle ground.

---

### Final Choice

The final design uses:

- **Member name detection + filtering**
- **TF–IDF for relevance ranking**
- **Rule-based extraction (regex) for numeric, date, location, and preference facts**
- **Conservative fallbacks** that return  
  `"I don't know based on the available messages."`  
  whenever evidence is insufficient.

This gives a good balance of **simplicity, explainability, and safety**.

---

## Data Insights – Dataset Anomalies & Inconsistencies

I analyzed the upstream `/messages` dataset using `scripts/inspect_data.py`. The script checks for empty texts, duplicate messages, unparseable timestamps, and conflicting numeric “facts” per member (e.g., different counts of cars, children, or pets mentioned at different times).

From the current dataset:

- **Duplicate messages:** 0 distinct texts appeared more than once.  
  - This suggests there is no significant duplication of member messages, which simplifies retrieval logic.

- **Unparseable timestamps:** 100 messages contained timestamps that did not match expected formats.  
  - In practice, this means timestamps are not reliable enough to be used for strict time-based reasoning (e.g., ordering or “most recent” logic).  
  - The current implementation intentionally does **not** depend on timestamps; all reasoning is based on message content only.

- **Conflicting numeric facts:** 0 members appeared to have conflicting numeric counts (e.g., different numbers of cars, kids, or pets).  
  - This supports the assumption that simple numeric extraction per member is safe in this dataset.

These observations influenced the design: the service focuses on **content-based extraction and ranking**, and intentionally ignores timestamps to avoid building logic on top of inconsistent time metadata.
