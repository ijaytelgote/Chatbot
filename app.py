
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Levenshtein import distance as levenshtein_distance

model = SentenceTransformer("all-MiniLM-L6-v2")
lemmatizer = WordNetLemmatizer()


# ✅ 1. Normalize Text
def normalize(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


# ✅ 2. Synonym Expansion (WordNet)
def expand_synonyms(tokens):
    expanded = set(tokens)
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded.add(lemma.name())
    return expanded


# ✅ 3. Advanced Key-Match Score
def key_match_score(q, a):
    q_tokens = normalize(q)
    a_tokens = normalize(a)

    # expand with synonyms
    q_expanded = expand_synonyms(q_tokens)
    a_expanded = expand_synonyms(a_tokens)

    # direct matches
    direct_matches = q_expanded.intersection(a_expanded)

    # n-gram matches
    def make_ngrams(tokens, n):
        return set([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    q_bigrams = make_ngrams(q_tokens, 2)
    a_bigrams = make_ngrams(a_tokens, 2)
    bigram_matches = q_bigrams.intersection(a_bigrams)

    # score components
    direct_score = len(direct_matches)
    bigram_score = len(bigram_matches) * 2  # bigrams weigh more

    return direct_score + bigram_score, len(q_expanded), len(a_expanded)


# ✅ 4. Key Overlap Percentage
def overlap_percent(q_expanded, a_expanded):
    intersection = q_expanded.intersection(a_expanded)
    if len(q_expanded) == 0:
        return 0
    return len(intersection) / len(q_expanded)


# ✅ 5. Levenshtein Similarity
def levenshtein_similarity(q, a):
    d = levenshtein_distance(q.lower(), a.lower())
    max_len = max(len(q), len(a))
    if max_len == 0:
        return 1
    return 1 - (d / max_len)


# ✅ 6. Semantic Similarity
def semantic_similarity(q, a):
    q_emb = model.encode(q, convert_to_tensor=True)
    a_emb = model.encode(a, convert_to_tensor=True)
    return util.pytorch_cos_sim(q_emb, a_emb).item()


# ✅ 7. Final Relevance Score Pipeline
def relevance_advanced(question, answer):
    # semantic
    semantic = semantic_similarity(question, answer) * 100
    # key match
    key_score, q_len, a_len = key_match_score(question, answer)
    # expanded tokens
    q_expanded = expand_synonyms(normalize(question))
    a_expanded = expand_synonyms(normalize(answer))

    # overlap %
    overlap = overlap_percent(q_expanded, a_expanded) * 100

    # levenshtein
    lev = levenshtein_similarity(question, answer) * 100

    # ✅ Weighted final score
    final_score = (
        (semantic * 0.55) +
        (key_score * 5 * 0.15) +          # scaled
        (overlap * 0.10) +
        (lev * 0.20)
    )

    final_score = min(round(final_score, 2), 100)

    # ✅ Confidence Level
    if final_score >= 70:
        confidence = "High"
    elif final_score >= 40:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "Relevancy Score": final_score,
        "Confidence": confidence,
        "Metadata": {
            "Semantic Similarity": round(semantic, 2),
            "Key Match Score": key_score,
            "Key Overlap %": round(overlap, 2),
            "Levenshtein Similarity": round(lev, 2),

        },
        "Ground Truth": question,
        "Predicted": answer
    }



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()

    text1 = data.get("question", "")
    text2 = data.get("answer", "")
    result=relevance_advanced(text1, text2)


    return jsonify(result), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
