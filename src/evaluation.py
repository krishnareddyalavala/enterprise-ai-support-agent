def evaluate_grounding(answer, retrieved_docs):

    context = " ".join([doc.page_content for doc in retrieved_docs])

    answer_words = answer.lower().split()

    matched_words = 0

    for word in answer_words:

        if word in context.lower():
            matched_words += 1

    grounding_score = matched_words / len(answer_words)

    if grounding_score > 0.5:
        return "Grounded"

    return "Potential Hallucination"