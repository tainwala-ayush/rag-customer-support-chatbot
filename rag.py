from sentence_transformers.util import cos_sim
import ollama


def smartChunking(text, size = 300, overlap = 80):
    words = text.split()
    totalwords = len(words)
    chunks = []

    if totalwords < size:
        return [text]
    
    step = size - overlap
    for i in range(0, totalwords, step):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    if len(chunks) > 1 and len(chunks[-1].split()) < overlap:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    return chunks

def convertFile(FILES):
    docs, all_chunks, metadata = [], [], []
    
    for file in FILES:
        text = file.read().decode("utf-8")
        docs.append({
            "filename":file,
            "text":text
        })
    for doc in docs:
        smartChunks = smartChunking(doc["text"])
        for i, chunk in enumerate(smartChunks):
            all_chunks.append(chunk)
            metadata.append({
             "filename":doc['filename'].name,
             "id":i,
             "chunk_length":len(chunk.split())
            })
    
    return all_chunks, metadata

def semanticSearch(all_chunks, metadata, embeddings, query, threshold = .55):

    similarity = cos_sim(query, embeddings)
    kMin = len(all_chunks)
    topResults = similarity[0].topk(k=kMin)

    maxScore = 0
    results = []
    for score, idx in zip(topResults.values, topResults.indices):
        maxScore = max(maxScore, score.item())

        if score < threshold:
            continue

        result = {
            "text":all_chunks[idx.item()],
            "filename":metadata[idx.item()]['filename'],
            "similarity":score.item()
        }

        results.append(result)

    return results, maxScore

def buildPrompt(query, retrievedChunks):
    context = "\n\n".join([
            f"{r['filename']}\n{r['text']}" for r in retrievedChunks
        ])

    prompt = f""" 
    ## INSTRUCTIONS: 
    You are a CHATBOT for a company dealing with assistance in PAYMENT, REFUND, and DELIVERY ISSUES. Your PRIMARY TASK is to HELP CUSTOMERS WITH THEIR QUERIES.

    ## CONTEXT:
    {context}

    ## USER INPUT:
    {query}

    ANSWER CLEARLY AND DO NOT MIX UNRELATED IDEAS! """
    
    return prompt

def ask_bot(query, results, maxScore, threshold = 0.55):
    use_content = len(results) > 0 and maxScore >= threshold
    similarity, filename, text = [], [], []
    if use_content:
        prompt = buildPrompt(query, results)
        response = ollama.chat(
            model = "mistral",
            messages=[
                {
                    "role":"system",
                    "content":"You are a strict RAG assistant. Use the content ONLY AND ONLY IF IT IS RELEVANT! If context is IRRELEVANT, ignore it completely and answer using general knowledge. Do not force unrelated information onto the answer."
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )
        for r in results:
            similarity.append(r['similarity'])
            filename.append(r['filename'])
            text.append(r['text'])
        return response["message"]["content"], similarity, filename, text
    
    else:
        response = "I cannot help you with that question, please ask another question!"
        return response
    


