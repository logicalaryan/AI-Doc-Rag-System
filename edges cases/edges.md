# RAG Pipeline Edge Cases

This document outlines the potential edge cases and failure points within the RAG (Retrieval-Augmented Generation) Document Q&A application, categorized by the pipeline stages.

## 1. Document Ingestion Edge Cases (Data Extraction)

*   **Tables and Graphs:** Text extractors often struggle with tables, reading them column by column instead of row by row, or ignoring graphs completely. This results in jumbled or missing data.
*   **Unsupported File Formats:** Scanned PDFs (which are essentially images requiring OCR) or encrypted/password-protected PDFs will result in zero text being extracted by standard libraries.
*   **Very Large Documents:** Extremely large documents (e.g., a 500-page book) might take too long to process and generate embeddings, potentially timing out the application before data is saved to ChromaDB.

## 2. Chunking Edge Cases (Data Segmentation)

*   **Mid-Sentence Splits:** A chunk cuts off exactly in the middle of a crucial sentence if split strictly by character count without overlap or semantic awareness.
*   **Loss of Context (The "Pronoun Problem"):** Chunk 1 introduces a subject (e.g., "The CEO"), but Chunk 2 just uses a pronoun (e.g., "He"). If only Chunk 2 is retrieved, the LLM lacks the context of who "He" refers to.
*   **Code Snippets:** Breaking programming code strictly by character limits might split a function in half, destroying its logical structure and making it useless for the LLM.

## 3. Retrieval Edge Cases (Vector Search)

*   **Vocabulary Mismatch (Synonyms):** The user asks a question using different terminology (e.g., "workforce reduction") than what is in the document (e.g., "fired employees"). If the embeddings aren't mathematically close enough, the correct chunks won't be retrieved.
*   **Multi-hop Reasoning:** The user asks a question that requires connecting disparate pieces of information located far apart in the document (e.g., Page 2 and Page 45). Vector search typically retrieves chunks similar to the question, not chunks that logically connect to each other.
*   **"Needle in a Haystack":** The document contains the exact answer, but it's surrounded by a massive amount of irrelevant text within the same chunk. The embedding vector gets diluted, causing ChromaDB to rank it lower than less accurate chunks.

## 4. LLM Generation Edge Cases (Answer Synthesis)

*   **Context Window Overflow:** Retrieving too many chunks (e.g., top 20 matches) and injecting them into the prompt template might exceed the LLM's maximum token limit, causing a crash or prompt truncation.
*   **Conflicting Information:** The retrieved chunks contain contradictory data (e.g., two different dates for the same event from different pages). The LLM might get confused or confidently output the incorrect information.
*   **"I don't know" Bypass:** Even with strict prompt guardrails ("If the answer is not in the context, say 'I don't know'"), LLMs can sometimes hallucinate and rely on their pre-trained data to answer instead of strictly adhering to the provided context.

## 5. Streamlit / UI Edge Cases (User Interaction)

*   **Chat History Bloat:** Appending every message to the context window eventually makes the conversation too long, breaking the application when the token limit is reached.
*   **State Loss:** If the user refreshes the Streamlit page, the entire chat history and uploaded document state might disappear because Streamlit reruns the script top-to-bottom on reload.
