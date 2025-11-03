# Deep Learning - Project 13 (Exploring Tokenization Strategies for Large Language Models)

Tokenization is a fundamental step in training large language models. Current state-of-the-art models rely on very large vocabularies (for example, tens or even hundreds of thousands of tokens), where embeddings for the vocabulary can account for up to **30% of the total parameters**. This raises questions about efficiency, scalability, and multilingual coverage.

An alternative approach is to use **byte-level tokenization**, which avoids large vocabularies and natively supports Unicode text and therefore also any languages. Such approaches have been successfully explored in recent models, but their trade-offs in efficiency and performance compared to other methods like **Byte Pair Encoding (BPE)** remain an open question.

This project will investigate different tokenization schemes for language models, including BPE and byte-level encodings, hierarchical encodings and others. A central challenge is how to fairly evaluate models trained with different tokenization methods, since standard metrics such as **perplexity** depend on the chosen vocabulary. The project will therefore also explore alternative evaluation methods.

---

**Project Goals**

* Gain familiarity with common tokenization schemes (BPE, byte-level encoding, etc.).
* Train small-scale language models (e.g. Transformer or LSTM) with different tokenization strategies on openly available datasets.
* Investigate evaluation metrics that allow fair comparison across different vocabularies (for example, bits per character, compression-based measures).
* Analyze the **trade-offs in model size, training speed, and performance** for each tokenization scheme.
* (Optional extension) Explore generalization to non-text data, such as byte-level modeling of structured files.

*This project will be supervised by TA Anders Vestergaard Nørskov (aveno@dtu.dk).*
