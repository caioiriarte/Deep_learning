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

---

### Running Scripts (RNNLM Model and Transformer predefined models)

The project uses two main Python files to train and evaluate language models using different tokenization and architecture configurations. Those files are `token_rec.py` and `token_trans.py`. Both scripts require four command-line arguments to specify the experiment configuration: the architecture/model type, the tokenizer, the dataset, and the maximum number of samples to use.

<br>

### `token_rec.py` (RNN Language Model)

The file `token_rec.py` (referring to the RNN language model) is designed to run the simple RNN (LSTM-based) architecture on a chosen dataset and tokenizer. To run the RNN-based language model script (`token_rec.py`), you must provide three positional arguments to define the experiment setup: `TOKEN` (Tokenizer ID), `DATASET_OPTION` (Dataset ID), and `max_dataset_size` (number of samples).

**Execution in Terminal:**

```bash
python token_rec.py <TOKEN> <DATASET_OPTION> <max_dataset_size>
```

**Example:**

```bash
python token_rec.py 1 2 10000
```

<br>

### `token_rec.py` (RNN Language Model)

The file `token_trans.py` (referring to the Transformer language model) is designed to instantiate, train, and evaluate pre-defined Transformer architectures (like BERT or GPT-2 configuration) using a corresponding tokenizer and dataset detailed. To run the RNN-based language model script (`token_rec.py`), you must provide three positional arguments to define the experiment setup: `MODEL` (transformer prebuilt models), `TOKEN` (Tokenizer ID), `DATASET_OPTION` (Dataset ID), and `max_dataset_size` (number of samples).

**Execution in Terminal:**

```bash
python token_trans.py <MODEL> <TOKEN> <DATASET_OPTION> <max_dataset_size>
```

<br>

### Models, Tokenizers & Datasets employed

**Models (Random Weight Initialization) (from 1 to 3)**

* Base (GloVe-compatible)
* BERT-config
* GPT-2-config

**Tokenizers (from 1 to 4)**

* GloVe Tokenizer
* BERT WordPiece Tokenizer
* GPT-2 BPE Tokenizer
* Byte-level encoding

**Datasets (from 1 to 3)**

* AG News dataset
* Tiny Shakespeare dataset
* FineWeb2 dataset

---

**Group 86 Integrants**

* Caio Lorenzo Iriarte Salles (s250821)
* Andreas Riposati (s153426)
* Carlos Fernández de Heredia Liger (s243308)