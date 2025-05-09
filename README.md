# MTech-Project-Sem-VIII

Hello, my name is Divyanshi Chauhan and I am currently in the fourth part of my Integrated Dual Degree program, Computer Science and Engineering.

Our esteemed guide for this project is *Prof. A.K.Singh*. He is providing us with indispensable guidance and expertise throughout the course of this project.

I am both deeply committed to contributing to this project and pushing the boundaries of our knowledge.

Our current focus is to delve into the realm of '**Automatic Translation of Data in relatively High Resource Language to a Low Resource Language, using LLM based Few Shot Training APIs.** '. This is a fascinating area of study which explores the depth of natural language processing, involving deep learning models.

Topic: Automatic Translation of Data in relatively High Resource Language to a Low Resource Language, using LLM based Few Shot Training APIs.

1. Using LLM-based API to automatically translate the data.
2. Using few shot training of LLMs, or other translation models, to improve translation quality.
3. Creating synthetic translated data for a few language pairs with little or no parallel corpus or API support.

## Research Papers
1. Towards Guided Back-translation for Low-resource languages. Example: Kabyle-French
2. Reinforcement of low-resource language translation with neural machine translation and backtranslation synergies. Example: English-Kannada


# Fine-Tuning

During fine-tuning, we focus on sharpening the model's skills for particular tasks or fields, turning it from a general language learner into an expert for specific jobs. For example, imagine we're training the model using lots of medical texts. This helps it become really good at understanding medical terms and answering health-related questions. Similarly, if we train it on legal documents, it becomes excellent at summarizing contracts and discussions related to the law.

**Retrieval-Augmented Generation (RAG)
RAG (Retrieval-Augmented Generation) is a technique that helps enhance the performance of a Large Language Model (LLM) by incorporating specific information without altering the core model. This information can be not only more recent than what the LLM has learned but also tailored to a particular organization or industry. This capability enables the generative AI system to offer responses that are not only contextually fitting but also grounded in the most current and organization-specific data available. In simpler terms, it's like giving the AI a tool to fetch the latest and most relevant information, ensuring more accurate and up-to-date answers to questions.**

Models: GPT 3.5 Turbo, Babbage-002, Davinci-002

### Models

1. **GPT (Generative Pre-trained Transformer):** GPT refers to the family of language models developed by OpenAI, starting with GPT-1 and followed by GPT-2 and GPT-3. GPT models are based on the transformer architecture and are trained on diverse datasets to perform a wide range of natural language processing tasks, including text generation, completion, translation, and more. GPT-3, in particular, is known for its large scale, with 175 billion parameters, making it one of the most powerful language models.
2. **Babbage:** Babbage was the codename for an earlier version of OpenAI's language model. It was a precursor to GPT-3 and was part of OpenAI's efforts to iteratively improve upon language models. As OpenAI progresses with research and development, newer versions are often released with enhanced capabilities.
3. **Davinci:** Davinci is the codename for one of the configurations or variants of the GPT-3 model. OpenAI uses different codenames to represent various versions or configurations of their models. Davinci is known for its large size, having 175 billion parameters, making it one of the most powerful and versatile language models available.


- **IndicTrans2**, a **neural machine translation system** developed by **AI4Bharat**, designed specifically for translating between **English and Indian languages**, as well as between **Indian languages** themselves.
- `transformers`: for using Hugging Face models.
- `nltk`, `sacremoses`: for tokenization and text processing.
- `bitsandbytes`: for memory-efficient quantized model loading.
- `sentencepiece`: for tokenization of Indic scripts.

IndicTrans2 is the first open-source transformer-based multilingual NMT model that supports high-quality translations across all the 22 scheduled Indic languages — including multiple scripts for low-resource languages like Kashmiri, Manipuri and Sindhi. It adopts script unification wherever feasible to leverage transfer learning by lexical sharing between languages. Overall, the model supports five scripts Perso-Arabic (Kashmiri, Sindhi, Urdu), Ol Chiki (Santali), Meitei (Manipuri), Latin (English), and Devanagari (used for all the remaining languages).

Summary:

- **Uses IndicTrans2 for translation** between English and Indic languages, as well as between different Indic languages.
- **Batch processing** enables efficient translation of multiple texts.
- **Handles quantization** to optimize memory usage and performance.
- **Preprocessing and postprocessing steps** enhance translation quality.
- **Beam search** with 5 beams ensures accurate and fluent translations.


- language translation model using Python and the Natural Language Toolkit (NLTK)
- parallel corpus containing English and Spanish sentences, preprocess the data, train an IBM Model 1 translation model, and create a translation function
- IBM Model 1 is the **simplest** statistical model used for **word-level translation** between two languages
- **Use Expectation-Maximization (EM) algorithm**:

          **E-Step**: Estimate how likely each target word aligns to each source word.

          **M-Step**: Update the probabilities based on those estimates.

- `pandas`: Used to read the sentence dataset (CSV).
- `re`: For regular expressions (used in text cleaning).
- `nltk`: Natural Language Toolkit, used here for translation modeling.
- `AlignedSent`, `IBMModel1`: Classes from `nltk.translate` to build a translation model.

# Implementation 4 (MTP 6)

Implementation of few research papers
Research Paper1: Towards guided back-translation for low resource languages, a case study on Kabyle-French

Key-Points:

- Lexicon and Bias Injection:

Lexicon and Bias injection is a technique that aims to improve translation quality by introducing a bias towards less frequent words in the training corpus

- Guided Back-translation:

we opt to analyze the performance based on sentence length. This led us to obtain performances for specific sentence lengths based on the BLEU metric

- key parameters: sentence length and part-of-speech tags
- We observed that the predominant sentences had a length ranging from 6 to 10 words
- Part-of-speech (POS) tags reveal important insights into sentence structure. Our study used POS tagging during assessment and found that verbs and nouns significantly affected translation quality. This finding suggests we can improve results by adjusting translation models to focus more on specific POS tags.
Research Paper 2: Reinforcement of low-resource language translation with neural machine translation and backtranslation synergies

Key-Points:

- Recurrent neural networks (RNNs) with sequence-to-sequence (Seq2Seq) models and the encoder-decoder mechanism with long short-term memory (LSTM) as the RNN unit have been shown to be effective for MT in LRL scenarios
- Convolutional and sequence-to-sequence models trained on conditional distribution translate well but suffer as input phrase length grows
- A multi-source neural model with two independent encoders is designed to translate agglutinative languages with complex morphology and limited resources. These encoders put a language layer in the input embedding layer and consider lemma, POS tag, and morphological tag
- To develop a Doc2Doc NMT model that uses sequence-to-sequence transformation to generate target documents from input documents, utilise the P-transformer

---

---

---

---

---

---

Based on the above two papers, I modeled a neural machine translation model:-

- The Sequence-to-Sequence (seq2seq) model uses an encoder-decoder architecture with LSTM (Long Short-Term Memory) networks. The encoder network converts the input German sequence into a single Context Vector that contains an abstract representation of the input. The decoder network then processes this Context Vector to generate the English translation one word at a time.

# Implementation 5 (MTP 7)

- fine-tune the pre-trained hugging-face translation model **(Marian-MT)**
- neural machine translation
- hugging-face Transformers model for translating English to Romanian language
- AutoModelForSeq2SeqLM class
