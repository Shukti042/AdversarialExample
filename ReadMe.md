# Introduction
This repository contains the code for the paper - **From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation**
# Reproducing Adversarial Attacks on LLMs

This folder contains the implementation of the two adversarial attack frameworks—**Static Deceptor (StaDec)** and **Dynamic Deceptor (DyDec)** on GPT-4o and Llama-3-70B, evaluated across five sensitive text classification datasets.

---

## Required Libraries

Make sure the following Python packages are installed:

* `transformers`
* `torch`
* `accelerate`
* `safetensors`
* `openai`
* `bitsandbytes`

You can install them using:

```bash
pip install transformers torch accelerate safetensors openai bitsandbytes
```

---

## Working Directory

Ensure your working directory is set to the root `AdversarialExample` folder.

---

## OpenAI API Key

Update your valid OpenAI API key in the file `openai_api_key`. This file should contain only the key string.

---

## Dataset Selection

Specify the dataset by editing the `dataset` file with the dataset identifier. Supported datasets include:

| Dataset Name   | Identifier |
| -------------- | ---------- |
| SMS Spam       | `spam`     |
| Hate Speech    | `hate`     |
| Toxic Comment  | `toxic`    |
| LIAR           | `liar`     |
| Spam Detection | `spam3`    |

---

## Running the Attacks

### StaDec (Static Deceptor)

#### GPT-4o

```bash
python3 stadec/GPT/iterative_attack.py
```

#### LLama-3-70B

```bash
python3 stadec/Llama/iterative_attack.py
```

---

### DyDec (Dynamic Deceptor)

#### GPT-4o

```bash
python3 dydec/GPT/iterative_attack.py
```

#### LLama-3-70B

```bash
python3 dydec/Llama/iterative_attack.py
```

---

