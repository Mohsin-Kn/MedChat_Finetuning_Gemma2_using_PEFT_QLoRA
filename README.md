# MedChat: Fine-tuning Gemma2 using PEFT QLoRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fine-tune Google's Gemma 2B model using QLoRA to develop medical question-answering capabilities based on structured meaning representations.

This project provides a modular structure for fine-tuning the Gemma2 model using LoRA adapters on a CPU. The implementation is divided into multiple modules to ensure reusability, clarity, and scalability.

## Project Structure

```
├── data/
│   └── dataset.py
├── model/
│   ├── setup.py
│   └── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Dependencies

* Python 3.10+
* Transformers
* Datasets
* PEFT (Parameter-Efficient Fine-Tuning)

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/<username>/MedChat_Finetuning_Gemma2_using_PEFT_QLoRA.git
cd MedChat_Finetuning_Gemma2_using_PEFT_QLoRA
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify the structure and adjust paths if necessary.

## Dataset

This project uses the `ngram/medchat-qa` dataset from Hugging Face. The dataset is focused on medical question-answering and is specifically tailored for the MedChat fine-tuning process.

## Running the Fine-tuning

To initiate the fine-tuning process, run the following command:

```bash
python main.py
```

This will:

* Load the dataset.
* Initialize the Gemma2 model and tokenizer.
* Apply LoRA adapters.
* Fine-tune the model using the provided dataset.
* Save the fine-tuned model and adapters in the `./gemma2_finetuned` directory.

## Learning Resources

Learn more about QLoRA:

QLoRA Paper

Practical QLoRA Guide

LoRA/QLoRA Explained

## License

MIT License. See `LICENSE` for more information.
