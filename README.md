# MedChat: Fine-tuning Gemma2 using PEFT QLoRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fine-tune Google's Gemma 2B model using QLoRA to develop medical question-answering capabilities based on structured meaning representations.

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
python app.py
```

This will:

* Load the dataset.
* Initialize the Gemma2 model and tokenizer.
* Apply LoRA adapters.
* Fine-tune the model using the provided dataset.
* Launch the Gradio interface for interaction.
* Save the fine-tuned model and adapters in the `./gemma2_finetuned` directory.


## Learn more about QLoRA:
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Practical QLoRA Guide](https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07)
- [LoRA/QLoRA Explained](https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora)
