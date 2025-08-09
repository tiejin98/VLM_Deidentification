# VLM_Deidentification

## Requirements

This project is based on **Kosmos-2.5**.  
To install the base Kosmos-2.5 implementation, run:

```bash
pip install git+https://github.com/tic-top/transformers.git
```

We are using:

```
transformers==4.44.dev0
```

---

## Training

1. Download the training dataset from [Training Set](https://drive.google.com/file/d/1DQ23FMsgyXZGHraZwuIJUYInl_lmkrTV/view?usp=sharing)
2. Unzip the dataset into the same folder as the project code.
3. Run the training script:
   ```bash
   python training.py
   ```

---

## Evaluation

1. Download the sample evaluation dataset from [Evaluation Set](https://drive.google.com/file/d/12Z1V2pyFPpKoYXn77gvKLlhXAbTx-NlB/view?usp=sharing)
2. Run the evaluation script:
   ```bash
   python evaluation.py
   ```

---

## Generating a Custom Training Dataset

To create your own training dataset using the provided pipeline, consider use:

```bash
python generating.py
```