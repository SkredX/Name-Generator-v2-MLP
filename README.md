# Neural Name Generator – Trigram + MLP

This repository contains two versions of a character-level **name generator** built using a **trigram context** and a **Multi-Layer Perceptron (MLP)**:

- **Simple version (v2)** – clean baseline implementation  
- **Modified version (v2.1)** – improved / extended implementation

Both versions work the same way at a high level:

1. Load a list of names from a text file (names.txt)  
2. Build a trigram-based dataset  
3. Train an MLP to predict the next character  
4. Sample new, “fake” names from the trained model  

All you need to do is:

- Download the code + dataset for the version you want  
- Open it in your preferred environment (VS Code, Jupyter, Kaggle, Colab, etc.)  
- **Update the `DATA` path** in the code to point to your local `names.txt` file  
- Run the notebook/script  

---

## Repository Structure

```text
.
├── Simple-version-v2/
│   ├── name-generator-using-mlp.ipynb   # Jupyter Notebook version
│   ├── name-generator-using-mlp.py      # Python script version
│   └── names.txt                        # Dataset: one name per line
│
└── Modified-version-v2.1/
    ├── name-generator-v2.1.ipynb          # Jupyter Notebook version (enhanced)
    ├── name-generator-v2.1.py             # Python script version (enhanced)
    └── names.txt                        # Dataset: one name per line
