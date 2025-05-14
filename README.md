
# BERT文本分类项目

This pilot study implements a BERT-based model for text classification tasks, leveraging [HuggingFace Transformers](https://github.com/huggingface/transformers), PyTorch, and EDA (Easy Data Augmentation).

## Features

- Fine-tuned BERT for binary text classification
- Integration of data augmentation using EDA
---

## 📁 Project Structure

```
.
├── bert_code.py            
├── ai-ga-dataset.csv       # Dataset(you need to download form [AI-GA Dataset](https://github.com/panagiotisanagnostou/AI-GA) )
├── models/                 # Model saving path
├── plots/                  # Visualization result saving directory
└── README.md               
```

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers==4.46.3 (by HuggingFace)
- scikit-learn
- pandas

---


## References

- **Dataset**: [AI-GA Dataset](https://github.com/panagiotisanagnostou/AI-GA)  
- **BERT Model Implementation**: [BERT Sentiment Classification_IMDB](https://github.com/mdabashar/Deep-Learning-Algorithms/blob/master/BERT_Semntiment_Classification_IMDB.ipynb)  
- **EDA (Easy Data Augmentation)**: [eda_nlp](https://github.com/jasonwei20/eda_nlp)

