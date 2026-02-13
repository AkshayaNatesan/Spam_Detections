**SMS Spam Detection using LSTM (PyTorch)**

*Overview*

This project implements a deep learning based SMS spam classifier using PyTorch.
The goal was to build a sequence model capable of identifying spam messages while handling dataset class imbalance.
The model uses an embedding layer followed by a bidirectional LSTM and a linear classifier trained with weighted binary cross-entropy.


*Repository Structure*

├── README.md
├── requirements.txt
├── data/
│   └── spam.csv
├── model/
│   ├── spam_detector.py
│   └── data_loader.py
├── utils/
│   ├── preprocessing.py
│   ├── metrics.py
├── train.py
└── evaluate.py
