# Portfolio Presentation

## One-line project description

**Built a leakage-safe breast-tumor classification pipeline with a PyTorch Neural Network, benchmarked against three classical models, and deployed reproducible batch inference through Streamlit.**

## Resume bullet

- Developed an end-to-end breast-tumor classification system using a PyTorch Neural Network on 569 WDBC samples, applying leakage-safe preprocessing, validation-based threshold selection and multi-metric evaluation; achieved **96.49% test accuracy and 0.986 ROC-AUC**, with CLI and Streamlit inference.

## Interview talking points

1. Why is accuracy not enough?  
   Because false negatives and false positives have different consequences; the project therefore reports sensitivity and specificity separately.

2. How did you avoid leakage?  
   The scaler is fit only on the training partition, never on validation/test data.

3. Why use a validation threshold?  
   A 0.50 threshold is arbitrary. The project chooses an operating threshold using validation data and freezes it before test evaluation.

4. Why compare multiple models?  
   To show that the neural network is a deliberate portfolio choice rather than an assumption that deep learning is automatically better.

5. Is this clinically deployable?  
   No. It is an educational portfolio project. Clinical use would require external validation, prospective evaluation, calibration, clinical oversight and regulatory processes.
