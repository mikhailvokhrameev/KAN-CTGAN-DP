# KAN-CTGAN-DP: Synthetic Tabular Data Generation using KAN Blocks and Differential Privacy

This repository contains a research project on developing a modified generative model based on the **CTGAN** (Conditional Tabular Generative Adversarial Network) architecture. In this work, the standard linear residual blocks of the generator were replaced with **Kolmogorov-Arnold Network (KAN) layers** using B-spline parametrization, and **Differential Privacy** (DP-SGD) mechanisms were integrated into the discriminator.

The model is designed to create synthetic banking data that is statistically identical to real data while guaranteeing the protection of personal information.

<div align="center">

<img src="https://github.com/user-attachments/assets/50b0a00b-8bb7-4d0d-afc2-ca0171325e26" width="300"><br>
  
<em>Example of the Kolmogorov-Arnold Network</em>
</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/25e45c6e-4369-4d3b-917b-506ae4f1ad3e" width="700"><br>
  
<em>Architectural scheme of the original CTGAN-DP</em>
</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/d8ee6416-8a84-43c0-a8fb-93a15ecf690f" width="700"><br>
  
<em>Architectural scheme of the proposed KAN-CTGAN-DP</em>
</div>


---

### Why I did this project?

The main goal was to overcome critical barriers in the financial sector:
*   **Legal Restrictions:** Privacy laws (like Federal Law No. 152-FZ) limit the sharing of real borrower data.
*   **Class Imbalance:** Credit datasets typically have a low share of default borrowers (2–10%), requiring tools for correct data augmentation that preserve complex non-linear dependencies.
*   **Technological Interest:** Investigating how **KAN blocks** can improve the generator's approximation capability compared to classical MLP layers while maintaining formal privacy guarantees.

---

### Technologies used:

*   **Python 3.10+**
*   **PyTorch** — core framework for building neural networks.
*   **Opacus** — library for implementing Differential Privacy via DP-SGD.
*   **KAN (Kolmogorov-Arnold Networks)** — innovative architecture featuring `KANLinear` layers.
*   **CTGAN** — base model for handling tabular data with mixed feature types.
*   **NumPy / Pandas** — for data processing and statistical analysis.

---

### Project Structure

*   `dataset/` — contains data for training and analysis (credit scoring).
*   `main/` — core scripts for running training and evaluating models.
*   `weights/` — saved weights of trained models.
*   `presentation/` — project presentation file.

---

### Setup and Installation

Follow these steps to set up the project environment:
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mikhailvokhrameev/KAN-CTGAN-DP.git
    ```
2.  **Create and Activate a Virtual Environment.**
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

---

### Usage Workflow

#### Step 1: Prepare the Data
The project utilizes credit scoring datasets (e.g., Beeline Challenge). Data undergoes **VGM normalization** (Variational Gaussian Mixture) to handle continuous distributions.

#### Step 2: Train the Model (KAN-CTGAN-DP)
Train the hybrid architecture. The process uses the **AdamW** optimizer and **Binary Cross-Entropy** loss.
*   The Generator employs `KAN_Residual` blocks with **SiLU** activation and **LayerNorm**.
*   The Discriminator is trained using **DP-SGD**, where gradients are clipped and noised to satisfy a privacy budget (e.g., $\epsilon = 5$).

#### Step 3: Evaluate Performance
Three groups of metrics are used to assess the synthetic data:
*   **Utility:** Measures effectiveness on a classification task (ROC-AUC). Testing showed a **2.43%** increase in Utility when using KAN.
*   **Fidelity:** Evaluates statistical matching of distributions and correlations (KL-Divergence, KS-Test).
*   **Privacy:** Checks resistance to Membership Inference Attacks (MIA). The model maintains a maximum protection level (1.0000).

---

#### External libraries and repositories used

*   **pykan** — implementation of Kolmogorov-Arnold Networks.
*   **CTGAN** — base implementation of conditional GANs for tables by SDV.
*   **Opacus** — Differential Privacy tools by PyTorch.
