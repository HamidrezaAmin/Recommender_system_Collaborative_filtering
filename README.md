# Book Recommendation System — Collaborative Filtering (Keras)

A lightweight **collaborative filtering** recommender built with **Keras** that learns
low-dimensional embeddings for **users** and **books** and predicts ratings via a **dot-product**
of the embeddings.

- **Model type:** Matrix factorization with learned embeddings  
- **Framework:** TensorFlow / Keras  
- **Dataset:** [Goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) (ratings)

> This repo accompanies a worked example where we load user–book ratings,
> learn 5-dimensional embeddings for users and books, and train with MSE loss.

---

## ✨ What this project does

- Loads ~**981k** ratings with **53,424** users and **10,000** books  
- Splits into train/test  
- Builds a two-tower embedding model:
  - `Embedding(n_users+1, 5)` for users
  - `Embedding(n_books+1, 5)` for books
  - **Dot product** → predicted rating
- Trains with **Adam** and **mean squared error**  
- Saves/loads the trained model (`regression_model.keras`)  
- Evaluates on the test set (reports MSE; RMSE shown below)

---

## 🧠 Model Architecture

```
User ID ─► User Embedding (dim=5) ┐
                                   ├─► Dot Product ─► Predicted Rating
Book ID ─► Book Embedding (dim=5) ┘
```

> Keras summary (example): total params ≈ **317,130** (all trainable).

---

## 📦 Requirements

- Python 3.9+
- tensorflow>=2.12
- pandas, numpy
- scikit-learn
- matplotlib (for the training curve)

Install:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

---

## 📂 Data

This example uses the **Goodbooks-10k** ratings (CSV with `book_id,user_id,rating`).
You can download from Kaggle or use a mirrored CSV as shown in the notebook/script.

- Kaggle dataset: <https://www.kaggle.com/datasets/zygmunt/goodbooks-10k>  
- Example CSV used in the project: `ratings.csv` (three columns)

> Be sure to comply with the dataset license and terms of use.

---

## 🚀 Quick Start

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1) Load data
df = pd.read_csv("ratings.csv")  # columns: book_id,user_id,rating

# 2) Train/val/test split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["rating"])

# 3) Cardinalities
n_users = df.user_id.nunique()
n_books = df.book_id.nunique()

# 4) Build model (matrix factorization w/ embeddings)
user_in = Input(shape=(1,), name="User-Input")
book_in = Input(shape=(1,), name="Book-Input")

user_vec = Flatten(name="Flatten-Users")(Embedding(n_users + 1, 5, name="User-Embedding")(user_in))
book_vec = Flatten(name="Flatten-Books")(Embedding(n_books + 1, 5, name="Book-Embedding")(book_in))

pred = Dot(axes=1, name="Dot-Product")([user_vec, book_vec])

model = Model([user_in, book_in], pred)
model.compile(optimizer=Adam(1e-3), loss="mean_squared_error")

# 5) Train (adjust epochs/batch_size as needed)
history = model.fit([train.user_id, train.book_id], train.rating,
                    validation_split=0.1,
                    epochs=10, batch_size=64, verbose=1)

# 6) Evaluate
mse = model.evaluate([test.user_id, test.book_id], test.rating, verbose=0)
rmse = float(np.sqrt(mse))
print(f"MSE={mse:.3f}  RMSE={rmse:.3f}")

# 7) Save
model.save("regression_model.keras")
```

---

## 📈 Example Results

- **Before training:** MSE ~ **15.8** (essentially untrained)
- **After training (10 epochs, bs=64):** MSE ~ **0.94** ⇒ **RMSE ≈ 0.97**

> Numbers above are indicative from a single run and will vary with random seeds,
> splits, and environment.

---

## 🔍 Predicting

```python
# Predict ratings for the first 10 (user, book) pairs in test
pred = model.predict([test.user_id.head(10), test.book_id.head(10)], verbose=0)
for p, r in zip(pred.flatten(), test.rating.head(10)):
    print(f"pred={p:.2f}  actual={r}")
```

---

## 🧪 Notes & Tips

- **Embedding dim**: start with 5–32; larger dims can help but risk overfitting.  
- **Regularization**: add `l2` on embeddings or **Dropout** on dense heads if you extend the model.  
- **Cold start**: this pure CF model needs historical interactions; to handle new users/books, add metadata (content-based features) or defaults.  
- **Better loss**: if ratings are bounded (e.g., 1–5), you can use a scaled output (e.g., `Dense(1, activation="linear")` and clip) or a calibration layer.

---

## 🗺️ Possible Extensions

- Add **bias terms** for users/books (classic matrix factorization enhancement)
- Increase embedding size & add **MLP head** (dot product + dense fusion)
- Switch to **implicit feedback** (e.g., BPR / sampled softmax)
- Add **book metadata** (authors/genres) and **user features** for hybrid recommendations
- Early stopping + model checkpointing for more stable results

---

## 📘 References

- Goodbooks-10k dataset: <https://www.kaggle.com/datasets/zygmunt/goodbooks-10k>  
- Koren et al., “Matrix Factorization Techniques for Recommender Systems”  
- Keras docs: Embedding, Dot, Model

