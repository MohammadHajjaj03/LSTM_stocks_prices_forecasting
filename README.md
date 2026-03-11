# 📈 Stock Price Prediction using LSTM

This project focuses on predicting stock prices using **Long Short-Term Memory (LSTM)** networks.
The model learns patterns from historical stock prices and attempts to forecast future price movements.

The project was originally developed as part of a **Machine Learning and Neural Networks course**, where six projects were implemented:

* **3 projects focused on time-series prediction**
* **3 projects focused on classification**

After completing all projects, I decided to further explore the **LSTM-based stock price prediction model**.

---

# 🧠 Project Idea

Stock prices form **time-series data**, where the value of a stock today is influenced by its previous values.

To capture these temporal dependencies, this project uses **LSTM**, a type of recurrent neural network designed to remember long-term patterns in sequential data.

The model predicts stock prices based on the **previous 10 days of historical data**.

---

# ⚙️ Project Details

### Model

* **Model Type:** Long Short-Term Memory (LSTM)
* **Input:** Stock prices from the previous **10 days**
* **Output:** Predicted stock price

### Performance

* Prediction accuracy ranged between **92% and 98%**
* Average accuracy across **6 different datasets: 94%**

### Data

The model was trained using **historical stock price data**, including:

* Daily **closing prices**
* Data collected from several sources
* Stocks from major companies over **multiple months**

---

# 📊 Example Results

The images included in this repository show prediction results using **Apple Inc. stock price data**.

The plots demonstrate:

* Predicted vs actual stock prices
* Model performance during training
* The model's ability to capture stock price trends

---

# ⚠️ Challenges Faced

### 1️⃣ Data Availability

Finding **large and clean datasets** for stock prediction was challenging.

### 2️⃣ Data Leakage

One of the biggest issues encountered was **data leakage**.

Initially, normalization was applied to the **entire target column**, including validation and test data.

The correct approach was to apply **normalization only to the training data**, then apply the same scaler to the validation and test sets.

This issue took significant time to identify and fix.

### 3️⃣ Data Complexity

Some datasets contained complex and irregular patterns, which made it difficult for the model to achieve highly accurate predictions.

---

# ❓ Why LSTM?

LSTM networks are particularly effective for **time-series problems** because they can retain information over long sequences.

This ability allows the model to capture **long-term dependencies in stock price movements**, making LSTM a suitable choice for financial prediction tasks.

---

# 💡 Inspiration for Future Research

Working on this project inspired an idea for a potential research direction:

Using **time-series and spatial data** to predict **military equipment failures** before they occur.

The same principles used in financial prediction could potentially be applied to **predictive maintenance systems**.

---

# 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn

---

# 📌 Note

This project was developed for **educational and research purposes** as part of a university course on **Machine Learning and Neural Networks**.

---

# 👨‍💻 Author

Mohammad Hajjaj
University of Jordan
