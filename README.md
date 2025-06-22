```markdown
# IMDb Sentiment Analysis API 🎬🔍

This project is a sentiment analysis API powered by a fine-tuned **DistilBERT** model using the IMDb movie review dataset. The model is served via a **FastAPI** backend and containerized using **Docker** for easy deployment.

---

## 🚀 Features

- ✅ Binary sentiment classification (Positive / Negative)
- ✅ FastAPI-based REST API
- ✅ Hugging Face Transformers integration
- ✅ Dockerized for local or production deployment
- ✅ Swagger UI for easy testing

---

## 📁 Project Structure

```

sentiment-analysis-bert/
├── app/
│   ├── main.py                # FastAPI app
│   ├── Dockerfile             # Docker config
├── training/
│   └── train.py               # Model fine-tuning script
├── model/                     # Saved fine-tuned model (locally mounted, not committed)
├── .gitignore
├── requirements.txt
└── README.md

````

---

## 📦 Installation (Local)

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/imdb-sentiment-analysis-api.git
   cd imdb-sentiment-analysis-api
````

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Start API**:

   ```bash
   cd app
   uvicorn main:app --reload
   ```

---

## 🐳 Run with Docker

Make sure your fine-tuned model exists in the `model/` directory first.

```bash
docker build -t sentiment-app ./app
docker run -p 8000:8000 -v $(pwd)/model:/model sentiment-app
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Usage (cURL Example)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The movie was absolutely amazing!"}'
```

**Response:**

```json
{
  "sentiment": "Positive",
  "confidence": 0.975
}
```

---

## 🧠 Model Info

* **Base model**: `distilbert-base-uncased`
* **Fine-tuned on**: IMDb dataset
* **Task**: Binary text classification

---

## 🔒 License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and use it for personal or commercial projects.

---

## 🙋‍♀️ Author

**Dedeepya Chittireddy**
Built with ❤️ using Python, Transformers, and Docker
GitHub: [@DedeepyaChittireddyy](https://github.com/DedeepyaChittireddyy)

---

## ⭐️ Show Your Support

If you found this project useful, please give it a ⭐️ on GitHub!
