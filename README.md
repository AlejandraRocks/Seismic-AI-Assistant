# Seismic AI Assistant: Automating Seismic Data Processing and Interpretation

Seismic AI Chatbot is a specialized assistant designed to automate seismic data processing and interpretation. It integrates **Hugging Face Transformers**, **LoRa fine-tuning**, **LangChain**, **Gradio**, **React**, and **RAG (Retrieval-Augmented Generation)** for an interactive and intelligent interface. This project supports both **inference using the Hugging Face API** and **local fine-tuning with LoRa**.

## Features
- Uses **Mistral-7B-Instruct** model for natural language understanding.
- Provides **Seismic Unix command suggestions** without execution.
- Supports **local fine-tuning with LoRa**.
- Allows **seismic document search** using **ChromaDB and RAG (Retrieval-Augmented Generation)**.
- Interactive UI powered by **Gradio and React**.

---
## Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/SeismicAI-Chatbot.git
cd SeismicAI-Chatbot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Hugging Face API Key
If using the Hugging Face API for inference, set your API key:
```bash
export HF_API_KEY="your_huggingface_api_key"
```
For Windows:
```cmd
setx HF_API_KEY "your_huggingface_api_key"
```

### 4️⃣ Run the Chatbot
To start the chatbot using the Hugging Face API:
```bash
python chatbot.py
```

---
## Running Fine-Tuning Locally
If you want to fine-tune the model locally, ensure you have a **GPU-enabled environment**.

```bash
python fine_tune.py
```

Fine-tuned models will be saved in the `modelo-entrenado/` directory.

---
## Project Structure
```
/SeismicAI-Chatbot
│── dataset/                  # Folder containing documents for indexing
│── screenshots/              # Folder for UI screenshots
│── requirements.txt          # Dependencies
│── .gitignore                # Ignored files
│── README.md                 # Documentation
│── chatbot.py                # Main chatbot script
│── fine_tune.py              # Fine-tuning script
│── react_frontend/           # React-based frontend for better UI
│── config.env                # Environment variables (ignored in Git)
```

---
## Example UI Screenshot
![Chatbot Screenshot](screenshots/chatbot_ui.png)

---
## Customizing Gradio and React UI
The chatbot UI is built with **Gradio** for quick deployment and **React** for an enhanced user experience. You can modify the chatbot's appearance in `chatbot.py` or customize the React frontend in `react_frontend/`.

### Customizing Gradio UI in Python
```python
custom_css = """
body { background-color: #f8f9fa; }
.gradio-container { font-family: Arial, sans-serif; }
input, textarea { font-size: 16px; }
button { background-color: #007bff !important; color: white !important; }
"""

interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Enter your message"),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Seismic AI Assistant",
    description="Ask questions about seismic processing.",
    theme="default",
    css=custom_css  # Apply custom styles
)
```

### Running the React Frontend
To run the **React UI**, navigate to the `react_frontend/` folder and install dependencies:
```bash
cd react_frontend
npm install
npm start
```
This will launch the frontend in the browser, where users can interact with the chatbot more efficiently.

---
## How RAG is Used
The chatbot enhances responses using **Retrieval-Augmented Generation (RAG)**. It searches **indexed seismic documents in ChromaDB** and combines relevant information with LLM responses for more accurate answers.

---
## Contributing
Feel free to **fork**, **improve**, and **submit pull requests** for enhancements!

---
## License
This project is licensed under the MIT License.

---
### 🌎 Developed for Geoscientists & Seismic Engineers

