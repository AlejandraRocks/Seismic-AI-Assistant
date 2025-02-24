import gradio as gr
from huggingface_hub import InferenceClient
from langchain.tools import Tool
import subprocess
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

import os
import json

# Get the script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Relative path to dataset.json
dataset_path = os.path.join(current_dir, "dataset.json")

# Load dataset using relative path
dataset = load_dataset("json", data_files=dataset_path)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
api_token = "YOUR-APIKEY"

model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_token)

# 🔹 Configure LoRa
config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)

# 🔹 Configure training
training_args = TrainingArguments(
    output_dir="trained-model",
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])

# 🔹 Train the model with LoRa
def fine_tune():
    trainer.train()
    model.save_pretrained("trained-model")
    tokenizer.save_pretrained("trained-model")
    return "Fine-tuning completed and model saved in 'trained-model'."

# 🔹 Load fine-tuned model in Hugging Face API
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3")

# 🔹 Function to execute Seismic Unix commands
def execute_su(command: str):
    """Executes a Seismic Unix command and returns the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr

# 🔹 Define tools in LangChain
tools = [
    Tool(name="SeismicUnix", func=execute_su, description="Executes commands in Seismic Unix.")
]

# 🔹 Function to interact with the chatbot
def chat(prompt):
    response = client.chat_completion(messages=[{"role": "user", "content": prompt}])
    return response["choices"][0]["message"]["content"]

# Create Gradio interface
interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Enter your message"),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Seismic AI Assistant: Automating Seismic Data Processing and Interpretation",
    description="Ask questions about seismic processing."
)

# 🔹 Add fine-tuning option
fine_tune_interface = gr.Interface(
    fn=fine_tune,
    inputs=[],
    outputs=gr.Textbox(label="Training Status"),
    title="Fine-Tuning with LoRa",
    description="Run fine-tuning on the model with custom data."
)

# 🔹 Launch the application
interface.launch(share=True)
fine_tune_interface.launch(share=True)
