import gradio as gr
from src.seismic_processor import process_segy_and_return_path

def handle_segy_upload(file):
    try:
        processed_path = process_segy_and_return_path(file.name)
        return processed_path
    except Exception as e:
        return f"Error al procesar el archivo: {str(e)}"

segy_interface = gr.Interface(
    fn=handle_segy_upload,
    inputs=gr.File(label="Upload SEG-Y Line (.sgy)"),
    outputs=gr.File(label="Download Processed SEG-Y"),
    title="2D Seismic Processor",
    description="Upload a .sgy file and download the .sgy file processed version (gain + filters) using Seismic Unix."
)

segy_interface.launch()
