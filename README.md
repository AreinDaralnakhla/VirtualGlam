# VirtualGlam

VirtualGlam is an innovative product that leverages advanced deep learning models to provide virtual try-on experiences for fashion items. With VirtualGlam, you can see how different garments look on a human model in real-time, all from the comfort of your device.

## Getting Started

Follow these steps to set up and run VirtualGlam locally on your machine:

### Prerequisites

- Ensure you have `git` installed on your system.
- Install `conda` for managing the virtual environment.

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/VirtualGlam.git
    ```

2. **Navigate to the project directory:**
    ```sh
    cd VirtualGlam
    ```

3. **Create the virtual environment:**
    ```sh
    conda env create -f environment.yaml
    ```

4. **Activate the virtual environment:**
    ```sh
    conda activate idm
    ```

### Running the Application

5. **Run the inference script:**
    ```sh
    sh inference.sh
    ```

6. **Start the Gradio demo:**
    ```sh
    python gradio_demo/flask.py
    ```

7. **Open `virtualGlam.html` in your preferred browser** to access the application interface.

### Troubleshooting

- If you need guidance while using the product, open the browser developer console to see step-by-step what is happening in the background.
- The server-side console also provides detailed logs that can help in troubleshooting any issues.
