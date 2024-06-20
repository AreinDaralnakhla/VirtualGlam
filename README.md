# VirtualGlam

VirtualGlam is an innovative product that leverages advanced deep learning models to provide a virtual try-on experience for fashion items. With VirtualGlam, you can see how any garment will look on you with the ability to mix and match, all from the comfort of your home. This product is in constant development, keep an eye on new features in the future. 

## Getting Started

Follow these steps to set up and run VirtualGlam locally on your machine:

### Prerequisites

- Ensure you have `git` installed on your system.
- Install `conda` for managing the virtual environment.

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/AreinDaralnakhla/VirtualGlam.git
    ```

2. **Navigate to the project directory:**
    ```sh
    cd VirtualGlam
    ```
3. **Download the complete [ckpt folder](https://drive.google.com/drive/folders/1GSpHnqK07lc6Sdta5IRn4xsmeOvQ1oIQ?q=sharedwith:public%20parent:1GSpHnqK07lc6Sdta5IRn4xsmeOvQ1oIQ) manually and replace it with the existing one in the VirtualGlam directory**

4. **Create the virtual environment:**
    ```sh
    conda env create -f environment.yaml
    ```

5. **Activate the virtual environment:**
    ```sh
    conda activate idm
    ```

### Running the Application

6. **Run the inference script:**
    ```sh
    sh inference.sh
    ```

7. **Start the server demo:**
    ```sh
    python product_demo/flask.py
    ```

8. **Open `virtualGlam.html` in your preferred browser** to access the application interface.

### Troubleshooting

- If you need guidance while using the product, open the browser developer console to see step-by-step what is happening in the background.
- The server-side console also provides detailed logs that can help in troubleshooting any issues.
