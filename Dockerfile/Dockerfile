# Use PyTorch base image (with CUDA support)
FROM python

# Set working directory inside container
WORKDIR /app

# Copy everything from current host directory into /app
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure common packages are available
RUN pip install matplotlib pyyaml tqdm pandas

# Default command (can override in CLI)
CMD ["python", "src/inference/train.py", "--config=configs/train_food_model.yml"]
