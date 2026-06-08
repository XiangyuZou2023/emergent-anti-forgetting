FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Copy everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run ablation experiment
# Override with: docker run ... python test_astar_matters.py
CMD ["python", "run_ablations.py"]
