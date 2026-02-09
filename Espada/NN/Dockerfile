FROM continuumio/miniconda3

# ── install OS deps for OpenCV ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY environment.yml .
RUN conda env create -f environment.yml \
 && conda clean -afy

SHELL ["conda", "run", "-n", "us-denoise", "/bin/bash", "-lc"]

# KAIR + its Python reqs
RUN git clone https://github.com/cszn/KAIR.git /workspace/KAIR \
 && pip install -r /workspace/KAIR/requirement.txt

EXPOSE 8888
CMD ["/bin/bash"]