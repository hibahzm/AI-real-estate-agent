# ============================================================
# Dockerfile — AI Real Estate Agent Backend
#
# What each instruction does (you must understand every line):
#
# FROM        — base image (Python 3.11 on lightweight Debian)
# WORKDIR     — sets the working directory inside the container
# COPY        — copies files from your machine into the container
# RUN         — runs a shell command during the build step
# EXPOSE      — documents which port the app uses (doesn't actually open it)
# ENV         — sets environment variables inside the container
# CMD         — the command that runs when the container starts
# ============================================================

# Start from the official Python 3.11 slim image
# "slim" = Debian without most tools — smaller image, faster build
FROM python:3.11-slim

# Set the working directory inside the container
# All subsequent commands run from here
WORKDIR /app

# Install uv — the fast Python package manager we use
# --no-cache-dir keeps the image small (no pip cache left behind)
RUN pip install --no-cache-dir uv

# Copy only the dependency manifest first
# Docker caches layers — by copying pyproject.toml before the app code,
# the expensive "uv sync" step is only re-run when dependencies change,
# not every time you change a line of Python code
COPY pyproject.toml ./

# Install all production dependencies from pyproject.toml
# --no-dev skips dev dependencies (pytest etc.) — not needed in production
RUN uv sync --no-dev

# Copy the rest of the application code
COPY app/ ./app/

# Copy the trained model files
# These must exist in backend/saved_model/ before you docker build
# (Run the Colab notebook first!)
COPY saved_model/ ./saved_model/

# Document which port the app listens on
# This does NOT open the port — you do that with docker run -p 8000:8000
EXPOSE 8000

# Set production environment variables
# These are defaults — override at runtime with -e flags or Railway env vars
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# The command that runs when the container starts
# Uses uv run to launch uvicorn inside the uv-managed environment
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 matches the EXPOSE instruction above
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]