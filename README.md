# Max-Plus Algebra & Timed Event Graph (TEG) Simulator

A Python-based toolkit for computing **Max-Plus algebra** operations (tropical mathematics) and simulating **Timed Event Graphs (TEGs)** for discrete-event dynamic systems, manufacturing lines, and network control models.

## Features

* **Max-Plus Matrix Operations**: Supports Max-Plus matrix multiplication ($\otimes$), addition ($\oplus$), and the Kleene star operation (($A_0$)*) for convergence and system representation.
* **TEG State-Space Simulator**: Simulates event-driven state and output trajectories based on the max-plus linear state equations:

$$\begin{aligned}   x(k) &= (A \otimes x(k-1)) \oplus (B \otimes u(k)) \\   y(k) &= C \otimes x(k)   \end{aligned}$$


* **Docker Support**: Containerized environment for quick, dependency-free execution.
* **Visualization**: Generates console tables and performance plots (`simulation_plot.png`) tracking event firing trajectories and bottleneck dynamics.

## Project Structure

```text
.
├── Dockerfile          # Container configuration for Python 3.11 slim
├── requirements.txt    # Project Python dependencies (NumPy, Matplotlib)
├── check.py            # Canonical system matrix analyzer & Kleene star solver
└── main.py             # Timed Event Graph (TEG) state-space simulator & plotter

```

## Requirements

* Python 3.11+
* NumPy (`>=1.21.0`)
* Matplotlib (`>=3.4.0`)

## Installation & Usage

### Option 1: Local Execution

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```


3. Run the canonical matrix analyzer (`check.py`):
```bash
python check.py
```


4. Run the main TEG simulation and generate plots (`main.py`):
```bash
python main.py
```



### Option 2: Docker Execution

1. Build the Docker image:
```bash
docker build -t teg-simulator .

```


2. Run the container:
```bash
docker run --rm -v $(pwd):/app teg-simulator

```# max-plus-simulation
