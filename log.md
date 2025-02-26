# ME5406 Deep Learning for Robotics

## Project 1: The Frozen Lake Problem and Variations

### Student Information
- **Name:** [Your Name]
- **Student Number:** [Your Student Number]
- **Email:** [Your Email Address]
- **Course:** ME5406 Deep Learning for Robotics
- **Project Title:** The Frozen Lake Problem and Variations

---

# README

## Project Description
This project aims to implement and evaluate three model-free reinforcement learning techniques—First-visit Monte Carlo, SARSA, and Q-learning—to solve the classic Frozen Lake problem. These methods are applied in both a small grid (4x4) and an extended grid (at least 10x10) to assess performance in terms of convergence speed, stability, and cumulative rewards.

---

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Results](#results)
- [Parameter Tuning](#parameter-tuning)
- [FAQ](#faq)
- [Contact](#contact)

---

## Installation
### 1. Create a Virtual Environment (Optional)
It is recommended to create a virtual environment to avoid conflicts with system packages:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

### 2. Install Required Packages
Ensure Python 3.6+ is installed along with the necessary scientific computing and reinforcement learning libraries.
```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes, but is not limited to:
- numpy
- matplotlib
- gym (OpenAI Gym)
- tqdm (for progress visualization)

---

## How to Run

### Method 1: Run in Jupyter Notebook
1. Ensure Jupyter Notebook is installed:
```bash
pip install notebook
```
2. Launch Jupyter Notebook:
```bash
jupyter notebook project1new.ipynb
```
3. Execute each cell in the notebook to observe the implementation and results.

### Method 2: Run via Command Line
1. Execute the main program:
```bash
python main.py
```
2. Make sure all auxiliary files (`frozen_lake_env.py`, `monte_carlo.py`, `sarsa.py`, `q_learning.py`, etc.) are in the same directory.

### Execution Environment
- **Operating Systems:** Windows, Linux, MacOS
- **Python Version:** Python 3.6 or higher

---

## Project Structure
```
Project_Folder/
│
├─ main.py                   # Main entry point
├─ project1new.ipynb         # Jupyter Notebook
├─ frozen_lake_env.py        # Environment setup (if not using OpenAI Gym)
├─ monte_carlo.py            # Monte Carlo algorithm implementation
├─ sarsa.py                  # SARSA algorithm implementation
├─ q_learning.py             # Q-Learning algorithm implementation
├─ utils.py                  # Helper functions and utilities
├─ requirements.txt          # Required packages list
└─ README.md                 # Project documentation
```

---

## Results
1. **Convergence Speed:** Displayed through cumulative reward vs. training episode graphs.
2. **Algorithm Performance Comparison:** Evaluates performance in both small (4x4) and large (10x10) environments.
3. **Example Plot Code:**
```python
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Learning Curve')
plt.show()
```

---

## Parameter Tuning
- `learning_rate`: Learning rate (e.g., 0.1, 0.01, 0.001)
- `discount_factor`: Discount factor (e.g., 0.9, 0.95, 0.99)
- `epsilon`: Exploration probability in ε-greedy policy (e.g., 0.1, 0.2)
- Parameters can be adjusted in `main.py` or `project1new.ipynb` to analyze their impact on the results.

---

## FAQ
1. **Environment Initialization Failed?**
   - Make sure the OpenAI Gym library is correctly installed using `pip install gym`.

2. **Jupyter Notebook Not Displaying Images?**
   - Ensure `%matplotlib inline` is run at the beginning of the notebook.

3. **Module Import Errors?**
   - Verify the project structure and ensure all `.py` files are in the same directory.

---

## Contact
For any questions, please contact me:
- **Email:** [Your Email Address]

Thank you for your attention and support!

