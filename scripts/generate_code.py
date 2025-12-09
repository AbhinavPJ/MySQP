'''
The purpose of this script is to parse Hock–Schittkowski (HS) problems in APMonitor-style format
and generate corresponding JAX and PyTorch function representations along with metadata.
This creates the benchmark data file used in the SQP benchmarking suite.
'''
import os
import time
import json
import glob
import re
from typing import List
import google.generativeai as genai  #type: ignore
API_KEY = os.environ.get("GEMINI_API_KEY")
INPUT_DIR = "../hs_problems" 
OUTPUT_FILE = "../Benchmarking/hs_benchmark_data.py"
BATCH_SIZE = 1
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
def get():
    existing = set()
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            match = re.match(r'^# --- (hs\d+)', line)
            if match:
                existing.add(match.group(1))
    return existing
def init():
    with open(OUTPUT_FILE, "w") as f:
        f.write("import jax.numpy as jnp\n")
        f.write("import numpy as np\n")
        f.write("import torch\n\n")
        f.write("PROBLEM_REGISTRY = []\n\n")
def read(file_paths: List[str]) -> str:
    data = []
    for fp in file_paths:
        with open(fp, "r") as f:
            content = f.read()
        name = os.path.basename(fp).replace(".apm", "")
        data.append(f"--- PROBLEM {name} ---\n{content}\n")
    return "\n".join(data)
def construct(batch_content: str) -> str:
    return f"""
You are an expert in Hock–Schittkowski (HS) parsing and mathematical optimization.

Your task:
Given several HS problems in APMonitor-style (.apm) format, generate a STRICT JSON
description for each problem, suitable for building:
    - JAX objective + constraint vector
    - Torch objective + list of scalar constraint functions
    - Metadata: bounds, initial guess, inequality indices, reference objective.

=====================================================================
INPUT (APM FORMAT)
=====================================================================
The following are one or more problems. Each problem starts with a header:

    --- PROBLEM hsXXX ---

and then APMonitor-style content.

{batch_content}

=====================================================================
APM PARSING RULES
=====================================================================

1) VARIABLES / BOUNDS / INITIAL GUESS
--------------------------------------
Look for variable declarations like:

    x1 = 100, >= 0.1, <= 100
    x2 = 12.5, >= 0, <= 25.6
    x3 = 3.0, >= 0, <= 5.0

Rules:
- Convert 1-based indices to 0-based positions:
    x1 → x[0], x2 → x[1], etc.
- x0 is the list of initial values in index order.
- For each x_i, extract lower and upper bounds:
    - If LB is missing, use -1e20.
    - If UB is missing, use +1e20.
- Final "bounds" must be:
    [[LB0, UB0], [LB1, UB1], ..., [LB(n-1), UB(n-1)]].

2) OBJECTIVE
------------
There will be a unique "objective" definition, e.g.:

    obj = (x1 - x2)^2 + (x1 + x2 - 10)^2 / 9 + (x3 - 5)^2

Rules to convert APMonitor syntax:
- ^       → **        (power)
- sin     → jnp.sin   (JAX)
- cos     → jnp.cos
- exp     → jnp.exp
- sqrt    → jnp.sqrt
- log     → jnp.log
- For Torch, use torch.sin, torch.cos, etc.

You must produce:
- A scalar JAX objective function: hsXXX_obj_jax(x)
- A scalar Torch objective function: hsXXX_obj_torch(x)

3) CONSTRAINTS
--------------
Every non-objective equation is a constraint.

APMonitor relations:
- expr = rhs     → equality constraint
- expr <= rhs    → inequality: expr - rhs <= 0
- expr >= rhs    → inequality: rhs - expr <= 0

You must:
- List all constraints in a fixed order.
- For JAX: a single function returning a 1D jnp.array of all constraints.
- For Torch: one scalar function per constraint in a list.

Inequality indices:
- Let constraints be c[0], c[1], ..., c[m-1] as returned by JAX.
- For each i:
    - If c[i] is equality (expr = rhs), DO NOT include it in inequality lists.
    - If c[i] is inequality (expr <= rhs or expr >= rhs rewritten),
      then i must be included in "ineq_indices" and "ineq_indices_jax".
- All inequalities must be converted to the canonical form:
      c_i(x) <= 0

4) REFERENCE OBJECTIVE
----------------------
Look for comments like:

    ! best known objective = 0.0

Use that value as "ref_obj". If missing, use null.

5) SAFETY & NUMERICAL STABILITY
-------------------------------
If the APMonitor objective contains something like:

    (u - x2)^x3

or, more generally, raising a potentially negative or sign-uncertain base
to a real-valued exponent:

    base^p   where base might be <= 0 and p is not guaranteed integer,

you MUST rewrite it safely as:

    let base_safe = max(base, 1e-8)
    base^p = exp(p * log(base_safe))

Apply this consistently for both JAX and Torch versions.

6) SHAPE & TYPE REQUIREMENTS
----------------------------
JAX:
- Objective: scalar jnp.ndarray or Python float, returned by hsXXX_obj_jax(x).
- Constraints: 1D jnp.ndarray of shape (m,), returned by hsXXX_constr_jax(x).
- Use dtype matching x (e.g., dtype=x.dtype) when appropriate.

Torch:
- Objective: scalar tensor returned by hsXXX_obj_torch(x).
- Constraints: each hsXXX_constrK_torch(x) returns a scalar tensor.

Unconstrained problems:
- JAX: hsXXX_constr_jax(x) must return jnp.zeros((0,), dtype=x.dtype).
- Torch: "constr_list_torch" must be an empty list [].
- "n_constr" must be 0.
- "ineq_indices" and "ineq_indices_jax" must be [].

7) NAMING CONVENTIONS
---------------------

If the problem name is "hs025":

- JAX objective:       def hs025_obj_jax(x): ...
- JAX constraints:     def hs025_constr_jax(x): ...

- Torch objective:     def hs025_obj_torch(x): ...
- Torch constraints:   def hs025_constr0_torch(x): ...
                        def hs025_constr1_torch(x): ...
                        etc.

The Torch constraint definitions should appear in the same order
as entries in the constraint vector in JAX.

8) PYTHON CODE STYLE
--------------------
- NO imports inside the generated functions.
- Use existing names: jnp for JAX, torch for Torch.
- Objective must be a simple return expression, no printing or side effects.
- Constraints must be direct expressions; do not reshape manually unless needed.

=====================================================================
EXAMPLE (ILLUSTRATIVE)
=====================================================================

Example APM block (simplified):

--- PROBLEM hs065 ---
! best known objective = 0.9535288567

Variables
    x1 = -5, >= -1e20, <= 1e20
    x2 =  5, >= -1e20, <= 1e20
    x3 =  0, >= -1e20, <= 1e20

Equations
    obj = (x1 - x2)^2 + (x1 + x2 - 10)^2 / 9 + (x3 - 5)^2

    x1^2 + x2^2 + x3^2 <= 48
    x1 >= -4.5
    x1 <=  4.5
    x2 >= -4.5
    x2 <=  4.5
    x3 >= -5.0
    x3 <=  5.0

The correct JSON for this example is:

{{
  "name": "hs065",
  "n_vars": 3,
  "ref_obj": 0.9535288567,
  "bounds": [[-1e20, 1e20], [-1e20, 1e20], [-1e20, 1e20]],
  "x0": [-5.0, 5.0, 0.0],

  "obj_func_code_jax":
    "def hs065_obj_jax(x):\\n"
    "    return (x[0] - x[1])**2 + (x[0] + x[1] - 10)**2/9 + (x[2] - 5)**2",

  "constr_func_code_jax":
    "def hs065_constr_jax(x):\\n"
    "    return jnp.array(["
    "x[0]**2 + x[1]**2 + x[2]**2 - 48,"
    " -4.5 - x[0],"
    " x[0] - 4.5,"
    " -4.5 - x[1],"
    " x[1] - 4.5,"
    " -5.0 - x[2],"
    " x[2] - 5.0"
    "])",

  "obj_func_code_torch":
    "def hs065_obj_torch(x):\\n"
    "    return (x[0] - x[1])**2 + (x[0] + x[1] - 10)**2/9 + (x[2] - 5)**2",

  "constr_list_torch": [
    "def hs065_constr0_torch(x): return x[0]**2 + x[1]**2 + x[2]**2 - 48",
    "def hs065_constr1_torch(x): return -4.5 - x[0]",
    "def hs065_constr2_torch(x): return x[0] - 4.5",
    "def hs065_constr3_torch(x): return -4.5 - x[1]",
    "def hs065_constr4_torch(x): return x[1] - 4.5",
    "def hs065_constr5_torch(x): return -5.0 - x[2]",
    "def hs065_constr6_torch(x): return x[2] - 5.0"
  ],

  "ineq_indices": [0, 1, 2, 3, 4, 5, 6],
  "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6],
  "n_constr": 7
}}

Note:
- All 7 constraints are inequalities → all indices 0..6 are in inequality lists.
- Objective and constraints are structurally identical between JAX and Torch.

=====================================================================
OUTPUT FORMAT — STRICT JSON ONLY
=====================================================================

Your output for the entire batch must be a single JSON object:

{{
  "problems": [
    {{
      "name": "hsXXX",
      "n_vars": INTEGER,
      "ref_obj": FLOAT or null,
      "bounds": [[LB0, UB0], [LB1, UB1], ...],
      "x0": [...],

      "obj_func_code_jax":
        "def hsXXX_obj_jax(x):\\n    ...",

      "constr_func_code_jax":
        "def hsXXX_constr_jax(x):\\n    return jnp.array([...])",

      "obj_func_code_torch":
        "def hsXXX_obj_torch(x):\\n    ...",

      "constr_list_torch": [
        "def hsXXX_constr0_torch(x): return ...",
        "def hsXXX_constr1_torch(x): return ...",
        ...
      ],

      "ineq_indices": [...],
      "ineq_indices_jax": [...],
      "n_constr": INTEGER
    }},
    ...
  ]
}}

CRITICAL:
- NO markdown.
- NO comments in the JSON.
- NO extra keys.
- All functions must be syntactically valid Python when inserted in a .py file.
- Use exactly the function names implied by "name".
"""
def append(problems: list) -> None:
    with open(OUTPUT_FILE, "a") as f:
        for p in problems:
            name = p["name"]
            f.write(f"# --- {name} (JAX + Torch) ---\n")
            f.write(p["obj_func_code_jax"] + "\n\n")
            f.write(p["constr_func_code_jax"] + "\n\n")
            f.write(p["obj_func_code_torch"] + "\n\n")
            torch_constr_names = []
            for k, constr_code in enumerate(p["constr_list_torch"]):
                f.write(constr_code + "\n")
                torch_constr_names.append(f"{name}_constr{k}_torch")
            f.write("\n")
            reg = f"""
PROBLEM_REGISTRY.append({{
    "name": "{name}",
    "n_vars": {p["n_vars"]},
    "ref_obj": {p["ref_obj"]},
    "bounds": {p["bounds"]},
    "x0": np.array({p["x0"]}),
    "funcs_jax": ({name}_obj_jax, {name}_constr_jax),
    "funcs_torch": ({name}_obj_torch, [{", ".join(torch_constr_names)}]),
    "ineq_indices": {p["ineq_indices"]},
    "ineq_indices_jax": {p["ineq_indices_jax"]},
    "n_constr": {p["n_constr"]}
}})
"""
            f.write(reg + "\n\n")
            print(f"Wrote {name}")
def run():
    existing_problems = get()
    if not existing_problems:
        init()
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "hs*.apm")))
    files_to_process = []
    for f in files:
        problem_name = os.path.basename(f).replace(".apm", "")
        if problem_name not in existing_problems:
            with open(f, "r") as file:
                lines = file.readlines()
                if len(lines) <= 35:
                    files_to_process.append(f)
    if not files_to_process:
        return
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch = files_to_process[i : i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}: {[os.path.basename(b) for b in batch]}")
        content = read(batch)
        prompt = construct(content)
        success = False
        for attempt in range(3):
            try:
                resp = model.generate_content(prompt)
                text = resp.text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
                    text = re.sub(r"\n```$", "", text)
                data = json.loads(text)
                append(data["problems"])
                success = True
                break
            except Exception as e:
                print(f"Error:  {e}")
                time.sleep(4)
        if not success:
            print("Failed.")
            continue
if __name__ == "__main__":
    run()