'''
MASSIVE DUMP OF HOCK SCHITTKOWSKI PROBLEMS OBTAINED THROUGH AUTOMATED TRANSLATION FROM MATLAB TO JAX AND TORCH.
'''
import jax.numpy as jnp#type:ignore
import numpy as np
import torch #type:ignore
import jax.scipy.special  # type: ignore
PROBLEM_REGISTRY = []

# --- hs001 (JAX + Torch) ---
def hs001_obj_jax(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def hs001_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs001_obj_torch(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2



PROBLEM_REGISTRY.append({
    "name": "hs001",
    "n_vars": 2,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs001_obj_jax, hs001_constr_jax),

    "funcs_torch": (hs001_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs002 (JAX + Torch) ---
def hs002_obj_jax(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def hs002_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs002_obj_torch(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2



PROBLEM_REGISTRY.append({
    "name": "hs002",
    "n_vars": 2,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1.5]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs002_obj_jax, hs002_constr_jax),

    "funcs_torch": (hs002_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs003 (JAX + Torch) ---
def hs003_obj_jax(x):
    return x[1] + 0.00001*(x[1]-x[0])**2

def hs003_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs003_obj_torch(x):
    return x[1] + 0.00001*(x[1]-x[0])**2



PROBLEM_REGISTRY.append({
    "name": "hs003",
    "n_vars": 2,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [0.0, 1e+20]],
    "x0": np.array([10.0, 1.0]),

    "funcs_jax": (hs003_obj_jax, hs003_constr_jax),

    "funcs_torch": (hs003_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs004 (JAX + Torch) ---
def hs004_obj_jax(x):
    return (x[0]+1)**3/3 + x[1]

def hs004_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs004_obj_torch(x):
    return (x[0]+1)**3/3 + x[1]



PROBLEM_REGISTRY.append({
    "name": "hs004",
    "n_vars": 2,
    "ref_obj": 2.6666666666666665,
    "bounds": [[1.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([1.125, 0.125]),

    "funcs_jax": (hs004_obj_jax, hs004_constr_jax),

    "funcs_torch": (hs004_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs005 (JAX + Torch) ---
def hs005_obj_jax(x):
    return jnp.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def hs005_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs005_obj_torch(x):
    return torch.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1



PROBLEM_REGISTRY.append({
    "name": "hs005",
    "n_vars": 2,
    "ref_obj": -1.91322207,
    "bounds": [[-1.5, 4.0], [-3.0, 3.0]],
    "x0": np.array([0.0, 0.0]),

    "funcs_jax": (hs005_obj_jax, hs005_constr_jax),

    "funcs_torch": (hs005_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs006 (JAX + Torch) ---
def hs006_obj_jax(x):
    return (1-x[0])**2

def hs006_constr_jax(x):
    return jnp.array([
        10*(x[1] - x[0]**2)
    ], dtype=x.dtype)

def hs006_obj_torch(x):
    return (1-x[0])**2

def hs006_constr0_torch(x): return 10*(x[1] - x[0]**2)


PROBLEM_REGISTRY.append({
    "name": "hs006",
    "n_vars": 2,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-1.2, 1.0]),

    "funcs_jax": (hs006_obj_jax, hs006_constr_jax),

    "funcs_torch": (hs006_obj_torch, [hs006_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs007 (JAX + Torch) ---
def hs007_obj_jax(x):
    return jnp.log(1+x[0]**2) - x[1]

def hs007_constr_jax(x):
    return jnp.array([
        (1+x[0]**2)**2 + x[1]**2 - (4)
    ])

def hs007_obj_torch(x):
    return torch.log(1+x[0]**2) - x[1]

def hs007_constr0_torch(x): return (1+x[0]**2)**2 + x[1]**2 - (4)


PROBLEM_REGISTRY.append({
    "name": "hs007",
    "n_vars": 2,
    "ref_obj": -1.7320508100000003,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0]),

    "funcs_jax": (hs007_obj_jax, hs007_constr_jax),

    "funcs_torch": (hs007_obj_torch, [hs007_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})
# --- hs008 (JAX + Torch) ---

def hs008_obj_jax(x):
    return jnp.array(0.0, dtype=x.dtype)


def hs008_constr_jax(x):
    c1 = x[0]**2 + x[1]**2 - 25.0  
    c2 = x[0]*x[1] - 9.0             
    return jnp.array([c1, c2], dtype=x.dtype)


def hs008_obj_torch(x):
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)


def hs008_constr_torch(x):
    c1 = x[0]**2 + x[1]**2 - 25.0
    c2 = x[0]*x[1] - 9.0
    return torch.stack([c1, c2])


PROBLEM_REGISTRY.append({
    "name": "hs008",
    "n_vars": 2,
    "ref_obj": 0.0,   
    "bounds": [[-1e20, 1e20], [-1e20, 1e20]],
    "x0": np.array([2.0, 1.0]),
    "funcs_jax": (hs008_obj_jax, hs008_constr_jax),
    "funcs_torch": (hs008_obj_torch, hs008_constr_torch),
    "ineq_indices": [],     
    "ineq_indices_jax": [],
    "n_constr": 2
})

# --- hs009 (JAX + Torch) ---
def hs009_obj_jax(x):
    return jnp.sin(3.14159 * x[0] / 12) * jnp.cos(3.14159 * x[1] / 16)

def hs009_constr_jax(x):
    return jnp.array([
        4*x[0] - 3*x[1] - (0)
    ])

def hs009_obj_torch(x):
    return torch.sin(3.14159 * x[0] / 12) * torch.cos(3.14159 * x[1] / 16)

def hs009_constr0_torch(x): return 4*x[0] - 3*x[1] - (0)


PROBLEM_REGISTRY.append({
    "name": "hs009",
    "n_vars": 2,
    "ref_obj": -0.5,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.0, 0.0]),

    "funcs_jax": (hs009_obj_jax, hs009_constr_jax),

    "funcs_torch": (hs009_obj_torch, [hs009_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs010 (JAX + Torch) ---
def hs010_obj_jax(x):
    return x[0] - x[1]

def hs010_constr_jax(x):
    return jnp.array([
        -1 - (-3*x[0]**2 + 2*x[0]*x[1] - x[1]**2)
    ])

def hs010_obj_torch(x):
    return x[0] - x[1]

def hs010_constr0_torch(x): return -1 - (-3*x[0]**2 + 2*x[0]*x[1] - x[1]**2)


PROBLEM_REGISTRY.append({
    "name": "hs010",
    "n_vars": 2,
    "ref_obj": -1.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-10.0, 10.0]),

    "funcs_jax": (hs010_obj_jax, hs010_constr_jax),

    "funcs_torch": (hs010_obj_torch, [hs010_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs011 (JAX + Torch) ---
def hs011_obj_jax(x):
    return (x[0] - 5)**2 + x[1]**2 - 25

def hs011_constr_jax(x):
    return jnp.array([
        x[0]**2 - (x[1])
    ])

def hs011_obj_torch(x):
    return (x[0] - 5)**2 + x[1]**2 - 25

def hs011_constr0_torch(x): return x[0]**2 - (x[1])


PROBLEM_REGISTRY.append({
    "name": "hs011",
    "n_vars": 2,
    "ref_obj": -8.498464223,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([4.9, 0.1]),

    "funcs_jax": (hs011_obj_jax, hs011_constr_jax),

    "funcs_torch": (hs011_obj_torch, [hs011_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs012 (JAX + Torch) ---
def hs012_obj_jax(x):
    return x[0]**2/2 + x[1]**2 - x[0]*x[1] - 7*x[0] - 7*x[1]

def hs012_constr_jax(x):
    return jnp.array([
        4*x[0]**2 + x[1]**2 - (25)
    ])

def hs012_obj_torch(x):
    return x[0]**2/2 + x[1]**2 - x[0]*x[1] - 7*x[0] - 7*x[1]

def hs012_constr0_torch(x): return 4*x[0]**2 + x[1]**2 - (25)


PROBLEM_REGISTRY.append({
    "name": "hs012",
    "n_vars": 2,
    "ref_obj": -30.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.0, 0.0]),

    "funcs_jax": (hs012_obj_jax, hs012_constr_jax),

    "funcs_torch": (hs012_obj_torch, [hs012_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})
# --- hs013 (JAX + Torch) ---
def hs013_obj_jax(x):
    return (x[0] - 2.0)**2 + x[1]**2


def hs013_constr_jax(x):
    c = x[1] - (1.0 - x[0])**3
    return jnp.array([c], dtype=x.dtype)

def hs013_obj_torch(x):
    return (x[0] - 2.0)**2 + x[1]**2


def hs013_constr_torch(x):
    c = x[1] - (1.0 - x[0])**3
    return torch.stack([c])



PROBLEM_REGISTRY.append({
    "name": "hs013",
    "n_vars": 2,
    "ref_obj": 1.0,
    "bounds": [[0.0, 1e20], [0.0, 1e20]],
    "x0": np.array([1.0, 0.0]),
    "funcs_jax": (hs013_obj_jax, hs013_constr_jax),
    "funcs_torch": (hs013_obj_torch, hs013_constr_torch),
    "ineq_indices": [0],       
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs014 (JAX + Torch) ---
def hs014_obj_jax(x):
    return (x[0] - 2)**2 + (x[1]-1)**2

def hs014_constr_jax(x):
    return jnp.array([
        x[0]**2/4 + x[1]**2 - 1,
        x[0] - 2*x[1] + 1
    ], dtype=x.dtype)

def hs014_obj_torch(x):
    return (x[0] - 2)**2 + (x[1]-1)**2

def hs014_constr0_torch(x): return x[0]**2/4 + x[1]**2 - 1
def hs014_constr1_torch(x): return x[0] - 2*x[1] + 1


PROBLEM_REGISTRY.append({
    "name": "hs014",
    "n_vars": 2,
    "ref_obj": 2.2289196562095817,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([2.0, 2.0]),

    "funcs_jax": (hs014_obj_jax, hs014_constr_jax),

    "funcs_torch": (hs014_obj_torch, [hs014_constr0_torch, hs014_constr1_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 2
})


# --- hs015 (JAX + Torch) ---
def hs015_obj_jax(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def hs015_constr_jax(x):
    return jnp.array([
        1 - x[0]*x[1],
        -(x[0] + x[1]**2),
        x[0] - 1/2
    ], dtype=x.dtype)

def hs015_obj_torch(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def hs015_constr0_torch(x): return 1 - x[0]*x[1]
def hs015_constr1_torch(x): return -(x[0] + x[1]**2)
def hs015_constr2_torch(x): return x[0] - 1/2


PROBLEM_REGISTRY.append({
    "name": "hs015",
    "n_vars": 2,
    "ref_obj": 306.5,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs015_obj_jax, hs015_constr_jax),

    "funcs_torch": (hs015_obj_torch, [hs015_constr0_torch, hs015_constr1_torch, hs015_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})


# --- hs016 (JAX + Torch) ---
def hs016_obj_jax(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def hs016_constr_jax(x):
    return jnp.array([
    -(x[0]**2 + x[1]),
    -(x[0] + x[1]**2),
    -0.5 - x[0],
    x[0] - 0.5,
    x[1] - 1
    ], dtype=x.dtype)

def hs016_obj_torch(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def hs016_constr0_torch(x): return -(x[0]**2 + x[1])
def hs016_constr1_torch(x): return -(x[0] + x[1]**2)
def hs016_constr2_torch(x): return -0.5 - x[0]
def hs016_constr3_torch(x): return x[0] - 0.5
def hs016_constr4_torch(x): return x[1] - 1


PROBLEM_REGISTRY.append({
    "name": "hs016",
    "n_vars": 2,
    "ref_obj": 0.25,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs016_obj_jax, hs016_constr_jax),

    "funcs_torch": (hs016_obj_torch, [hs016_constr0_torch, hs016_constr1_torch, hs016_constr2_torch, hs016_constr3_torch, hs016_constr4_torch]),

    "ineq_indices": [0, 1, 2, 3, 4],
    "ineq_indices_jax": [0, 1, 2, 3, 4],
    "n_constr": 5
})


# --- hs017 (JAX + Torch) ---
def hs017_obj_jax(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def hs017_constr_jax(x):
    return jnp.array([
    x[0] - x[1]**2,
    -(x[0]**2 - x[1]),
    -0.5 - x[0],
    x[0] - 0.5,
    x[1] - 1
    ], dtype=x.dtype)

def hs017_obj_torch(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def hs017_constr0_torch(x): return x[0] - x[1]**2
def hs017_constr1_torch(x): return -(x[0]**2 - x[1])
def hs017_constr2_torch(x): return -0.5 - x[0]
def hs017_constr3_torch(x): return x[0] - 0.5
def hs017_constr4_torch(x): return x[1] - 1


PROBLEM_REGISTRY.append({
    "name": "hs017",
    "n_vars": 2,
    "ref_obj": 1.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs017_obj_jax, hs017_constr_jax),

    "funcs_torch": (hs017_obj_torch, [hs017_constr0_torch, hs017_constr1_torch, hs017_constr2_torch, hs017_constr3_torch, hs017_constr4_torch]),

    "ineq_indices": [0, 1, 2, 3, 4],
    "ineq_indices_jax": [0, 1, 2, 3, 4],
    "n_constr": 5
})


# --- hs018 (JAX + Torch) ---
def hs018_obj_jax(x):
    return x[0]**2/100 + x[1]**2

def hs018_constr_jax(x):
    return jnp.array([
    -(x[0]*x[1] - 25),
    -(x[0]**2 + x[1]**2 - 25)
    ], dtype=x.dtype)

def hs018_obj_torch(x):
    return x[0]**2/100 + x[1]**2

def hs018_constr0_torch(x): return -(x[0]*x[1] - 25)
def hs018_constr1_torch(x): return -(x[0]**2 + x[1]**2 - 25)


PROBLEM_REGISTRY.append({
    "name": "hs018",
    "n_vars": 2,
    "ref_obj": 5.0,
    "bounds": [[2.0, 50.0], [0.0, 50.0]],
    "x0": np.array([2.0, 2.0]),

    "funcs_jax": (hs018_obj_jax, hs018_constr_jax),

    "funcs_torch": (hs018_obj_torch, [hs018_constr0_torch, hs018_constr1_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 2
})


# --- hs019 (JAX + Torch) ---
def hs019_obj_jax(x):
    return (x[0] - 10)**3 + (x[1] - 20)**3

def hs019_constr_jax(x):
    return jnp.array([
        100 - ((x[0] - 5)**2 + (x[1] - 5)**2),
        (x[1] - 5)**2 + (x[0] - 6)**2 - 82.81,
        13 - x[0],
        x[0] - 100,
        0 - x[1],
        x[1] - 100
    ], dtype=x.dtype)

def hs019_obj_torch(x):
    return (x[0] - 10)**3 + (x[1] - 20)**3

def hs019_constr0_torch(x): return 100 - ((x[0] - 5)**2 + (x[1] - 5)**2)
def hs019_constr1_torch(x): return (x[1] - 5)**2 + (x[0] - 6)**2 - 82.81
def hs019_constr2_torch(x): return 13 - x[0]
def hs019_constr3_torch(x): return x[0] - 100
def hs019_constr4_torch(x): return 0 - x[1]
def hs019_constr5_torch(x): return x[1] - 100


PROBLEM_REGISTRY.append({
    "name": "hs019",
    "n_vars": 2,
    "ref_obj": -6961.81381,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([20.1, 5.84]),

    "funcs_jax": (hs019_obj_jax, hs019_constr_jax),

    "funcs_torch": (hs019_obj_torch, [hs019_constr0_torch, hs019_constr1_torch, hs019_constr2_torch, hs019_constr3_torch, hs019_constr4_torch, hs019_constr5_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5],
    "n_constr": 6
})


# --- hs020 (JAX + Torch) ---
def hs020_obj_jax(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def hs020_constr_jax(x):
    return jnp.array([
        0 - (x[0] + x[1]**2),
        0 - (x[0]**2 + x[1]),
        1 - (x[0]**2 + x[1]**2),
        -1/2 - x[0],
        x[0] - 1/2
    ], dtype=x.dtype)

def hs020_obj_torch(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def hs020_constr0_torch(x): return 0 - (x[0] + x[1]**2)
def hs020_constr1_torch(x): return 0 - (x[0]**2 + x[1])
def hs020_constr2_torch(x): return 1 - (x[0]**2 + x[1]**2)
def hs020_constr3_torch(x): return -1/2 - x[0]
def hs020_constr4_torch(x): return x[0] - 1/2


PROBLEM_REGISTRY.append({
    "name": "hs020",
    "n_vars": 2,
    "ref_obj": 38.19873,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs020_obj_jax, hs020_constr_jax),

    "funcs_torch": (hs020_obj_torch, [hs020_constr0_torch, hs020_constr1_torch, hs020_constr2_torch, hs020_constr3_torch, hs020_constr4_torch]),

    "ineq_indices": [0, 1, 2, 3, 4],
    "ineq_indices_jax": [0, 1, 2, 3, 4],
    "n_constr": 5
})


# --- hs021 (JAX + Torch) ---
def hs021_obj_jax(x):
    return x[0]**2/100 + x[1]**2 - 100

def hs021_constr_jax(x):
    return jnp.array([
        10 - (10*x[0] - x[1]),
        2 - x[0],
        x[0] - 50,
        -50 - x[1],
        x[1] - 50
    ], dtype=x.dtype)

def hs021_obj_torch(x):
    return x[0]**2/100 + x[1]**2 - 100

def hs021_constr0_torch(x): return 10 - (10*x[0] - x[1])
def hs021_constr1_torch(x): return 2 - x[0]
def hs021_constr2_torch(x): return x[0] - 50
def hs021_constr3_torch(x): return -50 - x[1]
def hs021_constr4_torch(x): return x[1] - 50


PROBLEM_REGISTRY.append({
    "name": "hs021",
    "n_vars": 2,
    "ref_obj": -99.96,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-1.0, 1.0]),

    "funcs_jax": (hs021_obj_jax, hs021_constr_jax),

    "funcs_torch": (hs021_obj_torch, [hs021_constr0_torch, hs021_constr1_torch, hs021_constr2_torch, hs021_constr3_torch, hs021_constr4_torch]),

    "ineq_indices": [0, 1, 2, 3, 4],
    "ineq_indices_jax": [0, 1, 2, 3, 4],
    "n_constr": 5
})


# --- hs022 (JAX + Torch) ---
def hs022_obj_jax(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def hs022_constr_jax(x):
    return jnp.array([
        x[0] + x[1] - 2,
        x[0]**2 - x[1]
    ])

def hs022_obj_torch(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def hs022_constr0_torch(x): return x[0] + x[1] - 2
def hs022_constr1_torch(x): return x[0]**2 - x[1]


PROBLEM_REGISTRY.append({
    "name": "hs022",
    "n_vars": 2,
    "ref_obj": 1.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0]),

    "funcs_jax": (hs022_obj_jax, hs022_constr_jax),

    "funcs_torch": (hs022_obj_torch, [hs022_constr0_torch, hs022_constr1_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 2
})


# --- hs023 (JAX + Torch) ---
def hs023_obj_jax(x):
    return x[0]**2 + x[1]**2

def hs023_constr_jax(x):
    return jnp.array([
        1 - (x[0] + x[1]),
        1 - (x[0]**2 + x[1]**2),
        9 - (9*x[0]**2 + x[1]**2),
        -(x[0]**2 - x[1]),
        -(x[1]**2 - x[0])
    ])

def hs023_obj_torch(x):
    return x[0]**2 + x[1]**2

def hs023_constr0_torch(x): return 1 - (x[0] + x[1])
def hs023_constr1_torch(x): return 1 - (x[0]**2 + x[1]**2)
def hs023_constr2_torch(x): return 9 - (9*x[0]**2 + x[1]**2)
def hs023_constr3_torch(x): return -(x[0]**2 - x[1])
def hs023_constr4_torch(x): return -(x[1]**2 - x[0])


PROBLEM_REGISTRY.append({
    "name": "hs023",
    "n_vars": 2,
    "ref_obj": 2.0,
    "bounds": [[-50.0, 50.0], [-50.0, 50.0]],
    "x0": np.array([3.0, 1.0]),

    "funcs_jax": (hs023_obj_jax, hs023_constr_jax),

    "funcs_torch": (hs023_obj_torch, [hs023_constr0_torch, hs023_constr1_torch, hs023_constr2_torch, hs023_constr3_torch, hs023_constr4_torch]),

    "ineq_indices": [0, 1, 2, 3, 4],
    "ineq_indices_jax": [0, 1, 2, 3, 4],
    "n_constr": 5
})


# --- hs024 (JAX + Torch) ---
def hs024_obj_jax(x):
    return ((x[0] - 3)**2 - 9) * x[1]**3 / (27*jnp.sqrt(3))

def hs024_constr_jax(x):
    return jnp.array([
        -(x[0]/jnp.sqrt(3) - x[1]),
        -(x[0] + jnp.sqrt(3)*x[1]),
        -6 - (-x[0] - jnp.sqrt(3)*x[1])
    ])

def hs024_obj_torch(x):
    return ((x[0] - 3)**2 - 9) * x[1]**3 / (27*torch.sqrt(torch.tensor(3.0)))

def hs024_constr0_torch(x): return -(x[0]/torch.sqrt(torch.tensor(3.0)) - x[1])
def hs024_constr1_torch(x): return -(x[0] + torch.sqrt(torch.tensor(3.0))*x[1])
def hs024_constr2_torch(x): return -6 - (-x[0] - torch.sqrt(torch.tensor(3.0))*x[1])


PROBLEM_REGISTRY.append({
    "name": "hs024",
    "n_vars": 2,
    "ref_obj": -1.0,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([1.0, 0.5]),

    "funcs_jax": (hs024_obj_jax, hs024_constr_jax),

    "funcs_torch": (hs024_obj_torch, [hs024_constr0_torch, hs024_constr1_torch, hs024_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})
def hs025_obj_jax(x):
    s = jnp.arange(1, 100, dtype=jnp.float32)
    z = jnp.log(jnp.clip(s / 100.0, 1e-6, 1 - 1e-6))
    base = jnp.abs(-50.0 * z) + 1e-12
    u = 25.0 + base ** (2.0 / 3.0)
    diff = jnp.clip(u - x[1], -50.0, 50.0)
    pow_term = jnp.sign(diff) * (jnp.abs(diff) + 1e-12) ** x[2]
    expo = -pow_term / jnp.clip(x[0], 1e-6, None)
    expo = jnp.clip(expo, -50.0, 50.0)
    residuals = -s / 100.0 + jnp.exp(expo)
    return jnp.sum(residuals ** 2)
def hs025_obj_torch(x):
    device = x.device
    dtype = x.dtype
    s = torch.arange(1, 100, dtype=dtype, device=device)
    z = torch.log(torch.clamp(s / 100.0, 1e-6, 1 - 1e-6))
    base = torch.abs(-50.0 * z) + 1e-12
    u = 25.0 + base ** (2.0 / 3.0)
    diff = torch.clamp(u - x[1], -50.0, 50.0)
    pow_term = torch.sign(diff) * (torch.abs(diff) + 1e-12) ** x[2]
    expo = -pow_term / torch.clamp(x[0], min=1e-6)
    expo = torch.clamp(expo, -50.0, 50.0)
    residuals = -s / 100.0 + torch.exp(expo)
    return torch.sum(residuals ** 2)
def hs025_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)
def hs025_constr_torch(x):
    return torch.zeros((0,), dtype=x.dtype)


PROBLEM_REGISTRY.append({
    "name": "hs025",
    "n_vars": 3,
    "ref_obj": 0.0,
    "bounds": [[0.1, 100.0], [0.0, 25.6], [0.0, 5.0]],
    "x0": np.array([100.0, 12.5, 3.0]),

    "funcs_jax": (hs025_obj_jax, hs025_constr_jax),

    "funcs_torch": (hs025_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs026 (JAX + Torch) ---
def hs026_obj_jax(x):
    return (x[0] - x[1])**2 + (x[1] - x[2])**4

def hs026_constr_jax(x):
    return jnp.array([
        (1 + x[1]**2)*x[0] + x[2]**4 - 3
    ], dtype=x.dtype)

def hs026_obj_torch(x):
    return (x[0] - x[1])**2 + (x[1] - x[2])**4

def hs026_constr0_torch(x): return (1 + x[1]**2)*x[0] + x[2]**4 - 3


PROBLEM_REGISTRY.append({
    "name": "hs026",
    "n_vars": 3,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.6, 2.0, 2.0]),

    "funcs_jax": (hs026_obj_jax, hs026_constr_jax),

    "funcs_torch": (hs026_obj_torch, [hs026_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs027 (JAX + Torch) ---
def hs027_obj_jax(x):
    return (x[0] - 1)**2/100 + (x[1] - x[0]**2)**2

def hs027_constr_jax(x):
    return jnp.array([
        x[0] + x[2]**2 + 1
    ], dtype=x.dtype)

def hs027_obj_torch(x):
    return (x[0] - 1)**2/100 + (x[1] - x[0]**2)**2

def hs027_constr0_torch(x): return x[0] + x[2]**2 + 1


PROBLEM_REGISTRY.append({
    "name": "hs027",
    "n_vars": 3,
    "ref_obj": 0.04,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0]),

    "funcs_jax": (hs027_obj_jax, hs027_constr_jax),

    "funcs_torch": (hs027_obj_torch, [hs027_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs028 (JAX + Torch) ---
def hs028_obj_jax(x):
    return (x[0] + x[1])**2 + (x[1] + x[2])**2

def hs028_constr_jax(x):
    return jnp.array([
x[0] + 2*x[1] + 3*x[2] - 1
    ])

def hs028_obj_torch(x):
    return (x[0] + x[1])**2 + (x[1] + x[2])**2

def hs028_constr0_torch(x): return x[0] + 2*x[1] + 3*x[2] - 1


PROBLEM_REGISTRY.append({
    "name": "hs028",
    "n_vars": 3,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-4.0, 1.0, 1.0]),

    "funcs_jax": (hs028_obj_jax, hs028_constr_jax),

    "funcs_torch": (hs028_obj_torch, [hs028_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs029 (JAX + Torch) ---
def hs029_obj_jax(x):
    return -x[0]*x[1]*x[2]

def hs029_constr_jax(x):
    return jnp.array([
x[0]**2 + 2*x[1]**2 + 4*x[2]**2 - 48
    ])

def hs029_obj_torch(x):
    return -x[0]*x[1]*x[2]

def hs029_constr0_torch(x): return x[0]**2 + 2*x[1]**2 + 4*x[2]**2 - 48


PROBLEM_REGISTRY.append({
    "name": "hs029",
    "n_vars": 3,
    "ref_obj": -22.627416997969522,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0]),

    "funcs_jax": (hs029_obj_jax, hs029_constr_jax),

    "funcs_torch": (hs029_obj_torch, [hs029_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs030 (JAX + Torch) ---
def hs030_obj_jax(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def hs030_constr_jax(x):
    return jnp.array([
x[0]**2 + x[1]**2 - 1,
1.0 - x[0],
x[0] - 10.0,
-10.0 - x[1],
x[1] - 10.0,
-10.0 - x[2],
x[2] - 10.0
    ])

def hs030_obj_torch(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def hs030_constr0_torch(x): return x[0]**2 + x[1]**2 - 1
def hs030_constr1_torch(x): return 1.0 - x[0]
def hs030_constr2_torch(x): return x[0] - 10.0
def hs030_constr3_torch(x): return -10.0 - x[1]
def hs030_constr4_torch(x): return x[1] - 10.0
def hs030_constr5_torch(x): return -10.0 - x[2]
def hs030_constr6_torch(x): return x[2] - 10.0


PROBLEM_REGISTRY.append({
    "name": "hs030",
    "n_vars": 3,
    "ref_obj": 1.0,
    "bounds": [[1.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]],
    "x0": np.array([1.0, 1.0, 1.0]),

    "funcs_jax": (hs030_obj_jax, hs030_constr_jax),

    "funcs_torch": (hs030_obj_torch, [hs030_constr0_torch, hs030_constr1_torch, hs030_constr2_torch, hs030_constr3_torch, hs030_constr4_torch, hs030_constr5_torch, hs030_constr6_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5, 6],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6],
    "n_constr": 7
})


# --- hs031 (JAX + Torch) ---
def hs031_obj_jax(x):
    return 9*x[0]**2 + x[1]**2 + 9*x[2]**2

def hs031_constr_jax(x):
    return jnp.array([
        1 - x[0] * x[1],
        x[0] - 10,
        -10 - x[0],
        x[1] - 10,
        1 - x[1],
        x[2] - 1,
        -10 - x[2]
    ], dtype=x.dtype)

def hs031_obj_torch(x):
    return 9*x[0]**2 + x[1]**2 + 9*x[2]**2

def hs031_constr0_torch(x): return 1 - x[0] * x[1]
def hs031_constr1_torch(x): return x[0] - 10
def hs031_constr2_torch(x): return -10 - x[0]
def hs031_constr3_torch(x): return x[1] - 10
def hs031_constr4_torch(x): return 1 - x[1]
def hs031_constr5_torch(x): return x[2] - 1
def hs031_constr6_torch(x): return -10 - x[2]


PROBLEM_REGISTRY.append({
    "name": "hs031",
    "n_vars": 3,
    "ref_obj": 6.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0]),

    "funcs_jax": (hs031_obj_jax, hs031_constr_jax),

    "funcs_torch": (hs031_obj_torch, [hs031_constr0_torch, hs031_constr1_torch, hs031_constr2_torch, hs031_constr3_torch, hs031_constr4_torch, hs031_constr5_torch, hs031_constr6_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5, 6],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6],
    "n_constr": 7
})


# --- hs032 (JAX + Torch) ---
def hs032_obj_jax(x):
    return (x[0] + 3*x[1] + x[2])**2 + 4*(x[0] - x[1])**2

def hs032_constr_jax(x):
    return jnp.array([
        3 - (6*x[1] + 4*x[2] - x[0]**3),
        (x[0] + x[1] + x[2]) - 1
    ], dtype=x.dtype)

def hs032_obj_torch(x):
    return (x[0] + 3*x[1] + x[2])**2 + 4*(x[0] - x[1])**2

def hs032_constr0_torch(x): return 3 - (6*x[1] + 4*x[2] - x[0]**3)
def hs032_constr1_torch(x): return (x[0] + x[1] + x[2]) - 1


PROBLEM_REGISTRY.append({
    "name": "hs032",
    "n_vars": 3,
    "ref_obj": 1.0,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([0.1, 0.7, 0.2]),

    "funcs_jax": (hs032_obj_jax, hs032_constr_jax),

    "funcs_torch": (hs032_obj_torch, [hs032_constr0_torch, hs032_constr1_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 2
})


# --- hs033 (JAX + Torch) ---
def hs033_obj_jax(x):
    return (x[0] - 1)*(x[0] - 2)*(x[0] - 3) + x[2]

def hs033_constr_jax(x):
    return jnp.array([
        x[0]**2 + x[1]**2 - x[2]**2,
        4 - (x[0]**2 + x[1]**2 + x[2]**2),
        x[2] - 5
    ], dtype=x.dtype)

def hs033_obj_torch(x):
    return (x[0] - 1)*(x[0] - 2)*(x[0] - 3) + x[2]

def hs033_constr0_torch(x): return x[0]**2 + x[1]**2 - x[2]**2
def hs033_constr1_torch(x): return 4 - (x[0]**2 + x[1]**2 + x[2]**2)
def hs033_constr2_torch(x): return x[2] - 5


PROBLEM_REGISTRY.append({
    "name": "hs033",
    "n_vars": 3,
    "ref_obj": -4.585786437626905,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([0.0, 0.0, 3.0]),

    "funcs_jax": (hs033_obj_jax, hs033_constr_jax),

    "funcs_torch": (hs033_obj_torch, [hs033_constr0_torch, hs033_constr1_torch, hs033_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})


# --- hs034 (JAX + Torch) ---
def hs034_obj_jax(x):
    return -x[0]

def hs034_constr_jax(x):
    return jnp.array([
        (jnp.exp(x[0])) - (x[1]),
        (jnp.exp(x[1])) - (x[2]),
        (x[0]) - (100),
        (x[1]) - (100),
        (x[2]) - (10)
    ])

def hs034_obj_torch(x):
    return -x[0]

def hs034_constr0_torch(x): return (torch.exp(x[0])) - (x[1])
def hs034_constr1_torch(x): return (torch.exp(x[1])) - (x[2])
def hs034_constr2_torch(x): return (x[0]) - (100)
def hs034_constr3_torch(x): return (x[1]) - (100)
def hs034_constr4_torch(x): return (x[2]) - (10)


PROBLEM_REGISTRY.append({
    "name": "hs034",
    "n_vars": 3,
    "ref_obj": -0.8340381488556516,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([0.0, 1.05, 2.9]),

    "funcs_jax": (hs034_obj_jax, hs034_constr_jax),

    "funcs_torch": (hs034_obj_torch, [hs034_constr0_torch, hs034_constr1_torch, hs034_constr2_torch, hs034_constr3_torch, hs034_constr4_torch]),

    "ineq_indices": [0, 1, 2, 3, 4],
    "ineq_indices_jax": [0, 1, 2, 3, 4],
    "n_constr": 5
})


# --- hs035 (JAX + Torch) ---
def hs035_obj_jax(x):
    return 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[0]*x[1] + 2*x[0]*x[2]

def hs035_constr_jax(x):
    return jnp.array([
        (x[0] + x[1] + 2*x[2]) - (3)
    ])

def hs035_obj_torch(x):
    return 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[0]*x[1] + 2*x[0]*x[2]

def hs035_constr0_torch(x): return (x[0] + x[1] + 2*x[2]) - (3)


PROBLEM_REGISTRY.append({
    "name": "hs035",
    "n_vars": 3,
    "ref_obj": 0.1111111111111111,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([0.5, 0.5, 0.5]),

    "funcs_jax": (hs035_obj_jax, hs035_constr_jax),

    "funcs_torch": (hs035_obj_torch, [hs035_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs036 (JAX + Torch) ---
def hs036_obj_jax(x):
    return -x[0]*x[1]*x[2]

def hs036_constr_jax(x):
    return jnp.array([
        (x[0] + 2*x[1] + 2*x[2]) - (72),
        (x[0]) - (20),
        (x[1]) - (11),
        (x[2]) - (42)
    ])

def hs036_obj_torch(x):
    return -x[0]*x[1]*x[2]

def hs036_constr0_torch(x): return (x[0] + 2*x[1] + 2*x[2]) - (72)
def hs036_constr1_torch(x): return (x[0]) - (20)
def hs036_constr2_torch(x): return (x[1]) - (11)
def hs036_constr3_torch(x): return (x[2]) - (42)


PROBLEM_REGISTRY.append({
    "name": "hs036",
    "n_vars": 3,
    "ref_obj": -3300.0,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([10.0, 10.0, 10.0]),

    "funcs_jax": (hs036_obj_jax, hs036_constr_jax),

    "funcs_torch": (hs036_obj_torch, [hs036_constr0_torch, hs036_constr1_torch, hs036_constr2_torch, hs036_constr3_torch]),

    "ineq_indices": [0, 1, 2, 3],
    "ineq_indices_jax": [0, 1, 2, 3],
    "n_constr": 4
})


# --- hs037 (JAX + Torch) ---
def hs037_obj_jax(x):
    return -x[0]*x[1]*x[2]

def hs037_constr_jax(x):
    return jnp.array([
x[0] + 2*x[1] + 2*x[2] - 72,
-(x[0] + 2*x[1] + 2*x[2])
], dtype=x.dtype)

def hs037_obj_torch(x):
    return -x[0]*x[1]*x[2]

def hs037_constr0_torch(x): return x[0] + 2*x[1] + 2*x[2] - 72
def hs037_constr1_torch(x): return -(x[0] + 2*x[1] + 2*x[2])


PROBLEM_REGISTRY.append({
    "name": "hs037",
    "n_vars": 3,
    "ref_obj": -3456.0,
    "bounds": [[0.0, 42.0], [0.0, 42.0], [0.0, 42.0]],
    "x0": np.array([10.0, 10.0, 10.0]),

    "funcs_jax": (hs037_obj_jax, hs037_constr_jax),

    "funcs_torch": (hs037_obj_torch, [hs037_constr0_torch, hs037_constr1_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 2
})


# --- hs038 (JAX + Torch) ---
def hs038_obj_jax(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 + 90*(x[3]-x[2]**2)**2 + (1-x[2])**2 + 10.1*( (x[1]-1)**2 + (x[3]-1)**2 ) + 19.8*(x[1]-1)*(x[3]-1)

def hs038_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs038_obj_torch(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 + 90*(x[3]-x[2]**2)**2 + (1-x[2])**2 + 10.1*( (x[1]-1)**2 + (x[3]-1)**2 ) + 19.8*(x[1]-1)*(x[3]-1)



PROBLEM_REGISTRY.append({
    "name": "hs038",
    "n_vars": 4,
    "ref_obj": 0.0,
    "bounds": [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]],
    "x0": np.array([-3.0, -1.0, -3.0, -1.0]),

    "funcs_jax": (hs038_obj_jax, hs038_constr_jax),

    "funcs_torch": (hs038_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs039 (JAX + Torch) ---
def hs039_obj_jax(x):
    return -x[0]

def hs039_constr_jax(x):
    return jnp.array([
x[1] - x[0]**3 - x[2]**2,
x[0]**2 - x[1] - x[3]**2
], dtype=x.dtype)

def hs039_obj_torch(x):
    return -x[0]

def hs039_constr0_torch(x): return x[1] - x[0]**3 - x[2]**2
def hs039_constr1_torch(x): return x[0]**2 - x[1] - x[3]**2


PROBLEM_REGISTRY.append({
    "name": "hs039",
    "n_vars": 4,
    "ref_obj": -1.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0, 2.0]),

    "funcs_jax": (hs039_obj_jax, hs039_constr_jax),

    "funcs_torch": (hs039_obj_torch, [hs039_constr0_torch, hs039_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs040 (JAX + Torch) ---
def hs040_obj_jax(x):
    return -x[0]*x[1]*x[2]*x[3]

def hs040_constr_jax(x):
    return jnp.array([
        x[0]**3 + x[1]**2 - (1),
        x[0]**2*x[3] - (x[2]),
        x[3]**2 - (x[1])
    ], dtype=x.dtype)

def hs040_obj_torch(x):
    return -x[0]*x[1]*x[2]*x[3]

def hs040_constr0_torch(x): return x[0]**3 + x[1]**2 - (1)
def hs040_constr1_torch(x): return x[0]**2*x[3] - (x[2])
def hs040_constr2_torch(x): return x[3]**2 - (x[1])


PROBLEM_REGISTRY.append({
    "name": "hs040",
    "n_vars": 4,
    "ref_obj": -0.25,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.8, 0.8, 0.8, 0.8]),

    "funcs_jax": (hs040_obj_jax, hs040_constr_jax),

    "funcs_torch": (hs040_obj_torch, [hs040_constr0_torch, hs040_constr1_torch, hs040_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs041 (JAX + Torch) ---
def hs041_obj_jax(x):
    return 2-x[0]*x[1]*x[2]

def hs041_constr_jax(x):
    return jnp.array([
        x[0] + 2*x[1] + 2*x[2] - x[3] - (0),
        x[0] - (1),
        x[1] - (1),
        x[2] - (1),
        x[3] - (2)
    ], dtype=x.dtype)

def hs041_obj_torch(x):
    return 2-x[0]*x[1]*x[2]

def hs041_constr0_torch(x): return x[0] + 2*x[1] + 2*x[2] - x[3] - (0)
def hs041_constr1_torch(x): return x[0] - (1)
def hs041_constr2_torch(x): return x[1] - (1)
def hs041_constr3_torch(x): return x[2] - (1)
def hs041_constr4_torch(x): return x[3] - (2)


PROBLEM_REGISTRY.append({
    "name": "hs041",
    "n_vars": 4,
    "ref_obj": 1.9259259259259258,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0, 2.0]),

    "funcs_jax": (hs041_obj_jax, hs041_constr_jax),

    "funcs_torch": (hs041_obj_torch, [hs041_constr0_torch, hs041_constr1_torch, hs041_constr2_torch, hs041_constr3_torch, hs041_constr4_torch]),

    "ineq_indices": [1, 2, 3, 4],
    "ineq_indices_jax": [1, 2, 3, 4],
    "n_constr": 5
})


# --- hs042 (JAX + Torch) ---
def hs042_obj_jax(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2 + (x[3]-4)**2

def hs042_constr_jax(x):
    return jnp.array([
        x[0] - (2),
        x[2]**2 + x[3]**2 - (2)
    ], dtype=x.dtype)

def hs042_obj_torch(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2 + (x[3]-4)**2

def hs042_constr0_torch(x): return x[0] - (2)
def hs042_constr1_torch(x): return x[2]**2 + x[3]**2 - (2)


PROBLEM_REGISTRY.append({
    "name": "hs042",
    "n_vars": 4,
    "ref_obj": 13.85786437626905,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs042_obj_jax, hs042_constr_jax),

    "funcs_torch": (hs042_obj_torch, [hs042_constr0_torch, hs042_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs043 (JAX + Torch) ---
def hs043_obj_jax(x):
    return x[0]**2 + x[1]**2 + 2*x[2]**2 + x[3]**2 - 5*x[0] - 5*x[1] - 21*x[2] + 7*x[3]

def hs043_constr_jax(x):
    return jnp.array([
        x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[0] - x[1] + x[2] - x[3] - 8,
        x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[0] - x[3] - 10,
        2*x[0]**2 + x[1]**2 + x[2]**2 + 2*x[0] - x[1] - x[3] - 5
    ], dtype=x.dtype)

def hs043_obj_torch(x):
    return x[0]**2 + x[1]**2 + 2*x[2]**2 + x[3]**2 - 5*x[0] - 5*x[1] - 21*x[2] + 7*x[3]

def hs043_constr0_torch(x): return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[0] - x[1] + x[2] - x[3] - 8
def hs043_constr1_torch(x): return x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[0] - x[3] - 10
def hs043_constr2_torch(x): return 2*x[0]**2 + x[1]**2 + x[2]**2 + 2*x[0] - x[1] - x[3] - 5


PROBLEM_REGISTRY.append({
    "name": "hs043",
    "n_vars": 4,
    "ref_obj": -44.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs043_obj_jax, hs043_constr_jax),

    "funcs_torch": (hs043_obj_torch, [hs043_constr0_torch, hs043_constr1_torch, hs043_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})


# --- hs044 (JAX + Torch) ---
def hs044_obj_jax(x):
    return x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]

def hs044_constr_jax(x):
    return jnp.array([
        x[0] + 2*x[1] - 8,
        4*x[0] + x[1] - 12,
        3*x[0] + 4*x[1] - 12,
        2*x[2] + x[3] - 8,
        x[2] + 2*x[3] - 8,
        x[2] + x[3] - 5
    ], dtype=x.dtype)

def hs044_obj_torch(x):
    return x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]

def hs044_constr0_torch(x): return x[0] + 2*x[1] - 8
def hs044_constr1_torch(x): return 4*x[0] + x[1] - 12
def hs044_constr2_torch(x): return 3*x[0] + 4*x[1] - 12
def hs044_constr3_torch(x): return 2*x[2] + x[3] - 8
def hs044_constr4_torch(x): return x[2] + 2*x[3] - 8
def hs044_constr5_torch(x): return x[2] + x[3] - 5


PROBLEM_REGISTRY.append({
    "name": "hs044",
    "n_vars": 4,
    "ref_obj": -15.0,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs044_obj_jax, hs044_constr_jax),

    "funcs_torch": (hs044_obj_torch, [hs044_constr0_torch, hs044_constr1_torch, hs044_constr2_torch, hs044_constr3_torch, hs044_constr4_torch, hs044_constr5_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5],
    "n_constr": 6
})


# --- hs045 (JAX + Torch) ---
def hs045_obj_jax(x):
    return 2 - x[0]*x[1]*x[2]*x[3]*x[4]/120

def hs045_constr_jax(x):
    return jnp.zeros((0,), dtype=x.dtype)

def hs045_obj_torch(x):
    return 2 - x[0]*x[1]*x[2]*x[3]*x[4]/120



PROBLEM_REGISTRY.append({
    "name": "hs045",
    "n_vars": 5,
    "ref_obj": 1.0,
    "bounds": [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs045_obj_jax, hs045_constr_jax),

    "funcs_torch": (hs045_obj_torch, []),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 0
})


# --- hs046 (JAX + Torch) ---
def hs046_obj_jax(x):
    return (x[0]-x[1])**2 + (x[2]-1)**2 + (x[3]-1)**4 + (x[4]-1)**6

def hs046_constr_jax(x):
    return jnp.array([
        (x[0]**2*x[3] + jnp.sin(x[3] - x[4])) - 1,
        (x[1] + x[2]**4*x[3]**2) - 2
    ])

def hs046_obj_torch(x):
    return (x[0]-x[1])**2 + (x[2]-1)**2 + (x[3]-1)**4 + (x[4]-1)**6

def hs046_constr0_torch(x): return (x[0]**2*x[3] + torch.sin(x[3] - x[4])) - 1
def hs046_constr1_torch(x): return (x[1] + x[2]**4*x[3]**2) - 2


PROBLEM_REGISTRY.append({
    "name": "hs046",
    "n_vars": 5,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.7071067811865476, 1.75, 0.5, 2.0, 2.0]),

    "funcs_jax": (hs046_obj_jax, hs046_constr_jax),

    "funcs_torch": (hs046_obj_torch, [hs046_constr0_torch, hs046_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs047 (JAX + Torch) ---
def hs047_obj_jax(x):
    return (x[0]-x[1])**2 + (x[1]-x[2])**3 + (x[2]-x[3])**4 + (x[3]-x[4])**4

def hs047_constr_jax(x):
    return jnp.array([
        (x[0] + x[1]**2 + x[2]**3) - 3,
        (x[1] - x[2]**2 + x[3]) - 1,
        (x[0]*x[4]) - 1
    ])

def hs047_obj_torch(x):
    return (x[0]-x[1])**2 + (x[1]-x[2])**3 + (x[2]-x[3])**4 + (x[3]-x[4])**4

def hs047_constr0_torch(x): return (x[0] + x[1]**2 + x[2]**3) - 3
def hs047_constr1_torch(x): return (x[1] - x[2]**2 + x[3]) - 1
def hs047_constr2_torch(x): return (x[0]*x[4]) - 1


PROBLEM_REGISTRY.append({
    "name": "hs047",
    "n_vars": 5,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 1.4142135623730951, -1.0, 0.5857864376269049, 0.5]),

    "funcs_jax": (hs047_obj_jax, hs047_constr_jax),

    "funcs_torch": (hs047_obj_torch, [hs047_constr0_torch, hs047_constr1_torch, hs047_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs048 (JAX + Torch) ---
def hs048_obj_jax(x):
    return (x[0]-1)**2 + (x[1]-x[2])**2 + (x[3]-x[4])**2

def hs048_constr_jax(x):
    return jnp.array([
        x[0] + x[1] + x[2] + x[3] + x[4] - 5,
        x[2] - 2 * (x[3] + x[4]) + 3
    ])
    

def hs048_obj_torch(x):
    return (x[0]-1)**2 + (x[1]-x[2])**2 + (x[3]-x[4])**2

def hs048_constr0_torch(x): return (((((x[0]) + x[1]) + x[2]) + x[3]) + x[4]) - 5
def hs048_constr1_torch(x): return (x[2] - 2*(x[3]+x[4])) - -3


PROBLEM_REGISTRY.append({
    "name": "hs048",
    "n_vars": 5,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([3.0, 5.0, -3.0, 2.0, -2.0]),

    "funcs_jax": (hs048_obj_jax, hs048_constr_jax),

    "funcs_torch": (hs048_obj_torch, [hs048_constr0_torch, hs048_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs049 (JAX + Torch) ---
def hs049_obj_jax(x):
    return (x[0]-x[1])**2 + (x[2]-1)**2 + (x[3]-1)**4 + (x[4]-1)**6

def hs049_constr_jax(x):
    return jnp.array([
        x[0]+x[1]+x[2]+x[3]+x[4] + 3*x[3] - (7),
        x[2] + 5*x[4] - (6)
    ])

def hs049_obj_torch(x):
    return (x[0]-x[1])**2 + (x[2]-1)**2 + (x[3]-1)**4 + (x[4]-1)**6

def hs049_constr0_torch(x): return x[0]+x[1]+x[2]+x[3]+x[4] + 3*x[3] - (7)
def hs049_constr1_torch(x): return x[2] + 5*x[4] - (6)


PROBLEM_REGISTRY.append({
    "name": "hs049",
    "n_vars": 5,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([10.0, 7.0, 2.0, -3.0, 0.8]),

    "funcs_jax": (hs049_obj_jax, hs049_constr_jax),

    "funcs_torch": (hs049_obj_torch, [hs049_constr0_torch, hs049_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs050 (JAX + Torch) ---
def hs050_obj_jax(x):
    return (x[0]-x[1])**2 + (x[1]-x[2])**2 + (x[2]-x[3])**4 + (x[3]-x[4])**2

def hs050_constr_jax(x):
    return jnp.array([
        x[0] + 2*x[1] + 3*x[2] - (6),
        x[1] + 2*x[2] + 3*x[3] - (6),
        x[2] + 2*x[3] + 3*x[4] - (6)
    ])

def hs050_obj_torch(x):
    return (x[0]-x[1])**2 + (x[1]-x[2])**2 + (x[2]-x[3])**4 + (x[3]-x[4])**2

def hs050_constr0_torch(x): return x[0] + 2*x[1] + 3*x[2] - (6)
def hs050_constr1_torch(x): return x[1] + 2*x[2] + 3*x[3] - (6)
def hs050_constr2_torch(x): return x[2] + 2*x[3] + 3*x[4] - (6)


PROBLEM_REGISTRY.append({
    "name": "hs050",
    "n_vars": 5,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([35.0, -31.0, 11.0, 5.0, -5.0]),

    "funcs_jax": (hs050_obj_jax, hs050_constr_jax),

    "funcs_torch": (hs050_obj_torch, [hs050_constr0_torch, hs050_constr1_torch, hs050_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs051 (JAX + Torch) ---
def hs051_obj_jax(x):
    return (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2

def hs051_constr_jax(x):
    return jnp.array([
        x[0] + 3*x[1] - (4),
        x[2] + x[3] - 2*x[4] - (0),
        x[1] - x[4] - (0)
    ])

def hs051_obj_torch(x):
    return (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2

def hs051_constr0_torch(x): return x[0] + 3*x[1] - (4)
def hs051_constr1_torch(x): return x[2] + x[3] - 2*x[4] - (0)
def hs051_constr2_torch(x): return x[1] - x[4] - (0)


PROBLEM_REGISTRY.append({
    "name": "hs051",
    "n_vars": 5,
    "ref_obj": 0.0,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.5, 0.5, 2.0, -1.0, 0.5]),

    "funcs_jax": (hs051_obj_jax, hs051_constr_jax),

    "funcs_torch": (hs051_obj_torch, [hs051_constr0_torch, hs051_constr1_torch, hs051_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs052 (JAX + Torch) ---
def hs052_obj_jax(x):
    return (
        (4*x[0] - x[1])**2 +
        (x[1] + x[2] - 2)**2 +
        (x[3] - 1)**2 +
        (x[4] - 1)**2
    )

def hs052_constr_jax(x):
    return jnp.array([
        x[0] + 3*x[1],
        x[2] + x[3] - 2*x[4],
        x[1] - x[4]
    ], dtype=x.dtype)


def hs052_obj_torch(x):
    return (4*x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2

def hs052_constr0_torch(x): return x[0] + 3*x[1]
def hs052_constr1_torch(x): return x[2] + x[3] - 2*x[4]
def hs052_constr2_torch(x): return x[1] - x[4]


PROBLEM_REGISTRY.append({
    "name": "hs052",
    "n_vars": 5,
    "ref_obj": 5.326647564469914,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),

    "funcs_jax": (hs052_obj_jax, hs052_constr_jax),

    "funcs_torch": (hs052_obj_torch, [hs052_constr0_torch, hs052_constr1_torch, hs052_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs053 (JAX + Torch) ---
def hs053_obj_jax(x):
    return (
        (x[0] - x[1])**2 +
        (x[1] + x[2] - 2)**2 +
        (x[3] - 1)**2 +
        (x[4] - 1)**2
    )

def hs053_constr_jax(x):
    return jnp.array([
        x[0] + 3*x[1],
        x[2] + x[3] - 2*x[4],
        x[1] - x[4]
    ], dtype=x.dtype)

def hs053_obj_torch(x):
    return (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2

def hs053_constr0_torch(x): return x[0] + 3*x[1]
def hs053_constr1_torch(x): return x[2] + x[3] - 2*x[4]
def hs053_constr2_torch(x): return x[1] - x[4]


PROBLEM_REGISTRY.append({
    "name": "hs053",
    "n_vars": 5,
    "ref_obj": 4.093023255813954,
    "bounds": [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]],
    "x0": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),

    "funcs_jax": (hs053_obj_jax, hs053_constr_jax),

    "funcs_torch": (hs053_obj_torch, [hs053_constr0_torch, hs053_constr1_torch, hs053_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})

# --- hs054 (JAX + Torch) ---

def hs054_obj_jax(x):
    quad12 = (x[0]**2 + 0.4 * x[0] * x[1] + x[1]**2) / 0.96
    quad_rest = x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2
    return quad12 + quad_rest


def hs054_constr_jax(x):
    c0 = x[0] + 0.5 * x[1] - 0.45
    return jnp.array([c0], dtype=x.dtype)


def hs054_obj_torch(x):
    quad12 = (x[0]**2 + 0.4 * x[0] * x[1] + x[1]**2) / 0.96
    quad_rest = x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2
    return quad12 + quad_rest


def hs054_constr_torch(x):
    c0 = x[0] + 0.5 * x[1] - 0.45
    return torch.stack([c0])


PROBLEM_REGISTRY.append({
    "name": "hs054",
    "n_vars": 6,
    "ref_obj": 0.19285714285714284, 
    "bounds": [
        [-1.25, 1.25],
        [-11.0, 9.0],
        [-0.2857142857142857, 1.1428571428571428],
        [-0.2, 0.2],
        [-20.02, 19.98],
        [-0.2, 0.2],
    ],
    "x0": np.array([-0.5, 0.5, 0.2857142857142857, -0.16, 0.04, -0.1]),
    "funcs_jax": (hs054_obj_jax, hs054_constr_jax),
    "funcs_torch": (hs054_obj_torch, hs054_constr_torch),
    "ineq_indices": [],         
    "ineq_indices_jax": [],
    "n_constr": 1,
})


# --- hs055 (JAX + Torch) ---
def hs055_obj_jax(x):
    return x[0] + 2*x[1] + 4*x[4] + jnp.exp(x[0]*x[3])

def hs055_constr_jax(x):
    return jnp.array([
        x[0] + 2*x[1] + 5*x[4] - (6),
        x[0] + x[1] + x[2] - (3),
        x[3] + x[4] + x[5] - (2),
        x[0] + x[3] - (1),
        x[1] + x[4] - (2),
        x[2] + x[5] - (2)
    ], dtype=x.dtype)

def hs055_obj_torch(x):
    return x[0] + 2*x[1] + 4*x[4] + torch.exp(x[0]*x[3])

def hs055_constr0_torch(x): return x[0] + 2*x[1] + 5*x[4] - (6)
def hs055_constr1_torch(x): return x[0] + x[1] + x[2] - (3)
def hs055_constr2_torch(x): return x[3] + x[4] + x[5] - (2)
def hs055_constr3_torch(x): return x[0] + x[3] - (1)
def hs055_constr4_torch(x): return x[1] + x[4] - (2)
def hs055_constr5_torch(x): return x[2] + x[5] - (2)


PROBLEM_REGISTRY.append({
    "name": "hs055",
    "n_vars": 6,
    "ref_obj": 6.333333333333333,
    "bounds": [[0.0, 1.0], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1.0], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([1.0, 2.0, 0.0, 0.0, 0.0, 2.0]),

    "funcs_jax": (hs055_obj_jax, hs055_constr_jax),

    "funcs_torch": (hs055_obj_torch, [hs055_constr0_torch, hs055_constr1_torch, hs055_constr2_torch, hs055_constr3_torch, hs055_constr4_torch, hs055_constr5_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 6
})


# --- hs056 (JAX + Torch) ---
def hs056_obj_jax(x):
    return -x[0]*x[1]*x[2]

def hs056_constr_jax(x):
    return jnp.array([
        x[0] - 4.2*jnp.sin(x[3])**2 - (0),
        x[1] - 4.2*jnp.sin(x[4])**2 - (0),
        x[2] - 4.2*jnp.sin(x[5])**2 - (0),
        x[0] + 2*x[1] + 2*x[2] - 7.2*jnp.sin(x[6])**2 - (0)
    ], dtype=x.dtype)

def hs056_obj_torch(x):
    return -x[0]*x[1]*x[2]

def hs056_constr0_torch(x): return x[0] - 4.2*torch.sin(x[3])**2 - (0)
def hs056_constr1_torch(x): return x[1] - 4.2*torch.sin(x[4])**2 - (0)
def hs056_constr2_torch(x): return x[2] - 4.2*torch.sin(x[5])**2 - (0)
def hs056_constr3_torch(x): return x[0] + 2*x[1] + 2*x[2] - 7.2*torch.sin(x[6])**2 - (0)


PROBLEM_REGISTRY.append({
    "name": "hs056",
    "n_vars": 7,
    "ref_obj": -3.456,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0, 0.5050853549265268, 0.5050853549265268, 0.5050853549265268, 0.916198947690326]),

    "funcs_jax": (hs056_obj_jax, hs056_constr_jax),

    "funcs_torch": (hs056_obj_torch, [hs056_constr0_torch, hs056_constr1_torch, hs056_constr2_torch, hs056_constr3_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 4
})


# --- hs057 (JAX + Torch) ---
def hs057_obj_jax(x):
    a_map = {1: 8, 2: 8, 3: 10, 4: 10, 5: 10, 6: 10, 7: 12, 8: 12, 9: 12, 10: 12, 11: 14, 12: 14, 13: 14, 14: 16, 15: 16, 16: 16, 17: 18, 18: 18, 19: 20, 20: 20, 21: 20, 22: 22, 23: 22, 24: 22, 25: 24, 26: 24, 27: 24, 28: 26, 29: 26, 30: 26, 31: 28, 32: 28, 33: 30, 34: 30, 35: 30, 36: 32, 37: 32, 38: 34, 39: 36, 40: 36, 41: 38, 42: 38, 43: 40, 44: 42}
    b_map = {1: 0.49, 2: 0.49, 3: 0.48, 4: 0.47, 5: 0.48, 6: 0.47, 7: 0.46, 8: 0.46, 9: 0.45, 10: 0.43, 11: 0.45, 12: 0.43, 13: 0.43, 14: 0.44, 15: 0.43, 16: 0.43, 17: 0.46, 18: 0.45, 19: 0.42, 20: 0.42, 21: 0.43, 22: 0.41, 23: 0.41, 24: 0.4, 25: 0.42, 26: 0.4, 27: 0.4, 28: 0.41, 29: 0.4, 30: 0.41, 31: 0.41, 32: 0.4, 33: 0.4, 34: 0.4, 35: 0.38, 36: 0.41, 37: 0.4, 38: 0.4, 39: 0.41, 40: 0.38, 41: 0.4, 42: 0.4, 43: 0.39, 44: 0.39}
    total_obj = 0.0
    for i in range(1, 45):
        total_obj += (b_map[i] - x[0] - (0.49 - x[0])*jnp.exp(-x[1]*(a_map[i]-8)))**2
    return total_obj

def hs057_constr_jax(x):
    return jnp.array([
        (0.09) - (0.49*x[1] - x[0]*x[1]),
        (0.4) - x[0],
        (-4) - x[1]
    ], dtype=x.dtype)

def hs057_obj_torch(x):
    a_map = {1: 8, 2: 8, 3: 10, 4: 10, 5: 10, 6: 10, 7: 12, 8: 12, 9: 12, 10: 12, 11: 14, 12: 14, 13: 14, 14: 16, 15: 16, 16: 16, 17: 18, 18: 18, 19: 20, 20: 20, 21: 20, 22: 22, 23: 22, 24: 22, 25: 24, 26: 24, 27: 24, 28: 26, 29: 26, 30: 26, 31: 28, 32: 28, 33: 30, 34: 30, 35: 30, 36: 32, 37: 32, 38: 34, 39: 36, 40: 36, 41: 38, 42: 38, 43: 40, 44: 42}
    b_map = {1: 0.49, 2: 0.49, 3: 0.48, 4: 0.47, 5: 0.48, 6: 0.47, 7: 0.46, 8: 0.46, 9: 0.45, 10: 0.43, 11: 0.45, 12: 0.43, 13: 0.43, 14: 0.44, 15: 0.43, 16: 0.43, 17: 0.46, 18: 0.45, 19: 0.42, 20: 0.42, 21: 0.43, 22: 0.41, 23: 0.41, 24: 0.4, 25: 0.42, 26: 0.4, 27: 0.4, 28: 0.41, 29: 0.4, 30: 0.41, 31: 0.41, 32: 0.4, 33: 0.4, 34: 0.4, 35: 0.38, 36: 0.41, 37: 0.4, 38: 0.4, 39: 0.41, 40: 0.38, 41: 0.4, 42: 0.4, 43: 0.39, 44: 0.39}
    total_obj = torch.tensor(0.0, dtype=x.dtype)
    for i in range(1, 45):
        total_obj += (b_map[i] - x[0] - (0.49 - x[0])*torch.exp(-x[1]*(a_map[i]-8)))**2
    return total_obj

def hs057_constr0_torch(x): return (0.09) - (0.49*x[1] - x[0]*x[1])
def hs057_constr1_torch(x): return (0.4) - x[0]
def hs057_constr2_torch(x): return (-4) - x[1]


PROBLEM_REGISTRY.append({
    "name": "hs057",
    "n_vars": 2,
    "ref_obj": 0.02845966972,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.42, 5.0]),

    "funcs_jax": (hs057_obj_jax, hs057_constr_jax),

    "funcs_torch": (hs057_obj_torch, [hs057_constr0_torch, hs057_constr1_torch, hs057_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})

# --- hs058 (JAX + Torch) ---
def hs058_obj_jax(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def hs058_constr_jax(x):
    return jnp.array([
        x[1] ** 2 - x[0],               # x2^2 - x1 = 0
        x[0] ** 2 - x[1],               # x1^2 - x2 = 0
        x[0] ** 2 + x[1] ** 2 - 1.0     # x1^2 + x2^2 - 1 = 0
    ], dtype=x.dtype)


def hs058_obj_torch(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def hs058_constr0_torch(x):  # x2^2 - x1 = 0
    return x[1] ** 2 - x[0]


def hs058_constr1_torch(x):  # x1^2 - x2 = 0
    return x[0] ** 2 - x[1]


def hs058_constr2_torch(x):  # x1^2 + x2^2 - 1 = 0
    return x[0] ** 2 + x[1] ** 2 - 1.0


PROBLEM_REGISTRY.append({
    "name": "hs058",
    "n_vars": 2,
    "ref_obj": 3.19033354957,
    "bounds": [[-2.0, 0.5], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.0]),

    "funcs_jax": (hs058_obj_jax, hs058_constr_jax),

    "funcs_torch": (hs058_obj_torch, [hs058_constr0_torch,
                                      hs058_constr1_torch,
                                      hs058_constr2_torch]),

    "ineq_indices": [],      
    "ineq_indices_jax": [],
    "n_constr": 3
})
# --- hs059 (JAX + Torch) ---
def hs059_obj_jax(x):
    x1, x2 = x
    return (
        -75.196
        + 3.8112 * x1
        + 0.0020567 * x1**3
        - 1.0345e-5 * x1**4
        + 6.8306 * x2
        - 0.030234 * x1 * x2
        + 1.28134e-3 * x2 * x1**2
        + 2.266e-7 * x1**4 * x2
        - 0.25645 * x2**2
        + 0.0034604 * x2**3
        - 1.3514e-5 * x2**4
        + 28.106 / (x2 + 1.0)
        + 5.2375e-6 * x1**2 * x2**2
        + 6.3e-8 * x1**3 * x2**2
        - 7e-10 * x1**3 * x2**3
        - 3.405e-4 * x1 * x2**2
        + 1.6638e-6 * x1 * x2**3
        + 2.8673 * jnp.exp(0.0005 * x1 * x2)
        - 3.5256e-5 * x1**3 * x2
    )


def hs059_constr_jax(x):
    x1, x2 = x
    return jnp.array([
        700.0 - x1 * x2,             
        x1**2 / 125.0 - x2,              
        5.0 * (x1 - 55.0) - (x2 - 50.0)**2
    ], dtype=x.dtype)


def hs059_obj_torch(x):
    x1, x2 = x[0], x[1]
    return (
        -75.196
        + 3.8112 * x1
        + 0.0020567 * x1**3
        - 1.0345e-5 * x1**4
        + 6.8306 * x2
        - 0.030234 * x1 * x2
        + 1.28134e-3 * x2 * x1**2
        + 2.266e-7 * x1**4 * x2
        - 0.25645 * x2**2
        + 0.0034604 * x2**3
        - 1.3514e-5 * x2**4
        + 28.106 / (x2 + 1.0)
        + 5.2375e-6 * x1**2 * x2**2
        + 6.3e-8 * x1**3 * x2**2
        - 7e-10 * x1**3 * x2**3
        - 3.405e-4 * x1 * x2**2
        + 1.6638e-6 * x1 * x2**3
        + 2.8673 * torch.exp(0.0005 * x1 * x2)
        - 3.5256e-5 * x1**3 * x2
    )


def hs059_constr0_torch(x):
    return 700.0 - x[0] * x[1]


def hs059_constr1_torch(x):
    return x[0]**2 / 125.0 - x[1]


def hs059_constr2_torch(x):
    return 5.0 * (x[0] - 55.0) - (x[1] - 50.0) ** 2


PROBLEM_REGISTRY.append({
    "name": "hs059",
    "n_vars": 2,
    "ref_obj": -7.804226324,
    "bounds": [[0.0, 75.0], [0.0, 65.0]],
    "x0": np.array([90.0, 10.0]),

    "funcs_jax": (hs059_obj_jax, hs059_constr_jax),

    "funcs_torch": (hs059_obj_torch,
                    [hs059_constr0_torch, hs059_constr1_torch, hs059_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})


# --- hs060 (JAX + Torch) ---
def hs060_obj_jax(x):
    return jnp.exp(2 * jnp.log(jnp.maximum((x[0] - 1), 1e-8))) + jnp.exp(2 * jnp.log(jnp.maximum((x[0] - x[1]), 1e-8))) + jnp.exp(4 * jnp.log(jnp.maximum((x[1] - x[2]), 1e-8)))

def hs060_constr_jax(x):
    return jnp.array([
        x[0]*(1 + jnp.exp(2 * jnp.log(jnp.maximum(x[1], 1e-8)))) + jnp.exp(4 * jnp.log(jnp.maximum(x[2], 1e-8))) - (4 + 3*jnp.sqrt(2))
    ])

def hs060_obj_torch(x):
    return torch.exp(2 * torch.log(torch.max((x[0] - 1), torch.tensor(1e-8)))) + torch.exp(2 * torch.log(torch.max((x[0] - x[1]), torch.tensor(1e-8)))) + torch.exp(4 * torch.log(torch.max((x[1] - x[2]), torch.tensor(1e-8))))

def hs060_constr0_torch(x): return x[0]*(1 + torch.exp(2 * torch.log(torch.max(x[1], torch.tensor(1e-8))))) + torch.exp(4 * torch.log(torch.max(x[2], torch.tensor(1e-8)))) - (4 + 3*torch.sqrt(torch.tensor(2.0)))


PROBLEM_REGISTRY.append({
    "name": "hs060",
    "n_vars": 3,
    "ref_obj": 0.03256820025,
    "bounds": [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]],
    "x0": np.array([2.0, 2.0, 2.0]),

    "funcs_jax": (hs060_obj_jax, hs060_constr_jax),

    "funcs_torch": (hs060_obj_torch, [hs060_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs061 (JAX + Torch) ---
def hs061_obj_jax(x):
    return 4*x[0]**2 + 2*x[1]**2 + 2*x[2]**2 - 33*x[0] + 16*x[1] - 24*x[2]

def hs061_constr_jax(x):
    return jnp.array([
        3*x[0] - 2*x[1]**2 - 7,
        4*x[0] - x[2]**2 - 11
    ], dtype=x.dtype)

def hs061_obj_torch(x):
    return 4*x[0]**2 + 2*x[1]**2 + 2*x[2]**2 - 33*x[0] + 16*x[1] - 24*x[2]

def hs061_constr0_torch(x): return 3*x[0] - 2*x[1]**2 - 7
def hs061_constr1_torch(x): return 4*x[0] - x[2]**2 - 11


PROBLEM_REGISTRY.append({
    "name": "hs061",
    "n_vars": 3,
    "ref_obj": -143.6461422,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.0, 0.0, 0.0]),

    "funcs_jax": (hs061_obj_jax, hs061_constr_jax),

    "funcs_torch": (hs061_obj_torch, [hs061_constr0_torch, hs061_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs062 (JAX + Torch) ---
def hs062_obj_jax(x):
    return -32.174*(255*jnp.log((x[0]+x[1]+x[2]+0.03)/(0.09*x[0] + x[1] + x[2] + 0.03)) +280*jnp.log((x[1]+x[2]+0.03)/(0.07*x[1] + x[2] + 0.03))+290*jnp.log((x[2]+0.03)/(0.13*x[2] + 0.03)))

def hs062_constr_jax(x):
    return jnp.array([
        x[0] + x[1] + x[2] - 1
    ], dtype=x.dtype)

def hs062_obj_torch(x):
    return -32.174*(255*torch.log((x[0]+x[1]+x[2]+0.03)/(0.09*x[0] + x[1] + x[2] + 0.03)) +280*torch.log((x[1]+x[2]+0.03)/(0.07*x[1] + x[2] + 0.03))+290*torch.log((x[2]+0.03)/(0.13*x[2] + 0.03)))

def hs062_constr0_torch(x): return x[0] + x[1] + x[2] - 1


PROBLEM_REGISTRY.append({
    "name": "hs062",
    "n_vars": 3,
    "ref_obj": -26272.51448,
    "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    "x0": np.array([0.7, 0.2, 0.1]),

    "funcs_jax": (hs062_obj_jax, hs062_constr_jax),

    "funcs_torch": (hs062_obj_torch, [hs062_constr0_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 1
})


# --- hs063 (JAX + Torch) ---
def hs063_obj_jax(x):
    return 1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2]

def hs063_constr_jax(x):
    return jnp.array([
        8*x[0] + 14*x[1] + 7*x[2] - 56,
        x[0]**2 + x[1]**2 + x[2]**2 - 25
    ], dtype=x.dtype)

def hs063_obj_torch(x):
    return 1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2]

def hs063_constr0_torch(x): return 8*x[0] + 14*x[1] + 7*x[2] - 56
def hs063_constr1_torch(x): return x[0]**2 + x[1]**2 + x[2]**2 - 25


PROBLEM_REGISTRY.append({
    "name": "hs063",
    "n_vars": 3,
    "ref_obj": 961.7151721,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0]),

    "funcs_jax": (hs063_obj_jax, hs063_constr_jax),

    "funcs_torch": (hs063_obj_torch, [hs063_constr0_torch, hs063_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs064 (JAX + Torch) ---
def hs064_obj_jax(x):
    return 5*x[0] + 50000/x[0] + 20*x[1] + 72000/x[1] + 10*x[2] + 144000/x[2]

def hs064_constr_jax(x):
    return jnp.array([
        (4/x[0] + 32/x[1] + 120/x[2]) - 1
    ])

def hs064_obj_torch(x):
    return 5*x[0] + 50000/x[0] + 20*x[1] + 72000/x[1] + 10*x[2] + 144000/x[2]

def hs064_constr0_torch(x): return (4/x[0] + 32/x[1] + 120/x[2]) - 1


PROBLEM_REGISTRY.append({
    "name": "hs064",
    "n_vars": 3,
    "ref_obj": 6299.842428,
    "bounds": [[1e-05, 1e+20], [1e-05, 1e+20], [1e-05, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0]),

    "funcs_jax": (hs064_obj_jax, hs064_constr_jax),

    "funcs_torch": (hs064_obj_torch, [hs064_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs065 (JAX + Torch) ---
def hs065_obj_jax(x):
    return (x[0] - x[1])**2 + (x[0] + x[1] - 10)**2/9 + (x[2] - 5)**2

def hs065_constr_jax(x):
    return jnp.array([
        (x[0]**2 + x[1]**2 + x[2]**2) - 48,
        x[0] - (4.5),
        (-4.5) - x[0],
        x[1] - (4.5),
        (-4.5) - x[1],
        x[2] - (5.0),
        (-5.0) - x[2]
    ])

def hs065_obj_torch(x):
    return (x[0] - x[1])**2 + (x[0] + x[1] - 10)**2/9 + (x[2] - 5)**2

def hs065_constr0_torch(x): return (x[0]**2 + x[1]**2 + x[2]**2) - 48
def hs065_constr1_torch(x): return x[0] - (4.5)
def hs065_constr2_torch(x): return (-4.5) - x[0]
def hs065_constr3_torch(x): return x[1] - (4.5)
def hs065_constr4_torch(x): return (-4.5) - x[1]
def hs065_constr5_torch(x): return x[2] - (5.0)
def hs065_constr6_torch(x): return (-5.0) - x[2]


PROBLEM_REGISTRY.append({
    "name": "hs065",
    "n_vars": 3,
    "ref_obj": 0.9535288567,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-5.0, 5.0, 0.0]),

    "funcs_jax": (hs065_obj_jax, hs065_constr_jax),

    "funcs_torch": (hs065_obj_torch, [hs065_constr0_torch, hs065_constr1_torch, hs065_constr2_torch, hs065_constr3_torch, hs065_constr4_torch, hs065_constr5_torch, hs065_constr6_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5, 6],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6],
    "n_constr": 7
})


# --- hs066 (JAX + Torch) ---
def hs066_obj_jax(x):
    return 0.2*x[2] - 0.8*x[0]

def hs066_constr_jax(x):
    return jnp.array([
        jnp.exp(x[0]) - x[1],
        jnp.exp(x[1]) - x[2],
        x[0] - (100),
        (0) - x[0],
        x[1] - (100),
        (0) - x[1],
        x[2] - (10),
        (0) - x[2]
    ])

def hs066_obj_torch(x):
    return 0.2*x[2] - 0.8*x[0]

def hs066_constr0_torch(x): return torch.exp(x[0]) - x[1]
def hs066_constr1_torch(x): return torch.exp(x[1]) - x[2]
def hs066_constr2_torch(x): return x[0] - (100)
def hs066_constr3_torch(x): return (0) - x[0]
def hs066_constr4_torch(x): return x[1] - (100)
def hs066_constr5_torch(x): return (0) - x[1]
def hs066_constr6_torch(x): return x[2] - (10)
def hs066_constr7_torch(x): return (0) - x[2]


PROBLEM_REGISTRY.append({
    "name": "hs066",
    "n_vars": 3,
    "ref_obj": 0.5181632741,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([0.0, 1.05, 2.9]),

    "funcs_jax": (hs066_obj_jax, hs066_constr_jax),

    "funcs_torch": (hs066_obj_torch, [hs066_constr0_torch, hs066_constr1_torch, hs066_constr2_torch, hs066_constr3_torch, hs066_constr4_torch, hs066_constr5_torch, hs066_constr6_torch, hs066_constr7_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5, 6, 7],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6, 7],
    "n_constr": 8
})


# --- hs067 (JAX + Torch) ---
def hs067_obj_jax(x):
    return -(0.063*x[3]*x[6] - 5.04*x[0] - 3.36*x[4] - 0.035*x[1] - 10*x[2])

def hs067_constr_jax(x):
    return jnp.array([
        0 - x[3],
        x[3] - 5000,
        0 - x[4],
        x[4] - 2000,
        85 - x[5],
        x[5] - 93,
        90 - x[6],
        x[6] - 95,
        3 - x[7],
        x[7] - 12,
        0.01 - x[8],
        x[8] - 4,
        145 - x[9],
        x[9] - 162,
        (x[4]) - (1.22*x[3] - x[0]),
        (x[7]) - ((x[1]+x[4])/x[0]),
        (x[3]) - (0.01*x[0]*(112 + 13.167*x[7] - 0.6667*x[7]**2)),
        (x[6]) - (86.35 + 1.098*x[7] - 0.038*x[7]**2 + 0.325*(x[5]-89)),
        (x[9]) - (3*x[6] - 133),
        (x[8]) - (35.82 - 0.222*x[9]),
        (x[5]) - (98000*x[2]/(x[3]*x[8] + 1000*x[2]))
    ])

def hs067_obj_torch(x):
    return -(0.063*x[3]*x[6] - 5.04*x[0] - 3.36*x[4] - 0.035*x[1] - 10*x[2])

def hs067_constr0_torch(x): return 0 - x[3]
def hs067_constr1_torch(x): return x[3] - 5000
def hs067_constr2_torch(x): return 0 - x[4]
def hs067_constr3_torch(x): return x[4] - 2000
def hs067_constr4_torch(x): return 85 - x[5]
def hs067_constr5_torch(x): return x[5] - 93
def hs067_constr6_torch(x): return 90 - x[6]
def hs067_constr7_torch(x): return x[6] - 95
def hs067_constr8_torch(x): return 3 - x[7]
def hs067_constr9_torch(x): return x[7] - 12
def hs067_constr10_torch(x): return 0.01 - x[8]
def hs067_constr11_torch(x): return x[8] - 4
def hs067_constr12_torch(x): return 145 - x[9]
def hs067_constr13_torch(x): return x[9] - 162
def hs067_constr14_torch(x): return (x[4]) - (1.22*x[3] - x[0])
def hs067_constr15_torch(x): return (x[7]) - ((x[1]+x[4])/x[0])
def hs067_constr16_torch(x): return (x[3]) - (0.01*x[0]*(112 + 13.167*x[7] - 0.6667*x[7]**2))
def hs067_constr17_torch(x): return (x[6]) - (86.35 + 1.098*x[7] - 0.038*x[7]**2 + 0.325*(x[5]-89))
def hs067_constr18_torch(x): return (x[9]) - (3*x[6] - 133)
def hs067_constr19_torch(x): return (x[8]) - (35.82 - 0.222*x[9])
def hs067_constr20_torch(x): return (x[5]) - (98000*x[2]/(x[3]*x[8] + 1000*x[2]))


PROBLEM_REGISTRY.append({
    "name": "hs067",
    "n_vars": 10,
    "ref_obj": -1162.02698006,
    "bounds": [[1.00001e-05, 1999.99999999], [1.00001e-05, 15999.99999999], [1.00001e-05, 119.99999999], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([1745.0, 12000.0, 110.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs067_obj_jax, hs067_constr_jax),

    "funcs_torch": (hs067_obj_torch, [hs067_constr0_torch, hs067_constr1_torch, hs067_constr2_torch, hs067_constr3_torch, hs067_constr4_torch, hs067_constr5_torch, hs067_constr6_torch, hs067_constr7_torch, hs067_constr8_torch, hs067_constr9_torch, hs067_constr10_torch, hs067_constr11_torch, hs067_constr12_torch, hs067_constr13_torch, hs067_constr14_torch, hs067_constr15_torch, hs067_constr16_torch, hs067_constr17_torch, hs067_constr18_torch, hs067_constr19_torch, hs067_constr20_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "n_constr": 21
})


# --- hs068 (JAX + Torch) ---
def hs068_obj_jax(x):
    return (0.0001*24 - (1.0*(jnp.exp(x[0])-1) - x[2])*x[3]/(jnp.exp(x[0]) - 1 + x[3]))/x[0]

def hs068_constr_jax(x):
    return jnp.array([
        (x[2]) - (2*jax.scipy.special.erf(-x[1])),
        (jax.scipy.special.erf(-x[1] + 4.898979485566356) + jax.scipy.special.erf(-x[1] - 4.898979485566356))
    ])

def hs068_obj_torch(x):
    return (0.0001*24 - (1.0*(torch.exp(x[0])-1) - x[2])*x[3]/(torch.exp(x[0]) - 1 + x[3]))/x[0]

def hs068_constr0_torch(x): return (x[2]) - (2*torch.erf(-x[1]))
def hs068_constr1_torch(x): return (torch.erf(-x[1] + 4.898979485566356) + torch.erf(-x[1] - 4.898979485566356))


PROBLEM_REGISTRY.append({
    "name": "hs068",
    "n_vars": 4,
    "ref_obj": -0.920425026,
    "bounds": [[0.0001, 100.0], [0.0, 100.0], [0.0, 2.0], [0.0, 2.0]],
    "x0": np.array([1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs068_obj_jax, hs068_constr_jax),

    "funcs_torch": (hs068_obj_torch, [hs068_constr0_torch, hs068_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs069 (JAX + Torch) ---
def hs069_obj_jax(x):
    return (0.1*24 - (1000.0*(jnp.exp(x[0])-1) - x[2])*x[3]/(jnp.exp(x[0]) - 1 + x[3]))/x[0]

def hs069_constr_jax(x):
    return jnp.array([
        (x[2]) - (2*jax.scipy.special.erf(-x[1])),
        (x[3]) - (jax.scipy.special.erf(-x[1] + 1.0*jnp.sqrt(24.0)) + jax.scipy.special.erf(-x[1] - 1.0*jnp.sqrt(24.0)))
    ])

def hs069_obj_torch(x):
    return (0.1*24 - (1000.0*(torch.exp(x[0])-1) - x[2])*x[3]/(torch.exp(x[0]) - 1 + x[3]))/x[0]

def hs069_constr0_torch(x): return (x[2]) - (2*torch.erf(-x[1]))
def hs069_constr1_torch(x): return (x[3]) - (torch.erf(-x[1] + 4.898979485566356) + torch.erf(-x[1] - 4.898979485566356))


PROBLEM_REGISTRY.append({
    "name": "hs069",
    "n_vars": 4,
    "ref_obj": -956.71288,
    "bounds": [[0.0001, 100.0], [0.0, 100.0], [0.0, 2.0], [0.0, 2.0]],
    "x0": np.array([1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs069_obj_jax, hs069_constr_jax),

    "funcs_torch": (hs069_obj_torch, [hs069_constr0_torch, hs069_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs070 (JAX + Torch) ---
def hs070_obj_jax(x):
    u_jax = jnp.array([100.0, 100.0, 1.0, 100.0], dtype=x.dtype)
    c_jax = jnp.array([0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype=x.dtype)
    y_obs_jax = jnp.array([0.00189, 0.1038, 0.268, 0.506, 0.577, 0.604, 0.725, 0.898, 0.947, 0.845, 0.702, 0.528, 0.385, 0.257, 0.159, 0.0869, 0.0453, 0.01509, 0.00189], dtype=x.dtype)
    def safe_power_jax(base, exponent):
        return jnp.exp(exponent * jnp.log(jnp.maximum(base, 1e-8)))

    x_2_apm = x[1]
    x_3_apm = x[2]
    x_4_apm = x[3]
    b_val = x_3_apm + (1.0 - x_3_apm) * x_4_apm

    term_1_coeff = 1.0 + 1.0 / (12.0 * x_2_apm)
    term_1_comp_1 = x_3_apm * safe_power_jax(b_val, x_2_apm)
    term_1_comp_2 = safe_power_jax(x_2_apm / 6.2832, 0.5)
    term_1_comp_3 = safe_power_jax(c_jax / 7.685, x_2_apm - 1.0)
    term_1_comp_4 = jnp.exp(x_2_apm - b_val * c_jax * x_2_apm / 7.658)
    term_1 = term_1_coeff * term_1_comp_1 * term_1_comp_2 * term_1_comp_3 * term_1_comp_4

    x_1_apm = x[0]
    term_2_coeff = 1.0 + 1.0 / (12.0 * x_1_apm)
    term_2_comp_1 = (1.0 - x_3_apm) * safe_power_jax(b_val / x_4_apm, x_1_apm)
    term_2_comp_2 = safe_power_jax(x_1_apm / 6.2832, 0.5)
    term_2_comp_3 = safe_power_jax(c_jax / 7.658, x_1_apm - 1.0)
    term_2_comp_4 = jnp.exp(x_1_apm - b_val * c_jax * x_1_apm / (7.658 * x_4_apm))
    term_2 = term_2_coeff * term_2_comp_1 * term_2_comp_2 * term_2_comp_3 * term_2_comp_4

    y_cal_jax = term_1 + term_2
    return jnp.sum((y_cal_jax - y_obs_jax)**2)

def hs070_constr_jax(x):
    def safe_power_jax(base, exponent):
        return jnp.exp(exponent * jnp.log(jnp.maximum(base, 1e-8)))

    x_3_apm = x[2]
    x_4_apm = x[3]
    b_val = x_3_apm + (1.0 - x_3_apm) * x_4_apm
    return jnp.array([
        (0.0 - (x[2] + (1.0 - x[2]) * x[3]))
    ], dtype=x.dtype)

def hs070_obj_torch(x):
    u_torch = torch.tensor([100.0, 100.0, 1.0, 100.0], dtype=x.dtype)
    c_torch = torch.tensor([0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype=x.dtype)
    y_obs_torch = torch.tensor([0.00189, 0.1038, 0.268, 0.506, 0.577, 0.604, 0.725, 0.898, 0.947, 0.845, 0.702, 0.528, 0.385, 0.257, 0.159, 0.0869, 0.0453, 0.01509, 0.00189], dtype=x.dtype)
    def safe_power_torch(base, exponent):
        return torch.exp(exponent * torch.log(torch.maximum(base, torch.tensor(1e-8, dtype=base.dtype))))

    x_2_apm = x[1]
    x_3_apm = x[2]
    x_4_apm = x[3]
    b_val = x_3_apm + (1.0 - x_3_apm) * x_4_apm

    term_1_coeff = 1.0 + 1.0 / (12.0 * x_2_apm)
    term_1_comp_1 = x_3_apm * safe_power_torch(b_val, x_2_apm)
    term_1_comp_2 = safe_power_torch(x_2_apm / 6.2832, 0.5)
    term_1_comp_3 = safe_power_torch(c_torch / 7.685, x_2_apm - 1.0)
    term_1_comp_4 = torch.exp(x_2_apm - b_val * c_torch * x_2_apm / 7.658)
    term_1 = term_1_coeff * term_1_comp_1 * term_1_comp_2 * term_1_comp_3 * term_1_comp_4

    x_1_apm = x[0]
    term_2_coeff = 1.0 + 1.0 / (12.0 * x_1_apm)
    term_2_comp_1 = (1.0 - x_3_apm) * safe_power_torch(b_val / x_4_apm, x_1_apm)
    term_2_comp_2 = safe_power_torch(x_1_apm / 6.2832, 0.5)
    term_2_comp_3 = safe_power_torch(c_torch / 7.658, x_1_apm - 1.0)
    term_2_comp_4 = torch.exp(x_1_apm - b_val * c_torch * x_1_apm / (7.658 * x_4_apm))
    term_2 = term_2_coeff * term_2_comp_1 * term_2_comp_2 * term_2_comp_3 * term_2_comp_4

    y_cal_torch = term_1 + term_2
    return torch.sum((y_cal_torch - y_obs_torch)**2)

def hs070_constr0_torch(x): 
    def safe_power_torch(base, exponent):
        return torch.exp(exponent * torch.log(torch.maximum(base, torch.tensor(1e-8, dtype=base.dtype))))

    x_3_apm = x[2]
    x_4_apm = x[3]
    b_val = x_3_apm + (1.0 - x_3_apm) * x_4_apm
    return (0.0 - (x[2] + (1.0 - x[2]) * x[3]))


PROBLEM_REGISTRY.append({
    "name": "hs070",
    "n_vars": 4,
    "ref_obj": 0.007498464,
    "bounds": [[1e-05, 100.0], [1e-05, 100.0], [1e-05, 1.0], [1e-05, 100.0]],
    "x0": np.array([2.0, 4.0, 0.04, 2.0]),

    "funcs_jax": (hs070_obj_jax, hs070_constr_jax),

    "funcs_torch": (hs070_obj_torch, [hs070_constr0_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 1
})


# --- hs071 (JAX + Torch) ---
def hs071_obj_jax(x):
    return x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]

def hs071_constr_jax(x):
    return jnp.array([
        (25.0 - x[0]*x[1]*x[2]*x[3]),
        (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) - (40)
    ], dtype=x.dtype)

def hs071_obj_torch(x):
    return x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]

def hs071_constr0_torch(x): return (25.0 - x[0]*x[1]*x[2]*x[3])
def hs071_constr1_torch(x): return (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) - (40)


PROBLEM_REGISTRY.append({
    "name": "hs071",
    "n_vars": 4,
    "ref_obj": 17.0140173,
    "bounds": [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
    "x0": np.array([1.0, 5.0, 5.0, 1.0]),

    "funcs_jax": (hs071_obj_jax, hs071_constr_jax),

    "funcs_torch": (hs071_obj_torch, [hs071_constr0_torch, hs071_constr1_torch]),

    "ineq_indices": [0],
    "ineq_indices_jax": [0],
    "n_constr": 2
})


# --- hs072 (JAX + Torch) ---
def hs072_obj_jax(x):
    return 1 + x[0] + x[1] + x[2] + x[3]

def hs072_constr_jax(x):
    return jnp.array([
        (4.0/x[0] + 2.25/x[1] + 1.0/x[2] + 0.25/x[3]) - (0.0401),
        (0.16/x[0] + 0.36/x[1] + 0.64/x[2] + 0.64/x[3]) - (0.010085),
        (x[0]) - (400000.0),
        (x[1]) - (300000.0),
        (x[2]) - (200000.0),
        (x[3]) - (100000.0)
    ], dtype=x.dtype)

def hs072_obj_torch(x):
    return 1 + x[0] + x[1] + x[2] + x[3]

def hs072_constr0_torch(x): return (4.0/x[0] + 2.25/x[1] + 1.0/x[2] + 0.25/x[3]) - (0.0401)
def hs072_constr1_torch(x): return (0.16/x[0] + 0.36/x[1] + 0.64/x[2] + 0.64/x[3]) - (0.010085)
def hs072_constr2_torch(x): return (x[0]) - (400000.0)
def hs072_constr3_torch(x): return (x[1]) - (300000.0)
def hs072_constr4_torch(x): return (x[2]) - (200000.0)
def hs072_constr5_torch(x): return (x[3]) - (100000.0)


PROBLEM_REGISTRY.append({
    "name": "hs072",
    "n_vars": 4,
    "ref_obj": 727.67937,
    "bounds": [[0.001, 400000.0], [0.001, 300000.0], [0.001, 200000.0], [0.001, 100000.0]],
    "x0": np.array([1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs072_obj_jax, hs072_constr_jax),

    "funcs_torch": (hs072_obj_torch, [hs072_constr0_torch, hs072_constr1_torch, hs072_constr2_torch, hs072_constr3_torch, hs072_constr4_torch, hs072_constr5_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5],
    "n_constr": 6
})


# --- hs073 (JAX + Torch) ---
def hs073_obj_jax(x):
    return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]

def hs073_constr_jax(x):
    return jnp.array([
        5 - (2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3]),
        (21 + 1.645*jnp.sqrt(0.28*x[0]**2 + 0.19*x[1]**2 + 20.5*x[2]**2 + 0.62*x[3]**2)) - (12*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3]),
        x[0]+x[1]+x[2]+x[3] - (1)
    ], dtype=x.dtype)

def hs073_obj_torch(x):
    return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]

def hs073_constr0_torch(x): return 5 - (2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3])
def hs073_constr1_torch(x): return (21 + 1.645*torch.sqrt(0.28*x[0]**2 + 0.19*x[1]**2 + 20.5*x[2]**2 + 0.62*x[3]**2)) - (12*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3])
def hs073_constr2_torch(x): return x[0]+x[1]+x[2]+x[3] - (1)


PROBLEM_REGISTRY.append({
    "name": "hs073",
    "n_vars": 4,
    "ref_obj": 29.894378,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs073_obj_jax, hs073_constr_jax),

    "funcs_torch": (hs073_obj_torch, [hs073_constr0_torch, hs073_constr1_torch, hs073_constr2_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 3
})


# --- hs074 (JAX + Torch) ---
def hs074_obj_jax(x):
    return 3*x[0] + 1.0e-6*x[0]**3 + 2*x[1] + 2.0e-6*x[1]**3/3

def hs074_constr_jax(x):
    return jnp.array([
        x[3] - x[2] - (0.55),
        (-0.55) - (x[3] - x[2]),
        x[0] - (1000*jnp.sin(-x[2] - 0.25) + 1000*jnp.sin(-x[3] - 0.25) + 894.8),
        x[1] - (1000*jnp.sin(x[2] - 0.25) + 1000*jnp.sin(x[2]-x[3] - 0.25) + 894.8),
        1000*jnp.sin(x[3] - 0.25) + 1000*jnp.sin(x[3] - x[2] - 0.25) + 1294.8 - (0)
    ], dtype=x.dtype)

def hs074_obj_torch(x):
    return 3*x[0] + 1.0e-6*x[0]**3 + 2*x[1] + 2.0e-6*x[1]**3/3

def hs074_constr0_torch(x): return x[3] - x[2] - (0.55)
def hs074_constr1_torch(x): return (-0.55) - (x[3] - x[2])
def hs074_constr2_torch(x): return x[0] - (1000*torch.sin(-x[2] - 0.25) + 1000*torch.sin(-x[3] - 0.25) + 894.8)
def hs074_constr3_torch(x): return x[1] - (1000*torch.sin(x[2] - 0.25) + 1000*torch.sin(x[2]-x[3] - 0.25) + 894.8)
def hs074_constr4_torch(x): return 1000*torch.sin(x[3] - 0.25) + 1000*torch.sin(x[3] - x[2] - 0.25) + 1294.8 - (0)


PROBLEM_REGISTRY.append({
    "name": "hs074",
    "n_vars": 4,
    "ref_obj": 5126.4981,
    "bounds": [[0.0, 1200.0], [0.0, 1200.0], [-0.55, 0.55], [-0.55, 0.55]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs074_obj_jax, hs074_constr_jax),

    "funcs_torch": (hs074_obj_torch, [hs074_constr0_torch, hs074_constr1_torch, hs074_constr2_torch, hs074_constr3_torch, hs074_constr4_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 5
})


# --- hs075 (JAX + Torch) ---
def hs075_obj_jax(x):
    return 3*x[0] + 1.0e-6*x[0]**3 + 2*x[1] + 2.0e-6*x[1]**3/3

def hs075_constr_jax(x):
    return jnp.array([
        x[3] - x[2] - (0.48),
        (-0.48) - (x[3] - x[2]),
        x[0] - (1000*jnp.sin(-x[2] - 0.25) + 1000*jnp.sin(-x[3] - 0.25) + 894.8),
        x[1] - (1000*jnp.sin(x[2] - 0.25) + 1000*jnp.sin(x[2]-x[3] - 0.25) + 894.8),
        1000*jnp.sin(x[3] - 0.25) + 1000*jnp.sin(x[3] - x[2] - 0.25) + 1294.8 - (0)
    ], dtype=x.dtype)

def hs075_obj_torch(x):
    return 3*x[0] + 1.0e-6*x[0]**3 + 2*x[1] + 2.0e-6*x[1]**3/3

def hs075_constr0_torch(x): return x[3] - x[2] - (0.48)
def hs075_constr1_torch(x): return (-0.48) - (x[3] - x[2])
def hs075_constr2_torch(x): return x[0] - (1000*torch.sin(-x[2] - 0.25) + 1000*torch.sin(-x[3] - 0.25) + 894.8)
def hs075_constr3_torch(x): return x[1] - (1000*torch.sin(x[2] - 0.25) + 1000*torch.sin(x[2]-x[3] - 0.25) + 894.8)
def hs075_constr4_torch(x): return 1000*torch.sin(x[3] - 0.25) + 1000*torch.sin(x[3] - x[2] - 0.25) + 1294.8 - (0)


PROBLEM_REGISTRY.append({
    "name": "hs075",
    "n_vars": 4,
    "ref_obj": 5174.4129,
    "bounds": [[0.0, 1200.0], [0.0, 1200.0], [-0.48, 0.48], [-0.48, 0.48]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs075_obj_jax, hs075_constr_jax),

    "funcs_torch": (hs075_obj_torch, [hs075_constr0_torch, hs075_constr1_torch, hs075_constr2_torch, hs075_constr3_torch, hs075_constr4_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 5
})


# --- hs076 (JAX + Torch) ---
def hs076_obj_jax(x):
    return x[0]**2 + 0.5*x[1]**2 + x[2]**2 + 0.5*x[3]**2 - x[0]*x[2] + x[2]*x[3] - x[0] - 3*x[1] + x[2] - x[3]

def hs076_constr_jax(x):
    return jnp.array([
        x[0] + 2*x[1] + x[2] + x[3] - 5,
        3*x[0] + x[1] + 2*x[2] - x[3] - 4,
        1.5 - (x[1] + 4*x[2])
    ])

def hs076_obj_torch(x):
    return x[0]**2 + 0.5*x[1]**2 + x[2]**2 + 0.5*x[3]**2 - x[0]*x[2] + x[2]*x[3] - x[0] - 3*x[1] + x[2] - x[3]

def hs076_constr0_torch(x): return x[0] + 2*x[1] + x[2] + x[3] - 5
def hs076_constr1_torch(x): return 3*x[0] + x[1] + 2*x[2] - x[3] - 4
def hs076_constr2_torch(x): return 1.5 - (x[1] + 4*x[2])


PROBLEM_REGISTRY.append({
    "name": "hs076",
    "n_vars": 4,
    "ref_obj": -4.681818181,
    "bounds": [[0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20], [0.0, 1e+20]],
    "x0": np.array([0.5, 0.5, 0.5, 0.5]),

    "funcs_jax": (hs076_obj_jax, hs076_constr_jax),

    "funcs_torch": (hs076_obj_torch, [hs076_constr0_torch, hs076_constr1_torch, hs076_constr2_torch]),

    "ineq_indices": [0, 1, 2],
    "ineq_indices_jax": [0, 1, 2],
    "n_constr": 3
})


# --- hs077 (JAX + Torch) ---
def hs077_obj_jax(x):
    return (x[0]-1)**2 + (x[0] - x[1])**2 + (x[2]-1)**2 + (x[3]-1)**4 + (x[4]-1)**6

def hs077_constr_jax(x):
    return jnp.array([
        x[0]**2 * x[3] + jnp.sin(x[3]-x[4]) - 2*jnp.sqrt(2),
        x[1] + x[2]**4 * x[3]**2 - (8 + jnp.sqrt(2))
    ])

def hs077_obj_torch(x):
    return (x[0]-1)**2 + (x[0] - x[1])**2 + (x[2]-1)**2 + (x[3]-1)**4 + (x[4]-1)**6

def hs077_constr0_torch(x): return x[0]**2 * x[3] + torch.sin(x[3]-x[4]) - 2*torch.sqrt(torch.tensor(2.0))
def hs077_constr1_torch(x): return x[1] + x[2]**4 * x[3]**2 - (8 + torch.sqrt(torch.tensor(2.0)))


PROBLEM_REGISTRY.append({
    "name": "hs077",
    "n_vars": 5,
    "ref_obj": 0.24150513,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),

    "funcs_jax": (hs077_obj_jax, hs077_constr_jax),

    "funcs_torch": (hs077_obj_torch, [hs077_constr0_torch, hs077_constr1_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 2
})


# --- hs078 (JAX + Torch) ---
def hs078_obj_jax(x):
    return x[0]*x[1]*x[2]*x[3]*x[4]

def hs078_constr_jax(x):
    return jnp.array([
        x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10,
        x[1]*x[2] - 5*x[3]*x[4],
        x[0]**3 + x[1]**3 + 1
    ])

def hs078_obj_torch(x):
    return x[0]*x[1]*x[2]*x[3]*x[4]

def hs078_constr0_torch(x): return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10
def hs078_constr1_torch(x): return x[1]*x[2] - 5*x[3]*x[4]
def hs078_constr2_torch(x): return x[0]**3 + x[1]**3 + 1


PROBLEM_REGISTRY.append({
    "name": "hs078",
    "n_vars": 5,
    "ref_obj": -2.91970041,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([-2.0, 1.5, 2.0, -1.0, -1.0]),

    "funcs_jax": (hs078_obj_jax, hs078_constr_jax),

    "funcs_torch": (hs078_obj_torch, [hs078_constr0_torch, hs078_constr1_torch, hs078_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs079 (JAX + Torch) ---
def hs079_obj_jax(x):
    return (x[0]-1)**2 + (x[0]-x[1])**2 + (x[1]-x[2])**2 + (x[2]-x[3])**4 + (x[3]-x[4])**4

def hs079_constr_jax(x):
    return jnp.array([
        x[0] + x[1]**2 + x[2]**3 - (2 + 3*jnp.sqrt(2)),
        x[1] - x[2]**2 + x[3] - (-2 + 2*jnp.sqrt(2)),
        x[0]*x[4] - (2)
    ], dtype=x.dtype)

def hs079_obj_torch(x):
    return (x[0]-1)**2 + (x[0]-x[1])**2 + (x[1]-x[2])**2 + (x[2]-x[3])**4 + (x[3]-x[4])**4

def hs079_constr0_torch(x): return x[0] + x[1]**2 + x[2]**3 - 6.242640687119285
def hs079_constr1_torch(x): return x[1] - x[2]**2 + x[3] + 0.1715728752538097
def hs079_constr2_torch(x): return x[0]*x[4] - (2)


PROBLEM_REGISTRY.append({
    "name": "hs079",
    "n_vars": 5,
    "ref_obj": 0.0787768209,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),

    "funcs_jax": (hs079_obj_jax, hs079_constr_jax),

    "funcs_torch": (hs079_obj_torch, [hs079_constr0_torch, hs079_constr1_torch, hs079_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs080 (JAX + Torch) ---
def hs080_obj_jax(x):
    return jnp.exp( x[0]*x[1]*x[2]*x[3]*x[4] )

def hs080_constr_jax(x):
    return jnp.array([
        x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2 - (10),
        x[1]*x[2] - 5*x[3]*x[4] - (0),
        x[0]**3 + x[1]**3 - (-1)
    ], dtype=x.dtype)

def hs080_obj_torch(x):
    return torch.exp( x[0]*x[1]*x[2]*x[3]*x[4] )

def hs080_constr0_torch(x): return x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2 - (10)
def hs080_constr1_torch(x): return x[1]*x[2] - 5*x[3]*x[4] - (0)
def hs080_constr2_torch(x): return x[0]**3 + x[1]**3 - (-1)


PROBLEM_REGISTRY.append({
    "name": "hs080",
    "n_vars": 5,
    "ref_obj": 0.0539498478,
    "bounds": [[-2.3, 2.3], [-2.3, 2.3], [-3.2, 3.2], [-3.2, 3.2], [-3.2, 3.2]],
    "x0": np.array([-2.0, 2.0, 2.0, -1.0, -1.0]),

    "funcs_jax": (hs080_obj_jax, hs080_constr_jax),

    "funcs_torch": (hs080_obj_torch, [hs080_constr0_torch, hs080_constr1_torch, hs080_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})


# --- hs081 (JAX + Torch) ---
def hs081_obj_jax(x):
    return jnp.exp( x[0]*x[1]*x[2]*x[3]*x[4] ) - 0.5*(x[0]**3 + x[1]**3 + 1)**2

def hs081_constr_jax(x):
    return jnp.array([
        x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2 - (10),
        x[1]*x[2] - 5*x[3]*x[4] - (0),
        x[0]**3 + x[1]**3 - (-1)
    ], dtype=x.dtype)

def hs081_obj_torch(x):
    return torch.exp( x[0]*x[1]*x[2]*x[3]*x[4] ) - 0.5*(x[0]**3 + x[1]**3 + 1)**2

def hs081_constr0_torch(x): return x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2 - (10)
def hs081_constr1_torch(x): return x[1]*x[2] - 5*x[3]*x[4] - (0)
def hs081_constr2_torch(x): return x[0]**3 + x[1]**3 - (-1)


PROBLEM_REGISTRY.append({
    "name": "hs081",
    "n_vars": 5,
    "ref_obj": 0.0539498478,
    "bounds": [[-2.3, 2.3], [-2.3, 2.3], [-3.2, 3.2], [-3.2, 3.2], [-3.2, 3.2]],
    "x0": np.array([-2.0, 2.0, 2.0, -1.0, -1.0]),

    "funcs_jax": (hs081_obj_jax, hs081_constr_jax),

    "funcs_torch": (hs081_obj_torch, [hs081_constr0_torch, hs081_constr1_torch, hs081_constr2_torch]),

    "ineq_indices": [],
    "ineq_indices_jax": [],
    "n_constr": 3
})

# --- hs093 (JAX + Torch) ---
def hs093_obj_jax(x):
    return 0.0204*x[0]*x[3]*(x[0] + x[1] + x[2]) + 0.0187*x[1]*x[2]*(x[0] + 1.57*x[1] + x[3]) + 0.0607*x[0]*x[3]*x[4]**2*(x[0] + x[1] + x[2]) + 0.0437*x[1]*x[2]*x[5]**2*(x[0] + 1.57*x[1] + x[3])

def hs093_constr_jax(x):
    return jnp.array([2.07 - (0.001 * ((x[5] * (x[4] * (x[3] * (x[2] * (x[1] * (x[0] * (1.0))))))))), (0.00062*x[0]*x[3]*x[4]**2*(x[0] + x[1] + x[2]) + 0.00058*x[1]*x[2]*x[5]**2*(x[0] + 1.57*x[1] + x[3])) - 1])

def hs093_obj_torch(x):
    return 0.0204*x[0]*x[3]*(x[0] + x[1] + x[2]) + 0.0187*x[1]*x[2]*(x[0] + 1.57*x[1] + x[3]) + 0.0607*x[0]*x[3]*x[4]**2*(x[0] + x[1] + x[2]) + 0.0437*x[1]*x[2]*x[5]**2*(x[0] + 1.57*x[1] + x[3])

def hs093_constr0_torch(x): return 2.07 - (0.001 * ((x[5] * (x[4] * (x[3] * (x[2] * (x[1] * (x[0] * (1.0)))))))))
def hs093_constr1_torch(x): return (0.00062*x[0]*x[3]*x[4]**2*(x[0] + x[1] + x[2]) + 0.00058*x[1]*x[2]*x[5]**2*(x[0] + 1.57*x[1] + x[3])) - 1


PROBLEM_REGISTRY.append({
    "name": "hs093",
    "n_vars": 6,
    "ref_obj": 135.075961,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([5.54, 4.4, 12.02, 11.82, 0.702, 0.852]),

    "funcs_jax": (hs093_obj_jax, hs093_constr_jax),

    "funcs_torch": (hs093_obj_torch, [hs093_constr0_torch, hs093_constr1_torch]),

    "ineq_indices": [0, 1],
    "ineq_indices_jax": [0, 1],
    "n_constr": 2
})

# --- hs095 (JAX + Torch) ---
def hs095_obj_jax(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs095_constr_jax(x):
    return jnp.array([
        4.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5]),
        -1.88 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5]),
        -29.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4]),
        -78.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])
    ])

def hs095_obj_torch(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs095_constr0_torch(x): return 4.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5])
def hs095_constr1_torch(x): return -1.88 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5])
def hs095_constr2_torch(x): return -29.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4])
def hs095_constr3_torch(x): return -78.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])


PROBLEM_REGISTRY.append({
    "name": "hs095",
    "n_vars": 6,
    "ref_obj": 0.015619514,
    "bounds": [[0.0, 0.31], [0.0, 0.046], [0.0, 0.068], [0.0, 0.042], [0.0, 0.028], [0.0, 0.0134]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs095_obj_jax, hs095_constr_jax),

    "funcs_torch": (hs095_obj_torch, [hs095_constr0_torch, hs095_constr1_torch, hs095_constr2_torch, hs095_constr3_torch]),

    "ineq_indices": [0, 1, 2, 3],
    "ineq_indices_jax": [0, 1, 2, 3],
    "n_constr": 4
})


# --- hs096 (JAX + Torch) ---
def hs096_obj_jax(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs096_constr_jax(x):
    return jnp.array([
        4.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5]),
        -1.88 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5]),
        -69.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4]),
        -118.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])
    ], dtype=x.dtype)

def hs096_obj_torch(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs096_constr0_torch(x): return 4.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5])
def hs096_constr1_torch(x): return -1.88 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5])
def hs096_constr2_torch(x): return -69.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4])
def hs096_constr3_torch(x): return -118.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])


PROBLEM_REGISTRY.append({
    "name": "hs096",
    "n_vars": 6,
    "ref_obj": 0.015619514,
    "bounds": [[0.0, 0.31], [0.0, 0.046], [0.0, 0.068], [0.0, 0.042], [0.0, 0.028], [0.0, 0.0134]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs096_obj_jax, hs096_constr_jax),

    "funcs_torch": (hs096_obj_torch, [hs096_constr0_torch, hs096_constr1_torch, hs096_constr2_torch, hs096_constr3_torch]),

    "ineq_indices": [0, 1, 2, 3],
    "ineq_indices_jax": [0, 1, 2, 3],
    "n_constr": 4
})


# --- hs097 (JAX + Torch) ---
def hs097_obj_jax(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs097_constr_jax(x):
    return jnp.array([
        32.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5]),
        25.12 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5]),
        -29.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4]),
        -78.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])
    ])

def hs097_obj_torch(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs097_constr0_torch(x): return 32.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5])
def hs097_constr1_torch(x): return 25.12 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5])
def hs097_constr2_torch(x): return -29.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4])
def hs097_constr3_torch(x): return -78.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])


PROBLEM_REGISTRY.append({
    "name": "hs097",
    "n_vars": 6,
    "ref_obj": 3.1358091,
    "bounds": [[0.0, 0.31], [0.0, 0.046], [0.0, 0.068], [0.0, 0.042], [0.0, 0.028], [0.0, 0.0134]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs097_obj_jax, hs097_constr_jax),

    "funcs_torch": (hs097_obj_torch, [hs097_constr0_torch, hs097_constr1_torch, hs097_constr2_torch, hs097_constr3_torch]),

    "ineq_indices": [0, 1, 2, 3],
    "ineq_indices_jax": [0, 1, 2, 3],
    "n_constr": 4
})


# --- hs098 (JAX + Torch) ---
def hs098_obj_jax(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs098_constr_jax(x):
    return jnp.array([
        32.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5]),
        25.12 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5]),
        -124.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4]),
        -173.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])
    ], dtype=x.dtype)

def hs098_obj_torch(x):
    return 4.3*x[0] + 31.8*x[1] + 63.3*x[2] + 15.8*x[3] + 68.5*x[4] + 4.7*x[5]

def hs098_constr0_torch(x): return 32.97 - (17.1*x[0] + 38.2*x[1] + 204.2*x[2] + 212.3*x[3] + 623.4*x[4] + 1495.5*x[5] - 169*x[0]*x[2] - 3580*x[2]*x[4] - 3810*x[3]*x[4] - 18500*x[3]*x[5] - 24300*x[4]*x[5])
def hs098_constr1_torch(x): return 25.12 - (17.9*x[0] + 36.8*x[1] + 113.9*x[2] + 169.7*x[3] + 337.8*x[4] + 1385.2*x[5] - 139*x[0]*x[2] - 2450*x[3]*x[4] - 16600*x[3]*x[5] - 17200*x[4]*x[5])
def hs098_constr2_torch(x): return -124.08 - (-273*x[1] - 70*x[3] - 819*x[4] + 26000*x[3]*x[4])
def hs098_constr3_torch(x): return -173.02 - (159.9*x[0] - 311*x[1] + 587*x[3] + 391*x[4] + 2198*x[5] - 14000*x[0]*x[5])


PROBLEM_REGISTRY.append({
    "name": "hs098",
    "n_vars": 6,
    "ref_obj": 3.1358091,
    "bounds": [[0.0, 0.31], [0.0, 0.046], [0.0, 0.068], [0.0, 0.042], [0.0, 0.028], [0.0, 0.0134]],
    "x0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    "funcs_jax": (hs098_obj_jax, hs098_constr_jax),

    "funcs_torch": (hs098_obj_torch, [hs098_constr0_torch, hs098_constr1_torch, hs098_constr2_torch, hs098_constr3_torch]),

    "ineq_indices": [0, 1, 2, 3],
    "ineq_indices_jax": [0, 1, 2, 3],
    "n_constr": 4
})


# --- hs100 (JAX + Torch) ---
def hs100_obj_jax(x):
    return (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 + 10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]

def hs100_constr_jax(x):
    return jnp.array([
        2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4] - 127,
        7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4] - 282,
        23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6] - 196,
        4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
    ], dtype=x.dtype)

def hs100_obj_torch(x):
    return (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 + 10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]

def hs100_constr0_torch(x): return 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4] - 127
def hs100_constr1_torch(x): return 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4] - 282
def hs100_constr2_torch(x): return 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6] - 196
def hs100_constr3_torch(x): return 4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]


PROBLEM_REGISTRY.append({
    "name": "hs100",
    "n_vars": 7,
    "ref_obj": 680.6300573,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0]),

    "funcs_jax": (hs100_obj_jax, hs100_constr_jax),

    "funcs_torch": (hs100_obj_torch, [hs100_constr0_torch, hs100_constr1_torch, hs100_constr2_torch, hs100_constr3_torch]),

    "ineq_indices": [0, 1, 2, 3],
    "ineq_indices_jax": [0, 1, 2, 3],
    "n_constr": 4
})


# --- hs104 (JAX + Torch) ---
def hs104_obj_jax(x):
    return 0.4*x[0]**0.67*x[6]**(-0.67)+0.4*x[1]**0.67*x[7]**(-0.67)+10-x[0]-x[1]

def hs104_constr_jax(x):
    return jnp.array([
        -(1-0.0588*x[4]*x[6]-0.1*x[0]),
        -(1-0.0588*x[5]*x[7]-0.1*x[0]-0.1*x[1]),
        -(1-4*x[2]/x[4]-2/(x[2]**0.71*x[4])-0.0588*x[6]/x[2]**1.3),
        -(1-4*x[3]/x[5]-2/(x[3]**0.71*x[5])-0.0588*x[7]/x[3]**1.3),
        0.1-(0.4*x[0]**0.67*x[6]**(-0.67)+0.4*x[1]**0.67*x[7]**(-0.67)+10-x[0]-x[1]),
        0.4*x[0]**0.67*x[6]**(-0.67)+0.4*x[1]**0.67*x[7]**(-0.67)+10-x[0]-x[1]-4.2
    ], dtype=x.dtype)

def hs104_obj_torch(x):
    return 0.4*x[0]**0.67*x[6]**(-0.67)+0.4*x[1]**0.67*x[7]**(-0.67)+10-x[0]-x[1]

def hs104_constr0_torch(x): return -(1-0.0588*x[4]*x[6]-0.1*x[0])
def hs104_constr1_torch(x): return -(1-0.0588*x[5]*x[7]-0.1*x[0]-0.1*x[1])
def hs104_constr2_torch(x): return -(1-4*x[2]/x[4]-2/(x[2]**0.71*x[4])-0.0588*x[6]/x[2]**1.3)
def hs104_constr3_torch(x): return -(1-4*x[3]/x[5]-2/(x[3]**0.71*x[5])-0.0588*x[7]/x[3]**1.3)
def hs104_constr4_torch(x): return 0.1-(0.4*x[0]**0.67*x[6]**(-0.67)+0.4*x[1]**0.67*x[7]**(-0.67)+10-x[0]-x[1])
def hs104_constr5_torch(x): return 0.4*x[0]**0.67*x[6]**(-0.67)+0.4*x[1]**0.67*x[7]**(-0.67)+10-x[0]-x[1]-4.2


PROBLEM_REGISTRY.append({
    "name": "hs104",
    "n_vars": 8,
    "ref_obj": 3.9511634396,
    "bounds": [[0.1, 10.0], [0.1, 10.0], [0.1, 10.0], [0.1, 10.0], [0.1, 10.0], [0.1, 10.0], [0.1, 10.0], [0.1, 10.0]],
    "x0": np.array([6.0, 3.0, 0.4, 0.2, 6.0, 6.0, 1.0, 0.5]),

    "funcs_jax": (hs104_obj_jax, hs104_constr_jax),

    "funcs_torch": (hs104_obj_torch, [hs104_constr0_torch, hs104_constr1_torch, hs104_constr2_torch, hs104_constr3_torch, hs104_constr4_torch, hs104_constr5_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5],
    "n_constr": 6
})


# --- hs108 (JAX + Torch) ---
def hs108_obj_jax(x):
    return -0.5*(x[0]*x[3]-x[1]*x[2]+x[2]*x[8]-x[4]*x[8]+x[4]*x[7]-x[5]*x[6])

def hs108_constr_jax(x):
    return jnp.array([
        x[2]**2 + x[3]**2 - 1,
        x[4]**2 + x[5]**2 - 1,
        x[8]**2 - 1,
        x[0]**2 + (x[1]-x[8])**2 - 1,
        (x[0]-x[4])**2 + (x[1]-x[5])**2 - 1,
        (x[0]-x[6])**2 + (x[1]-x[7])**2 - 1,
        (x[2]-x[6])**2 + (x[3]-x[7])**2 - 1,
        (x[2]-x[4])**2 + (x[3]-x[5])**2 - 1,
        x[6]**2 + (x[7]-x[8])**2 - 1,
        -(x[0]*x[3]-x[1]*x[2]),
        -(x[2]*x[8]),
        -(-x[4]*x[8]),
        -(x[4]*x[7]-x[5]*x[6]),
        -x[8]
    ], dtype=x.dtype)

def hs108_obj_torch(x):
    return -0.5*(x[0]*x[3]-x[1]*x[2]+x[2]*x[8]-x[4]*x[8]+x[4]*x[7]-x[5]*x[6])

def hs108_constr0_torch(x): return x[2]**2 + x[3]**2 - 1
def hs108_constr1_torch(x): return x[4]**2 + x[5]**2 - 1
def hs108_constr2_torch(x): return x[8]**2 - 1
def hs108_constr3_torch(x): return x[0]**2 + (x[1]-x[8])**2 - 1
def hs108_constr4_torch(x): return (x[0]-x[4])**2 + (x[1]-x[5])**2 - 1
def hs108_constr5_torch(x): return (x[0]-x[6])**2 + (x[1]-x[7])**2 - 1
def hs108_constr6_torch(x): return (x[2]-x[6])**2 + (x[3]-x[7])**2 - 1
def hs108_constr7_torch(x): return (x[2]-x[4])**2 + (x[3]-x[5])**2 - 1
def hs108_constr8_torch(x): return x[6]**2 + (x[7]-x[8])**2 - 1
def hs108_constr9_torch(x): return -(x[0]*x[3]-x[1]*x[2])
def hs108_constr10_torch(x): return -(x[2]*x[8])
def hs108_constr11_torch(x): return -(-x[4]*x[8])
def hs108_constr12_torch(x): return -(x[4]*x[7]-x[5]*x[6])
def hs108_constr13_torch(x): return -x[8]


PROBLEM_REGISTRY.append({
    "name": "hs108",
    "n_vars": 9,
    "ref_obj": -0.8660254038,
    "bounds": [[-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20], [-1e+20, 1e+20]],
    "x0": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),

    "funcs_jax": (hs108_obj_jax, hs108_constr_jax),

    "funcs_torch": (hs108_obj_torch, [hs108_constr0_torch, hs108_constr1_torch, hs108_constr2_torch, hs108_constr3_torch, hs108_constr4_torch, hs108_constr5_torch, hs108_constr6_torch, hs108_constr7_torch, hs108_constr8_torch, hs108_constr9_torch, hs108_constr10_torch, hs108_constr11_torch, hs108_constr12_torch, hs108_constr13_torch]),

    "ineq_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "ineq_indices_jax": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "n_constr": 14
})


