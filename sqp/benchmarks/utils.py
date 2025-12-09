'''
Purpose of this file:
The HS benchmark problems often have variable bounds specified.
This file provides utility functions to convert these variable bounds into additional constraints,
so that solvers that do not natively support variable bounds can still handle them.
This is done for both JAX and PyTorch function representations.
'''
import numpy as np
import jax.numpy as jnp #type:ignore
import torch#type:ignore
_jax_wrapper_cache = {} #cache for JAX wrappers
_torch_wrapper_cache = {} #cache for PyTorch wrappers
_BOUND_INF = 1e9 #any bound larger than this in magnitude is considered not present
def _safe_constraint_sample(values): #function to safely sample constraint values and compute scales
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0: #return empty array if no constraints
        return np.ones(0, dtype=np.float64) #return empty array if no constraints
    arr = np.nan_to_num(arr, nan=0.0) #replace NaNs with zeros
    return np.maximum(np.abs(arr), 1.0) #scale is max of abs value and 1.0
def _compute_bound_scales(bounds, has_lower, has_upper, x0): #the purpose of this function is to scale down constraints appropriately
    bound_scales = [] #list to hold scales for each bound constraint
    for i in range(bounds.shape[0]): #iterate over each variable
        baseline = max(abs(x0[i]), 1.0) #baseline scale based on initial guess
        width = abs(bounds[i, 1] - bounds[i, 0]) #width of the bound interval
        width = max(width, 1.0) #ensure width is at least 1.0
        if has_lower[i] and has_upper[i]: #both bounds present
            lower_scale = width #scale for lower bound
            upper_scale = width #scale for upper bound
        elif has_lower[i]:
            lower_scale = max(baseline, abs(bounds[i, 0]), 1.0) #scale for lower bound
            upper_scale = None #scaling not needed
        elif has_upper[i]:
            lower_scale = None #scaling not needed
            upper_scale = max(baseline, abs(bounds[i, 1]), 1.0) #scale for upper bound
        else:
            lower_scale = None #scaling not needed
            upper_scale = None #scaling not needed
        if has_lower[i]:
            bound_scales.append(lower_scale) #append scale for lower bound
        if has_upper[i]:
            bound_scales.append(upper_scale if upper_scale is not None else width) #append scale for upper bound
    return np.asarray(bound_scales, dtype=np.float64) #return array of bound scales
def add_bounds_to_constraints_jax(prob): #main function to add bounds to constraints for JAX
    prob_name = prob['name']    #get problem name
    if prob_name in _jax_wrapper_cache: #check cache
        return _jax_wrapper_cache[prob_name] #return cached result
    bounds = np.array(prob['bounds'], dtype=np.float64)  #array of variable bounds
    if bounds.ndim == 1:  #reshape if necessary
        bounds = bounds.reshape(0, 2)#reshape to 2D array
    x0 = np.array(prob['x0'], dtype=np.float64) #initial guess
    n_vars = prob['n_vars'] #number of variables
    has_lower = bounds[:, 0] > -_BOUND_INF #check for lower bounds
    has_upper = bounds[:, 1] < _BOUND_INF #check for upper bounds
    n_bound_constraints = np.sum(has_lower) + np.sum(has_upper) #total number of bound constraints
    original_c = prob['funcs_jax'][1] #get original constraint function
    original_ineq = list(prob['ineq_indices_jax']) #get original inequality constraint indices
    n_original = prob['n_constr'] #original number of constraints
    sample_vals = original_c(jnp.array(x0, dtype=jnp.float64)) #evaluate original constraints at x0
    sample_vals = np.asarray(sample_vals, dtype=np.float64).reshape(-1) #convert to numpy array
    if sample_vals.size == 0:  #handle case with no constraints
        sample_vals = np.zeros(n_original, dtype=np.float64) #zero array if no constraints
    orig_scales = _safe_constraint_sample(sample_vals) #compute original constraint scales
    orig_scale_arr = jnp.array(orig_scales, dtype=jnp.float64) #convert to JAX array
    if orig_scale_arr.size == 0:
        orig_scale_arr = jnp.ones(0, dtype=jnp.float64) #handle empty scale array
    bound_scales = (
        _compute_bound_scales(bounds, has_lower, has_upper, x0)
        if n_bound_constraints > 0 else np.ones(0, dtype=np.float64)
    )#compute bound constraint scales
    bound_scale_arr = jnp.array(bound_scales, dtype=jnp.float64) #convert to JAX array
    def c_with_bounds(x): #purpose is to create new constraint function with bounds
        c_orig = jnp.atleast_1d(original_c(x)) #evaluate original constraints
        if orig_scale_arr.size > 0: #scale original constraints if needed
            c_orig = c_orig / orig_scale_arr.astype(x.dtype) #scale original constraints
        bound_constraints = [] #list to hold bound constraint expressions
        for i in range(n_vars): #iterate over each variable
            if has_lower[i]:
                bound_constraints.append(bounds[i, 0] - x[i]) #lower bound constraint
            if has_upper[i]:
                bound_constraints.append(x[i] - bounds[i, 1]) #upper bound constraint
        if len(bound_constraints) > 0: #if there are bound constraints
            bounds_array = jnp.array(bound_constraints, dtype=x.dtype) #convert to JAX array
            if bound_scale_arr.size > 0:
                bounds_array = bounds_array / bound_scale_arr.astype(x.dtype) #scale bound constraints
            if c_orig.size > 0:
                return jnp.concatenate([c_orig, bounds_array]) #combine original and bound constraints
            return bounds_array #only bound constraints
        return c_orig #if no bound constraints, return original
    new_ineq_indices = list(original_ineq) + list(range(n_original, n_original + n_bound_constraints)) #update inequality indices
    result = (c_with_bounds, new_ineq_indices) #create result tuple
    _jax_wrapper_cache[prob_name] = result #cache the result
    return result#return the new constraint function and indices
def add_bounds_to_constraints_torch(prob): #equivalent function for PyTorch
    prob_name = prob['name']
    if prob_name in _torch_wrapper_cache:
        return _torch_wrapper_cache[prob_name]
    bounds = np.array(prob['bounds'], dtype=np.float64)
    if bounds.ndim == 1:
        bounds = bounds.reshape(0, 2)
    x0 = np.array(prob['x0'], dtype=np.float64)
    n_vars = prob['n_vars']
    has_lower = bounds[:, 0] > -_BOUND_INF
    has_upper = bounds[:, 1] < _BOUND_INF
    n_bound_constraints = np.sum(has_lower) + np.sum(has_upper)
    c_func = prob['funcs_torch'][1]
    original_ineq = list(prob['ineq_indices'])
    n_original = prob['n_constr']
    x0_t = torch.tensor(x0, dtype=torch.double)
    with torch.no_grad():
        try:
            if isinstance(c_func, list):
                if len(c_func) == 0:
                    sample_vals = torch.zeros(0, dtype=torch.double)
                else:
                    sample_vals = torch.stack([cf(x0_t) for cf in c_func])
            else:
                sample_vals = c_func(x0_t)
        except Exception:
            sample_vals = torch.zeros(n_original, dtype=torch.double)
    orig_scales = _safe_constraint_sample(sample_vals.detach().cpu().numpy())
    torch_scale_template = torch.tensor(orig_scales, dtype=torch.double)
    if torch_scale_template.numel() == 0:
        torch_scale_template = torch.ones(0, dtype=torch.double)
    bound_scales = ( _compute_bound_scales(bounds, has_lower, has_upper, x0) if n_bound_constraints > 0 else np.ones(0, dtype=np.float64) )
    torch_bound_scale_template = torch.tensor(bound_scales, dtype=torch.double)
    def c_with_bounds(x):
        if isinstance(c_func, list):
            if len(c_func) == 0:
                c_orig = torch.zeros(0, dtype=x.dtype, device=x.device)
            else:
                c_orig = torch.stack([cf(x) for cf in c_func])
        else:
            c_orig = c_func(x)
        c_orig = c_orig.reshape(-1)
        if torch_scale_template.numel() > 0:
            scale = torch_scale_template.to(dtype=x.dtype, device=x.device)
            c_orig = c_orig / scale
        bound_constraints = []
        for i in range(n_vars):
            if has_lower[i]:
                bound_constraints.append(bounds[i, 0] - x[i])
            if has_upper[i]:
                bound_constraints.append(x[i] - bounds[i, 1])
        if len(bound_constraints) > 0:
            bounds_tensor = torch.stack([torch.as_tensor(bc, dtype=x.dtype, device=x.device) for bc in bound_constraints])
            if torch_bound_scale_template.numel() > 0:
                b_scale = torch_bound_scale_template.to(dtype=x.dtype, device=x.device)
                bounds_tensor = bounds_tensor / b_scale
            if c_orig.numel() > 0:
                return torch.cat([c_orig, bounds_tensor])
            return bounds_tensor
        return c_orig
    new_ineq_indices = list(original_ineq) + list(range(n_original, n_original + n_bound_constraints))
    result = (c_with_bounds, new_ineq_indices)
    _torch_wrapper_cache[prob_name] = result
    return result