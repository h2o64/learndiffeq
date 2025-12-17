# Wrapper around the odeint framework

# Libraries
import torch
import warnings
from torchdiffeq._impl.solvers import AdaptiveStepsizeODESolver, FixedGridODESolver
from torchdiffeq._impl.misc import _check_inputs, _flat_to_shape
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.adjoint import find_parameters, handle_adjoint_norm_, OdeintAdjointMethod


class AdaptiveStepsizeODESolverWrapper:

    def __init__(self, solver):
        self.solver = solver

    def integrate(self, t):
        solution = self.solver.y0
        t = t.to(self.solver.dtype)
        self.solver._before_integrate(t)
        for i in range(1, len(t)):
            solution = self.solver._advance(t[i])
        return solution


class FixedGridODESolverWrapper:

    def __init__(self, solver):
        self.solver = solver

    def integrate(self, t):
        time_grid = self.solver.grid_constructor(self.solver.func, self.solver.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = self.solver.y0

        j = 1
        y0 = self.solver.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            dy, f0 = self.solver._step_func(self.solver.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                if self.solver.interp == "linear":
                    solution = self.solver._linear_interp(t0, t1, y0, y1, t[j])
                elif self.solver.interp == "cubic":
                    f1 = self.solver.func(t1, y1)
                    solution = self.solver._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.solver.interp}")
                j += 1
            y0 = y1

        return solution


def odeint_skip_intermediates(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):

    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(
        func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)
    if isinstance(solver, AdaptiveStepsizeODESolver):
        solver = AdaptiveStepsizeODESolverWrapper(solver)
    elif isinstance(solver, FixedGridODESolver):
        solver = FixedGridODESolverWrapper(solver)
    else:
        raise NotImplementedError('Solver not supported.')

    if event_fn is not None:
        raise NotImplementedError('Events are not supported.')

    solution = solver.integrate(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (1,), shapes)
        if isinstance(solution, tuple):
            solution = (s.squeeze(0) for s in solution)
        else:
            solution = solution.squeeze(0)

    return solution


class OdeintAdjointMethodCustom(OdeintAdjointMethod):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None

        with torch.no_grad():
            ans = odeint_skip_intermediates(func, y0, t, rtol=rtol, atol=atol,
                                            method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                y = ans
                ctx.save_for_backward(t, y, *adjoint_params)
            else:
                event_t, y = ans
                ctx.save_for_backward(t, y, event_t, *adjoint_params)

        return ans


def odeint_adjoint_skip_intermediates(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                                      adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, torch.nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(
        func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    # Handle the adjoint norm function.
    state_norm = options["norm"]
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethodCustom.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                          adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if event_fn is not None:
        raise NotImplementedError('Events are not supported.')

    solution = ans

    if shapes is not None:
        solution = _flat_to_shape(solution, (1,), shapes)
        if isinstance(solution, tuple):
            solution = (s.squeeze(0) for s in solution)
        else:
            solution = solution.squeeze(0)

    return solution
