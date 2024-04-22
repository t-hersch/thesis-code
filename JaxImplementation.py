import jax
import liealgebra as fla
import jax.numpy as jnp
import numpy as np
import diffrax
import functools as ft
import datetime as dt
from typing import Optional
import inspect as isp

from cubature_construction import wiener_cubature, dim_2_wiener_cubature


def get_partition(interval: tuple[float, float],
                  num_intervals: int,
                  gamma: float) -> list[float]:
    """
    Function to create partition, as described in Lyons & Victoir, Example 3.7.
    Gamma = 1 corresponds to uniform partition.

    :param interval: interval to partition.
    :param num_intervals: number of sub-intervals to partition the interval into.
    :param gamma: parameter. Large values of gamma correspond to larger intervals in the beginning, and smaller
    intervals at the end.
    :return: list of floats representing the points in the partition. Includes t_0 and t_1.
    """
    t_0, t_1 = interval
    points = [t_0]

    for j in range(1, num_intervals + 1):
        point = t_0 + (t_1 - t_0) * (1 - (1 - j / num_intervals) ** gamma)
        points.append(point)

    return points


def get_vf_derivatives(func: callable,
                       # args: jax.typing.ArrayLike | tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
                       truncation_level: int) -> list[callable]:

    # jaxpr = jax.make_jaxpr(func)(*args)
    #
    # if len(jaxpr.in_avals) > 1:
    #     final_func = ft.partial(func, t=args[0])
    #
    # else:
    #     final_func = func

    aux = func
    result = [aux]

    def auxiliary_func(x, f_circ, f):
        return jnp.tensordot(jax.jacfwd(f_circ)(x), f(x), axes=1)

    for i in range(truncation_level - 1):
        result.append(ft.partial(auxiliary_func, f_circ=result[-1], f=func))

    return result


def _add_callable(f: callable,
                  g: callable) -> callable:
    return lambda x: f(x) + g(x)


def _word_to_idx(word: fla.Word) -> tuple[int, ...]:
    """
    Helper Function; Converts an fla.Word into an index that can be used with jax.numpy.
    """
    return tuple([int(l) for l in word.letters])


def _inhomogeneous_degree(word: fla.Word) -> int:
    """
    Helper function; Computes the inhomogeneous degree of an fla.Word for truncation purposes.
    """
    return len(word.letters) + sum([1 for letter in word.letters if letter == 0])


def _elt_to_tuple(elt: fla.Elt,
                  dim: int,
                  truncation_level: int) -> tuple[jax.typing.ArrayLike, ...]:
    """
    Helper function; represents an element of the Lie Algebra as a tuple of n-dim arrays. Note: we do
    not add the 0-th level of the element to the new representation.

    :param elt: fla.Elt element to convert
    :param dim: dimension of space over which the tensor algebra is defined
    :param truncation_level: truncation level
    :return: tuple of size truncation_level, where tuple[i] is the i+1 th level of elt.
             shape of tuple[i] is (dim, dim, ..., dim) (i+1 times)
    """

    tensor_list = [np.zeros([dim] * i) for i in range(1, truncation_level + 1)]

    for data_dict in elt.data:
        if data_dict.keys():
            hom_level = len(list(data_dict.keys())[0].letters)
            if hom_level <= truncation_level:
                curr_tensor = tensor_list[hom_level - 1]

                for word, value in data_dict.items():
                    if _inhomogeneous_degree(word) <= truncation_level:
                        curr_tensor[_word_to_idx(word)] = value

    return tuple(tensor_list)


def _tuple_to_jax(tpl: tuple[jax.typing.ArrayLike, ...],
                  dim: int,
                  truncation_level: int) -> jax.typing.ArrayLike:
    """
    Helper function; converts an element of the Lie Algebra as represented in _elt_to_tuple into a Jax array. To do
    this, we add an extra dimension to the array to index the levels.
    :param tpl: tuple object to convert
    :param dim: dimension of space over which the tensor algebra is defined
    :param truncation_level: truncation level
    :return: Jax array of shape (truncation_level, dim, dim, ... dim), where dim is repeated truncation_level times.
    the last level of the element will be indexed by [truncation_level - 1, ...], the second to last will be
    indexed by [truncation_level - 2, 0, ...], and so on.
    """
    aux = []

    for i, array in enumerate(tpl):
        if i < truncation_level - 1:
            array = jnp.stack([array] + [jnp.zeros(array.shape)] * (dim - 1), axis=0)
            for _ in range(i + 2, truncation_level):
                array = jnp.stack([array] + [jnp.zeros(array.shape)] * (dim - 1), axis=0)
        aux.append(array)

    stacked_array = jnp.stack(aux, axis=0)

    for i in range(truncation_level):
        index = [i] + [0] * (truncation_level - i - 1)
        assert jnp.all(stacked_array[tuple(index)] == tpl[i])

    return stacked_array


def _rescale_elt(elt: fla.Elt,
                 scale: float,
                 signature_flag: bool) -> fla.Elt:
    """
    Helper function; implements the inhomogeneous rescaling of an element of the Lie Algebra.

    :param elt: Lie Algebra element to rescale
    :param scale: scaling factor
    :return: rescaled element
    """
    if not signature_flag:
        rescaled = [{fla.emptyWord: fla.unit_coefficient()}]

    else:
        rescaled = [{}]

    for data_dict in elt.data:
        rescaled_dict = {}

        if data_dict.keys() and fla.emptyWord not in data_dict.keys():
            for word, value in data_dict.items():
                num_zeros = len([x for x in word.letters if x == 0])
                num_non_zeros = len(word.letters) - num_zeros
                rescaled_dict[word] = value * (scale ** (num_zeros + num_non_zeros / 2))

            rescaled.append(rescaled_dict)

    if not signature_flag:
        return fla.log(fla.Elt(rescaled))

    else:
        return fla.Elt(rescaled)


def _pad_to_power_two(lst):
    """
    pads a list at the end with 0's until it reaches length that is power of 2.
    """
    n = len(lst)
    new_length = 2 ** int(jnp.ceil(jnp.log2(n)))
    return jnp.concatenate([jnp.array(lst), jnp.zeros((new_length - n))])

key = jax.random.PRNGKey(42)

def _get_intermediate_massses(masses, k):
    """
    Helper function, computes masses of tree nodes needed for intermediate tree embedding of TBBA algo.

    :param masses: array of length 2^k, representing n * a(m_j) for each child of the node j.
    See https://www.degruyter.com/document/doi/10.1515/mcma.2002.8.4.343/html?lang=en for details.

    :param k: number of intermediate steps we need. this is log_2(n), where n is the smallest power of two greater
    than or equal to the number of points in our cubature measure.

    :return: list of length k, i-th element is tuple of arrays of length 2^i, first array is masses of left nodes,
    second array is masses of right nodes.
    """
    step_masses = []
    final = []

    for i in range(1, k+1):
        new_prob_list = jax.lax.map(jnp.sum, jnp.stack(jnp.split(masses, 2 ** i)))
        step_masses.append(new_prob_list)

    for array in step_masses:
        new = (array[::2], array[1::2])
        final.append(new)

    return final


def _TBBA_sampling(key, cubature, num_intervals, num_particles):
    """
    Main sampling function, see Lyons Crisan 2009 for details.
    https://www.degruyter.com/document/doi/10.1515/mcma.2002.8.4.343/html?lang=en

    We save at each point the paths from the root node to each node we have particles in, how many particles we have
    in each such node, and num_particles * prob_mass of each such node. We do num_particles * prob_mass for floating
    point precision reasons, otherwise everything gets set to 0 after a relatively small number of intervals.

    :param key: key for jax PRNG.
    :param cubature: cubature formula
    :param num_particles: number of particles we send down the tree
    :param num_intervals: number of intervals in our partition

    :return: array of shape (num_intervals, num_particles) representing indices of lie polynomials that we pick. See
    monte carlo function for details.
    """
    # we pad to power of 2 for intermediate steps, which consist of constructing a binary tree between each node and its
    # children.
    probabilities = _pad_to_power_two(cubature[1])
    k = jnp.log2(len(probabilities)).astype(int).item()
    # we will scan across these arrays later, we need them to have fixed sizes.
    xi_array = jnp.concatenate([jnp.array([num_particles]), jnp.zeros(num_particles-1)])
    # mass does not represent a(m), as in the paper by Lyons and Crisan (LC09), but instead represents n * a(m). This
    # is done for floating point precision; without this, most masses are set to 0 after 10-15 intervals even
    # with degree 5, dimension 1 cubature, which has a relatively small number of elements.
    mass_array = jnp.concatenate([jnp.array([num_particles]), jnp.zeros(num_particles-1)])

    # one_full_step sends particles down from a_t(j) to a_t(j_1),  ... a_t(j_n), in the notation of LC09 (figure 2).

    def one_full_step(mass, xi, path, key, k, probabilities):
        lambdas = probabilities * mass
        # print below is to check if we lose mass during intermediate steps. We shouldn't be losing mass here, only in
        # the scanning phase. Left in for posterity.
        # jax.debug.print("{}", jnp.sum(lambdas) - mass)

        # intermediate_step_curr_masses are lists of masses of nodes to left and nodes to right. intermediate_step_prev_
        # masses are the same masses, except rolled forward and stacked and ravelled such that they match their
        # positions in the cubature measure, so n * a(j_k) is again on the k-th spot in the array.
        intermediate_step_curr_masses = _get_intermediate_massses(lambdas, k)
        intermediate_step_prev_masses = [jnp.expand_dims(mass, axis=0)] + [jnp.ravel(jnp.stack(x, axis=-1), order='C')
                                                                           for x in intermediate_step_curr_masses[:-1]]

        def intermediate_step(key, init_xi, init_mass, mass_1, mass_2):
            """
            Intermediate steps function, builds one level of the binary tree between nodes in the big tree and sends
            particles down.
            :param key: key for PRNG
            :param init_xi: number of particles at current node
            :param init_mass: mass of current node
            :param mass_1: mass of node to the left
            :param mass_2: mass of node to the right
            :return:
            """
            frac_init_mass, integ_init_mass = jnp.modf(init_mass)
            frac_mass_1, integ_mass_1 = jnp.modf(mass_1)
            frac_mass_2, integ_mass_2 = jnp.modf(mass_2)

            def true_fun(key, xi, i1, i2, f1, f2):
                # case: floor(a(m)) = floor(a(m_1)) + floor(a(m_2))
                eta = jax.random.choice(key, a=2, p=jnp.array([f2, f1]))
                xi_1 = i1 + (xi - i1 - i2) * eta
                xi_2 = i2 + (xi - i1 - i2) * (1 - eta)
                return xi_1, xi_2

            def false_fun(key, xi, i1, i2, f1, f2):
                # case: floor(a(m)) = floor(a(m_1)) + floor(a(m_2)) + 1
                eta = jax.random.choice(key, a=2, p=jnp.array([1 - f2, 1 - f1]))
                xi_1 = i1 + 1 + (xi - i1 - i2 - 2) * eta
                xi_2 = i2 + 1 + (xi - i1 - i2 - 2) * (1 - eta)
                return xi_1, xi_2

            partial_true = ft.partial(true_fun, xi=init_xi, i1=integ_mass_1, i2=integ_mass_2, f1=frac_mass_1,
                                      f2=frac_mass_2)
            partial_false = ft.partial(false_fun, xi=init_xi, i1=integ_mass_1, i2=integ_mass_2, f1=frac_mass_1,
                                       f2=frac_mass_2)

            return jax.lax.cond(integ_init_mass == integ_mass_1 + integ_mass_2, partial_true, partial_false, key)

        vmapped_intermediate_step = jax.vmap(intermediate_step, in_axes=(None, 0, 0, 0, 0))

        ans = jnp.expand_dims(xi, axis=0)
        for i in range(k):
            # get through all levels of the intermediate binary tree. Lax.scan could maybe also work here?
            arg1, arg2 = intermediate_step_prev_masses[i], intermediate_step_curr_masses[i]
            new = vmapped_intermediate_step(key, ans, arg1, *arg2)
            new = jnp.ravel(jnp.stack(new, axis=-1), order='C')
            ans = new

        # append the correct index for each node to which a point could have gotten sent to, i.e. each child of our
        # current node. Our previous stacking/ravelling ensures these are in the correct order.
        new_paths = jnp.stack([path] * 2**k, axis=0)
        new_paths = jnp.concatenate([new_paths, jnp.expand_dims(jnp.arange(2**k), axis=-1)], axis=-1)
        return ans, lambdas, new_paths

    partial_full_step = ft.partial(one_full_step, k=k, probabilities=probabilities)
    vmapped_full_step = jax.vmap(partial_full_step, in_axes=(0, 0, 0, 0))


    def body_fun(carry, x, full_step_func, num_particles):
        within_step_key_array = jax.random.split(x, num_particles)
        xis, masses, paths = carry
        # jax.debug.print('xi array {}', xis)
        # jax.debug.print('mass array {}', masses)
        # jax.debug.print('paths array {}', paths)
        # paths is array of shape (num_particles, len_paths)
        new_xis, new_masses, new_paths = full_step_func(masses, xis, paths, within_step_key_array)
        new_xis, new_masses = jnp.expand_dims(new_xis, axis=-1), jnp.expand_dims(new_masses, axis=-1)

        stacked = jnp.concatenate([new_xis, new_masses, new_paths], axis=-1)
        stacked = stacked.reshape(-1, stacked.shape[-1])
        stacked = jnp.concatenate([stacked, jnp.expand_dims(jnp.zeros(stacked.shape[-1]), axis=0)], axis=0)
        # below we save the index of each position where we have >0 particles. We only save those
        # paths/num_particles/masses.
        idx = jnp.where(stacked[..., 0] > 0, size=num_particles, fill_value=-1)

        new_stacked = stacked[idx[0], ...]
        new_xis = jnp.squeeze(new_stacked[..., 0])
        new_masses = jnp.squeeze(new_stacked[..., 1])
        new_paths = new_stacked[..., 3:]

        carry = (new_xis, new_masses, new_paths)

        return carry, jnp.array([0.])

    # we split the initial key once on the outside of the scan to get a separate key for each interval, and once inside
    # the scan to get a separate key for each particle.
    outside_step_key_array = jax.random.split(key, num_intervals)
    # pad the paths as jax needs fixed size arrays.
    init_paths = jnp.pad(jnp.expand_dims((-1) * jnp.ones(num_particles), axis=-1), pad_width = ((0, 0), (num_intervals, 0)))
    init = (xi_array, mass_array, init_paths)
    partial_body = ft.partial(body_fun, full_step_func=vmapped_full_step, num_particles=num_particles)

    final_values, _ = jax.lax.scan(partial_body, init, xs=outside_step_key_array, length=num_intervals)

    paths = final_values[-1]
    paths = paths[..., 1:]
    xis = final_values[0]
    paths = jnp.repeat(paths, xis.astype(int), axis=0)
    return jnp.swapaxes(paths, axis1=0, axis2=1).astype(int)


def _monte_carlo_sampling(key: jax.typing.ArrayLike,
                          cubature: tuple[list[fla.Elt], list[float]],
                          num_intervals: int,
                          num_particles: int) -> jax.typing.ArrayLike:
    """
    Monte Carlo sampling through the CDE tree
    :param key: key for pseudo-random number generator
    :param cubature: cubature formula
    :param num_particles: number of particles that trickle down the CDE tree
    :return: num_intervals x num_samples array of paths
    """
    _, weights = cubature
    m = len(weights)
    weights = jnp.array(weights)
    return jax.random.choice(key=key, a=m, shape=(num_intervals, num_particles), p=weights, replace=True)


def _get_rescaled_polys(cubature: tuple[list[fla.Elt], list[float]],
                        partition: list[float],
                        truncation_level: int,
                        dim: int,
                        signature_flag: Optional[bool] = False) -> jax.typing.ArrayLike:
    """
    :param cubature: cubature formula
    :param partition: Partition of the domain of the SDE
    :param truncation_level: truncation level
    :param dim: dimension of diffusion + drift
    :param signature_flag: Flag signifying whether we return signatures or log-signatures for Taylor increment
     method or log-ode method respectively.
    :return: returns (num_intervals x num_polys) shaped array where array[i][j] is
    the j-th lie polynomial, rescaled according to the i-th interval of the _partition, or
    exp(j-th polynomial) if signature_flag = True.
    """

    lie_poly_list, _ = cubature
    num_intervals = len(partition) - 1

    m = len(lie_poly_list)
    rescaled_polys = []

    for i in range(1, num_intervals + 1):
        t_1, t_0 = partition[i], partition[i - 1]
        s = t_1 - t_0
        lvl_polys = []

        for j in range(m):
            p = lie_poly_list[j]
            rescaled_sig = _rescale_elt(fla.exp(p), s, signature_flag)
            lvl_polys.append(rescaled_sig)

        rescaled_polys.append(lvl_polys)

    partial_ett = ft.partial(_elt_to_tuple, dim=dim, truncation_level=truncation_level)
    partial_ttj = ft.partial(_tuple_to_jax, dim=dim, truncation_level=truncation_level)

    stack_list = []

    for i in range(num_intervals):
        lvl_tpl_lie_poly_list = [partial_ett(lie_poly) for lie_poly in rescaled_polys[i]]
        lvl_lie_poly_array = jnp.stack([partial_ttj(tpl) for tpl in lvl_tpl_lie_poly_list], axis=0)
        stack_list.append(lvl_lie_poly_array)

    return jnp.stack(stack_list, axis=0)


def _get_sampled_polys(sampled_points: jax.typing.ArrayLike,
                       rescaled_polys: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    """
    Helper function; Vectorizes the process of picking the rescaled polys according to the sampled points.
    :return: jax array of shape (num_intervals, num_sampled_points, ...), where array[i, j, ...] is the
    polynomial corresponding to the j-th picked point, rescaled accordingly for the i-th interval.
    """

    def pick(point, array):
        return array[point]

    pick_path = jax.vmap(pick, in_axes=(0, 0), out_axes=(0))
    pick_all = jax.vmap(pick_path, in_axes=(1, None), out_axes=(1))
    return pick_all(sampled_points, rescaled_polys)


def make_data(key: jax.typing.ArrayLike,
                          cubature: tuple[list[fla.Elt], list[float]],
                          partition: list[float],
                          num_particles: int,
                          contr_dim: int,
                          truncation_level: int,
                          sampling_method: str,
                          signature_flag: Optional[bool] = False) -> jax.typing.ArrayLike:
    """
    Generate the data via Monte Carlo sampling through the CDE tree. Gets sampled points, picks polys according to
    the sampled points.
    :param key: key for pseudo random number generator.
    :param cubature: cubature formula.
    :param partition: partition used.
    :param num_particles: number of particles to send down the tree.
    :param dim: dimension of the control process. (drift diffusion combined)
    :param truncation_level: truncation_level
    :param signature_flag: flag signifying whether we return signatures or log-signatures in our data.
    :return: jax array of shape (num_intervals, num_sampled_points, ...), where array[i, j, ...] is the
    polynomial corresponding to the j-th picked point, rescaled accordingly for the i-th interval.
    """

    dim = contr_dim + 1
    num_intervals = len(partition) - 1
    if sampling_method == 'Monte Carlo':
        sampled_points = _monte_carlo_sampling(key, cubature, num_intervals, num_particles)
    elif sampling_method == 'TBBA':
        sampled_points = _TBBA_sampling(key, cubature, num_intervals, num_particles)
    rescaled_polys = _get_rescaled_polys(cubature, partition, truncation_level, dim, signature_flag)
    sampled_polys = _get_sampled_polys(sampled_points, rescaled_polys)

    return sampled_polys


def _taylor_expansion_vector_field(derivatives: list[callable],
                                   lie_poly: jax.typing.ArrayLike) -> callable:
    """
    Helper function; Computes the taylor expansion of the SDE vector field.
    :param lie_poly: lie polynomial on which we evaluate the taylor expansion.
    :return: callable object, vf(x) is the taylor expansion of the SDE vector field evaluated at x.
    """
    truncation_level = len(derivatives)
    vf = lambda x: 0

    for i in range(truncation_level):
        # get vf derivative and appropriate level of lie polynomial
        derivative = derivatives[i]
        index = [i] + [0] * (truncation_level - i - 1)
        poly_level = lie_poly[tuple(index)]

        # combine into single callable
        def vf_level(arr, derivative, poly_level, i=i):
            return jnp.tensordot(derivative(arr), poly_level, axes=i + 1)

        # we use ft.partial here as it creates a new object, rather than referencing to the vf_level function.
        vf = _add_callable(vf, ft.partial(vf_level, derivative=derivative, poly_level=poly_level))

    return vf


def _rescale_dense_solutions(partition: list[float],
                             solution_paths: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    """
    Concatenates dense solutions with the appropriate indices.
    Note: solution_paths returned by both dense taylor and dense log-ode will be of shape (num_intervals x num_samples)
    :param partition: partition
    :param solution_paths: dense solution paths.
    :return: array of shape [num_particles, num_save_points * num_intervals, 2], where:
             array[i, j, 0] contains the values of the dense solution, and:
             array[i, j, 1] contains the times corresponding to those values.
    """

    num_particles = solution_paths.shape[1]
    intervals = jnp.array(partition[1:])

    # expand the new indices list to match the points in the dense solutions, and combine
    new_indices = jnp.expand_dims(jnp.stack([intervals] * num_particles, axis=0), axis=-1)
    paths = jnp.swapaxes(solution_paths, axis1=0, axis2=1)
    combined = jnp.concatenate([paths, new_indices], axis=-1)

    return combined


def log_ode_method_sampling(combined_vf: callable,
                            initial_condition: jax.typing.ArrayLike,
                            sampled_lie_polys: jax.typing.ArrayLike,
                            ode_solver: diffrax.AbstractSolver,
                            step_size_controller: diffrax.AbstractStepSizeController,
                            truncation_level: int = 5,
                            dense: Optional[bool] = False,
                            verbose: Optional[bool] = True
                            ) -> tuple[jax.typing.ArrayLike, dt.timedelta]:
    """
    Main function; Implementation of the cubature method with sampling through the CDE tree, using the log-ODE method
    to transform each CDE into an ODE.

    :param combined_vf: Python function that represents the drift and diffusion vector fields. The function should take
    in an array of shape (sol_dim,) or (sol_dim, 1) and produce an array of shape (sol_dim, contr_dim + 1).

    :param initial_condition: Initial condition of the SDE. Should be an array of shape (sol_dim,) or (sol_dim, 1).

    :param sampled_lie_polys: An array of shape (num_partition_intervals, num_samples, ...) representing the paths in
    the CDE tree that we will go down. Each path is represented by a sequence of Lie Polynomials, rescaled according to
    the interval on which we are working. Note, the leading dimension must be the interval, as we will lax.scan across
    this array.

    :param ode_solver: The ODE solver we use to implement the log-ODE Method. For ODE solving, we use the Diffrax module
    by Patrick Kidger, see https://docs.kidger.site/diffrax/ for documentation.

    :param step_size_controller: Step size controller for ODEs. See https://docs.kidger.site/diffrax/api/stepsize_controller/
    for details on how ODE solvers and step size controllers interact.

    :param truncation_level: Level up to which we compute the taylor expansion of the vector field. As there is no
    reason to compute the taylor expansion at any level above the degree of the cubature, we set this as default to 5.

    :param dense: flag, signifying whether we are interested in solutions or solution paths. Set as default to False
    for solutions.

    :param verbose: flag for printing

    :return: array of shape (num_intervals, num_samples) representing the approximated solutions evaluated at all the
    points of the partition if dense=True, otherwise an array of shape (num_samples,) representing the approximated
    solutions evaluated at the final time, as well as the time it took to perform the approximations.
    """

    vf_sig = isp.signature(combined_vf)
    time_dep_flag = 't' in vf_sig.parameters.keys()

    num_particles = sampled_lie_polys.shape[1]
    initial_vector = jnp.stack([initial_condition] * num_particles, axis=0)
    derivatives = get_vf_derivatives(combined_vf, truncation_level)

    # we define here the single step solver, which we then vectorise and scan across our sampled lie polynomials. Note,
    # this function will automatically be jit-compiled within lax.scan, so we want to make it pure, hence all the args
    # that get set using ft.partial. See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html for more
    # information on lax.scan

    def solve_step(initial_condition: jax.typing.ArrayLike,
                   sampled_poly: jax.typing.ArrayLike,
                   ode_solver: diffrax.AbstractSolver,
                   step_size_controller: diffrax.AbstractStepSizeController):

        vf = _taylor_expansion_vector_field(derivatives, sampled_poly)
        ode_term = diffrax.ODETerm(lambda t, y, args: vf(y))
        save_at = diffrax.SaveAt(ts=[0., 1.])

        sol = diffrax.diffeqsolve(ode_term,
                                  ode_solver,
                                  t0=0., t1=1.,
                                  dt0=0.02,
                                  y0=initial_condition,
                                  saveat=save_at,
                                  stepsize_controller=step_size_controller)

        # note, we return sol.ys[-1] twice as lax.scan requires its body function to be of type (c, x) -> (c, y).
        # Additionally, lax.scan will automatically stack all of its y outputs, to produce our path solutions.
        return sol.ys[-1], sol.ys[-1]

    partial = ft.partial(solve_step, ode_solver=ode_solver, step_size_controller=step_size_controller)
    vmapped = jax.vmap(partial, in_axes=(0, 0), out_axes=(0, 0))

    if verbose:
        print('starting solve')

    time_0 = dt.datetime.now()
    ans_tuple = jax.lax.scan(vmapped, init=initial_vector, xs=sampled_lie_polys)
    time_1 = dt.datetime.now()

    # note, ans_tuple[0] contains the solutions and ans_tuple[1] contains the solution paths

    if verbose:
        print('solve ended, time elapsed: ', time_1 - time_0)

    return (ans_tuple[1], time_1 - time_0) if dense else (ans_tuple[0], time_1 - time_0)


def log_ode_method_full_tree(combined_vf: callable,
                             initial_condition: jax.typing.ArrayLike,
                             cubature: tuple[list[fla.Elt], list[float]],
                             partition: list[float],
                             p: int,
                             q: int,
                             ode_solver: diffrax.AbstractSolver,
                             step_size_controller: diffrax.AbstractStepSizeController,
                             payoffs: list[callable],
                             truncation_level: int = 5,
                             dense: Optional[bool] = False,
                             verbose: Optional[bool] = True):
    """
    Main function; Computes the full CDE tree of the cubature method using the log-ODE method to estimate the solutions.
    We do p intervals in parallel, fully, then q more intervals sequentially, starting from each final value in the
    parallel intervals. Note, for p and q large enough, we cannot return the full paths or even the final solutions, as
    that would require O(m ** (p + q)) memory, where m is the number of cubature points. For this reason, we do the prep
    work for the points/paths within each computation of the final q levels and return the required expected value only.

    :param combined_vf: Python function that represents the drift and diffusion vector fields. The function should take
    in an array of shape (sol_dim,) or (sol_dim, 1) and produce an array of shape (sol_dim, contr_dim + 1).

    :param initial_condition: Initial condition of the SDE. Should be an array of shape (sol_dim,) or (sol_dim, 1).

    :param cubature: Cubature formula.

    :param partition: Partition of the interval of the SDE. Should be a list of floats of length (num_intervals + 1),
    i.e. including 0 and T.

    :param p: Number of intervals to compute in parallel, fully.

    :param q: Number of additional intervals to compute sequentially.

    :param payoff: Function/Functional to be evaluated on the points/paths respectively.

    :param truncation_level: Level up to which we compute the taylor expansion of the vector field.

    :param dense: flag, specifying whether we consider solutions or solution paths. dense=True corresponds to solution
    paths.

    :param verbose: flag for printing.

    :return: approximation of expected value of the function/functional of the points/paths.
    """

    dim = combined_vf(initial_condition).shape[1]
    data = _get_rescaled_polys(cubature, partition, truncation_level, dim, signature_flag=False)
    num_intervals = len(partition) - 1
    m = len(cubature[0])
    nr_functions = len(payoffs)

    def payoff(value, list):
        # note, value may be either a point or a path
        return jnp.stack([c(value) for c in list])

    partial_payoff = ft.partial(payoff, list=payoffs)
    vmapped_payoff = jax.jit(jax.vmap(partial_payoff, in_axes=(0)))

    assert p + q == num_intervals, 'all intervals must be computed either in parallel or sequentially'

    # get starting data
    # note, we expand dimensions below as we will be using the solver in its double vmapped state from the beginning,
    # which requires an extra 'batch' dimension.

    points = jnp.expand_dims(initial_condition, axis=0)
    probabilities = jnp.expand_dims(jnp.array([1.]), axis=0)
    prob_array = jnp.array(cubature[1])
    derivatives = get_vf_derivatives(combined_vf, truncation_level)

    # define single step solver
    def solve_step(initial_condition: jax.typing.ArrayLike,
                   lie_poly: jax.typing.ArrayLike,
                   initial_prob: jax.typing.ArrayLike,
                   lmbd: jax.typing.ArrayLike,
                   derivatives: list[callable],
                   ode_solver: diffrax.AbstractSolver,
                   step_size_controller: diffrax.AbstractStepSizeController):

        vf = _taylor_expansion_vector_field(derivatives, lie_poly)
        ode_term = diffrax.ODETerm(lambda t, y, args: vf(y))
        save_at = diffrax.SaveAt(ts=[0., 1.])

        sol = diffrax.diffeqsolve(ode_term,
                                  ode_solver,
                                  t0=0., t1=1.,
                                  dt0=0.02,
                                  y0=initial_condition,
                                  saveat=save_at,
                                  stepsize_controller=step_size_controller)

        return sol.ys[-1], initial_prob * lmbd

    # note, we no longer use lax.scan for everything, so we need to explicitly jit the function.
    partial_solve_step = jax.jit(ft.partial(solve_step, derivatives=derivatives, ode_solver=ode_solver, step_size_controller=step_size_controller))
    # vectorise the single step solver across lie polynomials and probabilities
    solve_node_children = jax.vmap(partial_solve_step, in_axes=(None, 0, None, 0), out_axes=(0, 0))
    # vectorise the solver again across initial conditions and probabilities
    solve_level = jax.vmap(solve_node_children, in_axes=(0, None, 0, None), out_axes=(0, 0))

    # function to compute the first p levels fully, in parallel.
    def first_levels(p: int,
                     points: jax.typing.ArrayLike,
                     probabilities: jax.typing.ArrayLike,
                     dense: bool):

        if dense:
            path_list = []

        for i in range(p):
            if dense:
                curr_level_paths = jnp.repeat(points, m ** (p - i), axis=0)
                path_list.append(curr_level_paths)

            level_data = data[i]
            points, probabilities = solve_level(points, level_data, probabilities, prob_array)

            points = jnp.reshape(points, newshape=(-1, points.shape[-1]))
            probabilities = jnp.reshape(probabilities, newshape=(-1, probabilities.shape[-1]))

            if verbose:
                print('parallel level ' + str(i + 1) + ' done')

        if dense:
            path_list.append(points)
            parallel_paths = jnp.swapaxes(jnp.stack(path_list, axis=0), axis1=0, axis2=1)
            # parallel paths has dimension (m ** p, p, sol_dim) now. Note, we swap axes as lax.scan can only scan across
            # the leading dimension, and we will need to scan for the remaining q levels.

        return (points, probabilities) if not dense else (parallel_paths, probabilities)

    # get first p levels. Note, values will either be points or paths, depending on the parameter dense. dense=True
    # corresponds to paths.
    values, probabilities = first_levels(p=p, points=points, probabilities=probabilities, dense=dense)

    # create the body function for the scan we will be doing. As this will be jitted, we want to make it a pure
    # function, hence all the arguments that then get set through ft.partial.
    def final_levels(carry, x, m, partition, remaining, solver, lie_poly_data, lmbd_array, _payoff, dense):
        value, probability = x
        # note value will either be a point or a path, depending on the parameter dense. dense=True corresponds to
        # a path.
        num_levels = lie_poly_data.shape[0]
        completed = num_levels - remaining
        points = jnp.expand_dims(value[-1], axis=0) if dense else jnp.expand_dims(value, axis=0)
        probabilities = jnp.expand_dims(probability, axis=0)

        if dense:
            plist = []

        for i in range(remaining):
            level_data = lie_poly_data[completed + i]
            points, probabilities = solver(points, level_data, probabilities, lmbd_array)

            points = jnp.reshape(points, newshape=(-1, points.shape[-1]))
            probabilities = jnp.reshape(probabilities, newshape=(-1, probabilities.shape[-1]))

            if dense:
                curr_level_paths = jnp.repeat(points, m ** (remaining - i - 1), axis=0)
                plist.append(curr_level_paths)

        if dense:
            # we prep each path to vmap the payoff across them. To do this, we must concatenate the path prefix obtained
            # from the parallel phase above with the suffixes obtained in this phase.
            if len(plist) > 0:
                fin_paths = jnp.swapaxes(jnp.stack(plist, axis=0), axis1=0, axis2=1)
                full_fin_paths = jnp.concatenate([jnp.stack([value] * m ** remaining, axis=0), fin_paths], axis=1)
            else:
                full_fin_paths = jnp.expand_dims(value, axis=0)
            # note, full_fin_paths will now have shape (m ** q, p + q + 1, sol_dim)
            # we extract the price path (if needed, concatenate with time indices here, uncomment the code)
            idx = jnp.stack([jnp.array(partition)] * m ** p, axis=0)
            price_paths = full_fin_paths[..., 0]
            # price_paths = jnp.concatenate([price_paths, idx], axis=1)
            carry += jnp.tensordot(_payoff(price_paths), probabilities, axes=[[0], [0]])

        else:
            # get price points
            points = points[..., 0]
            carry += jnp.tensordot(_payoff(points), probabilities, axes=[[0], [0]])

        # note, jnp.zeros(1) is a placeholder, as lax.scan expects its body function to be (c, x) -> (c, y)
        return carry, jnp.zeros(1)

    partial_final_levels = ft.partial(final_levels,
                                      m=m,
                                      partition=partition,
                                      remaining=q,
                                      solver=solve_level,
                                      lie_poly_data=data,
                                      lmbd_array=prob_array,
                                      _payoff=vmapped_payoff,
                                      dense=dense)

    # get last q levels
    final_ev, _ = jax.lax.scan(partial_final_levels, xs=(values, probabilities), init=jnp.zeros((nr_functions, 1)))

    return final_ev


def taylor_increment_sampling(combined_vf: callable,
                              initial_condition: jax.typing.ArrayLike,
                              sampled_lie_polys: jax.typing.ArrayLike,
                              truncation_level: int = 5,
                              dense: Optional[bool] = False,
                              verbose: Optional[bool] = True
                              ) -> tuple[jax.typing.ArrayLike, dt.timedelta]:
    """
    Main function; Implementation of the cubature method with sampling through the CDE tree, using a simple one-step
    taylor expansion to approximate the solution to each CDE. While this loses some desirable properties of the log-ODE
    method, it presents a significant speed improvement.

    :param combined_vf: Python function that represents the drift and diffusion vector fields. The function should take
    in an array of shape (sol_dim,) or (sol_dim, 1) and produce an array of shape (sol_dim, contr_dim + 1).

    :param initial_condition: Initial condition of the SDE. Should be an array of shape (sol_dim,) or (sol_dim, 1).

    :param sampled_lie_polys: An array of shape (num_partition_intervals, num_samples, ...) representing the paths in
    the CDE tree that we will go down. Each path is represented by a sequence of Lie Polynomials, rescaled according to
    the interval on which we are working. Note, the leading dimension must be the interval, as we will lax.scan across
    this array.

    :param truncation_level: Level up to which we compute the taylor expansion of the vector field. As there is no
    reason to compute the taylor expansion at any level above the degree of the cubature, we set this as default to 5.

    :param dense: flag, signifying whether we are interested in solutions or solution paths. Set as default to False
    for solutions.

    :param verbose: flag for printing

    :return: array of shape (num_intervals, num_samples) representing the approximated solutions evaluated at all the
    points of the partition if dense=True, otherwise an array of shape (num_samples,) representing the approximated
    solutions evaluated at the final time, as well as the time it took to perform the approximations.
    """

    num_particles = sampled_lie_polys.shape[1]
    initial_vector = jnp.array([initial_condition] * num_particles)
    derivatives = get_vf_derivatives(combined_vf, truncation_level)

    def solve_step(initial_condition, sampled_poly):
        vf = _taylor_expansion_vector_field(derivatives, sampled_poly)
        # note, we return the approximated value twice as lax.scan requires its body function to be (c, x) -> (c, y).
        # Additionally, lax.scan will automatically stack all of its y outputs, to produce our path solutions.
        return initial_condition + vf(initial_condition), initial_condition + vf(initial_condition)

    vmapped = jax.vmap(solve_step, in_axes=(0, 0), out_axes=(0, 0))

    if verbose:
        print('starting solve')

    time_0 = dt.datetime.now()
    ans_tuple = jax.lax.scan(vmapped, init=initial_vector, xs=sampled_lie_polys)
    time_1 = dt.datetime.now()

    # note, ans_tuple[0] contains the solutions and ans_tuple[1] contains the solution paths

    if verbose:
        print('solve ended, time elapsed: ', time_1 - time_0)

    return (ans_tuple[1], time_1 - time_0) if dense else (ans_tuple[0], time_1 - time_0)


def taylor_increment_full_tree(combined_vf: callable,
                               initial_condition: jax.typing.ArrayLike,
                               cubature: tuple[list[fla.Elt], list[float]],
                               partition: list[float],
                               p: int,
                               q: int,
                               payoffs: list[callable],
                               truncation_level: int = 5,
                               dense: Optional[bool] = False,
                               verbose: Optional[bool] = True
                               ) -> jax.typing.ArrayLike:
    """
    Main function; Computes the full CDE tree of the cubature method with taylor increments to estimate the solutions.
    We do p intervals in parallel, fully, then q more intervals sequentially, starting from each final value in the
    parallel intervals. Note, for p and q large enough, we cannot return the full paths or even the final solutions, as
    that would require O(m ** (p + q)) memory, where m is the number of cubature points. For this reason, we do the prep
    work for the points/paths within each computation of the final q levels and return the required expected value only.

    :param combined_vf: Python function that represents the drift and diffusion vector fields. The function should take
    in an array of shape (sol_dim,) or (sol_dim, 1) and produce an array of shape (sol_dim, contr_dim + 1).

    :param initial_condition: Initial condition of the SDE. Should be an array of shape (sol_dim,) or (sol_dim, 1).

    :param cubature: Cubature formula.

    :param partition: Partition of the interval of the SDE. Should be a list of floats of length (num_intervals + 1),
    i.e. including 0 and T.

    :param p: Number of intervals to compute in parallel, fully.

    :param q: Number of additional intervals to compute sequentially.

    :param payoff: Function/Functional to be evaluated on the points/paths respectively.

    :param truncation_level: Level up to which we compute the taylor expansion of the vector field.

    :param dense: flag, specifying whether we consider solutions or solution paths. dense=True corresponds to solution
    paths.

    :param verbose: flag for printing.

    :return: approximation of expected value of the function/functional of the points/paths.
    """

    dim = combined_vf(initial_condition).shape[1]
    data = _get_rescaled_polys(cubature, partition, truncation_level, dim, signature_flag=True)
    num_intervals = len(partition) - 1
    m = len(cubature[0])
    nr_functions = len(payoffs)

    def payoff(value, list):
        # note, value may be either a point or a path
        return jnp.stack([c(value) for c in list])

    partial_payoff = ft.partial(payoff, list=payoffs)
    vmapped_payoff = jax.jit(jax.vmap(partial_payoff, in_axes=(0)))

    assert p + q == num_intervals, 'all intervals must be computed either in parallel or sequentially'

    # get starting data
    # note, we expand dimensions below as we will be using the solver in its double vmapped state from the beginning,
    # which requires an extra 'batch' dimension.

    points = jnp.expand_dims(initial_condition, axis=0)
    probabilities = jnp.expand_dims(jnp.array([1.]), axis=0)
    prob_array = jnp.array(cubature[1])
    derivatives = get_vf_derivatives(combined_vf, truncation_level)

    # define single step solver
    # note, we no longer use lax.scan for everything, so we need to explicitly jit the function.
    @jax.jit
    def solve_step(initial_condition: jax.typing.ArrayLike,
                   lie_poly: jax.typing.ArrayLike,
                   initial_prob: jax.typing.ArrayLike,
                   lmbd: jax.typing.ArrayLike):

        vf = _taylor_expansion_vector_field(derivatives, lie_poly)
        return initial_condition + vf(initial_condition), initial_prob * lmbd

    # vectorise the single step solver across lie polynomials and probabilities
    solve_node_children = jax.vmap(solve_step, in_axes=(None, 0, None, 0), out_axes=(0, 0))
    # vectorise the solver again across initial conditions and probabilities
    solve_level = jax.vmap(solve_node_children, in_axes=(0, None, 0, None), out_axes=(0, 0))

    # function to compute the first p levels fully, in parallel.
    def first_levels(p: int,
                     points: jax.typing.ArrayLike,
                     probabilities:jax.typing.ArrayLike,
                     dense: bool):
        if dense:
            path_list = []

        for i in range(p):
            if dense:
                curr_level_paths = jnp.repeat(points, m ** (p - i), axis=0)
                path_list.append(curr_level_paths)

            level_data = data[i]
            points, probabilities = solve_level(points, level_data, probabilities, prob_array)

            points = jnp.reshape(points, newshape=(-1, points.shape[-1]))
            probabilities = jnp.reshape(probabilities, newshape=(-1, probabilities.shape[-1]))

            if verbose:
                print('parallel level ' + str(i + 1) + ' done')

        if dense:
            path_list.append(points)
            parallel_paths = jnp.swapaxes(jnp.stack(path_list, axis=0), axis1=0, axis2=1)
            # parallel paths has dimension (m ** p, p, sol_dim) now. Note, we swap axes as lax.scan can only scan across
            # the leading dimension, and we will need to scan for the remaining q levels.

        return (points, probabilities) if not dense else (parallel_paths, probabilities)

    # get first p levels. Note, values will either be points or paths, depending on the parameter dense. dense=True
    # corresponds to paths.
    values, probabilities = first_levels(p=p, points=points, probabilities=probabilities, dense=dense)

    # create the body function for the scan we will be doing. As this will be jitted, we want to make it a pure
    # function, hence all the arguments that then get set through ft.partial.
    def final_levels(carry, x, m, partition, remaining, solver, lie_poly_data, lmbd_array, _payoff, dense):
        value, probability = x
        # note value will either be a point or a path, depending on the parameter dense. dense=True corresponds to
        # a path.
        num_levels = lie_poly_data.shape[0]
        completed = num_levels - remaining
        points = jnp.expand_dims(value[-1], axis=0) if dense else jnp.expand_dims(value, axis=0)
        probabilities = jnp.expand_dims(probability, axis=0)

        if dense:
            plist = []

        for i in range(remaining):
            level_data = lie_poly_data[completed + i]
            points, probabilities = solver(points, level_data, probabilities, lmbd_array)

            points = jnp.reshape(points, newshape=(-1, points.shape[-1]))
            probabilities = jnp.reshape(probabilities, newshape=(-1, probabilities.shape[-1]))

            if dense:
                curr_level_paths = jnp.repeat(points, m ** (remaining - i - 1), axis=0)
                plist.append(curr_level_paths)

        if dense:
            # we prep each path to vmap the payoff across them. To do this, we must concatenate the path prefix obtained
            # from the parallel phase above with the suffixes obtained in this phase.
            if len(plist) > 0:
                fin_paths = jnp.swapaxes(jnp.stack(plist, axis=0), axis1=0, axis2=1)
                full_fin_paths = jnp.concatenate([jnp.stack([value] * m ** remaining, axis=0), fin_paths], axis=1)
            else:
                full_fin_paths = jnp.expand_dims(value, axis=0)

            # note, full_fin_paths will now have shape (m ** q, p + q + 1, sol_dim)
            # we extract the price path (if needed, concatenate with time indices here, uncomment the code)
            idx = jnp.stack([jnp.array(partition)] * m ** p, axis=0)
            price_paths = full_fin_paths[..., 0]
            # price_paths = jnp.concatenate([price_paths, idx], axis=1)
            carry += jnp.tensordot(_payoff(price_paths), probabilities, axes=[[0], [0]])

        else:
            # get price points
            points = points[..., 0]
            carry += jnp.tensordot(_payoff(points), probabilities, axes=[[0], [0]])

        # note, jnp.zeros(1) is a placeholder, as lax.scan expects its body function to be (c, x) -> (c, y)
        return carry, jnp.zeros(1)

    partial_final_levels = ft.partial(final_levels,
                                      m=m,
                                      partition=partition,
                                      remaining=q,
                                      solver=solve_level,
                                      lie_poly_data=data,
                                      lmbd_array=prob_array,
                                      _payoff=vmapped_payoff,
                                      dense=dense)

    # get last q levels
    final_ev, _ = jax.lax.scan(partial_final_levels, xs=(values, probabilities), init=jnp.zeros((nr_functions, 1)))

    return final_ev


if __name__ == '__main__':

    nr_intervals = 100
    gamma = 1
    nr_steps = 100
    level = 3

    seed = 1890
    key = jax.random.PRNGKey(seed)

    mu = 0.05  # drift of asset
    kappa = 1.7  # amplitude of mean reversion component of vol process
    theta = 0.12  # vol mean
    rho = -0.7  # correlation of driving brownian motions
    xi = 0.1  # volvol
    initial_price = 150.  # initial (observed) asset price
    initial_vol = 0.1  # initial (observed) volatility
    T = 0.164  # Expiry date of options (two month expiry chosen, can change)

    cd_1 = lambda x: mu * x[0] - 0.5 * x[1] * x[0] - 0.25 * xi * rho * x[0]
    cd_2 = lambda x: kappa * (theta - x[1]) - 0.25 * xi ** 2

    cd = lambda x: jnp.stack([cd_1(x), cd_2(x)]).reshape(-1, 1)

    cdd_11 = lambda x: jnp.sqrt(x[1]) * x[0]
    cdd_12 = lambda x: 0.
    cdd_21 = lambda x: xi * rho * jnp.sqrt(x[1])
    cdd_22 = lambda x: xi * jnp.sqrt(x[1] * (1.0 - rho ** 2))

    cdd = lambda x: jnp.stack([cdd_11(x), cdd_12(x), cdd_21(x), cdd_22(x)]).reshape(2, 2)

    comb = lambda x: jnp.hstack([cd(x), cdd(x)])

    print(comb(jnp.array([150., 0.1])))

    contr_dim = 2
    dim = contr_dim + 1
    cubature = wiener_cubature(dim - 1)
    # cubature = dim_2_wiener_cubature(7)

    print('making data')
    time_0 = dt.datetime.now()

    partition = get_partition((0., 0.164), nr_intervals, gamma)

    data = make_data(key, cubature, partition, nr_steps, contr_dim, level, 'TBBA', False)
    print('DONE, time elapsed: ', dt.datetime.now() - time_0)

    ode_solver_2 = diffrax.Dopri5()
    ode_solver_1 = diffrax.Euler()
    ssc_1 = diffrax.ConstantStepSize()
    ssc_2 = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    ans_3 = log_ode_method_sampling(comb, jnp.array([150., 0.1]), cubature, partition, ssc_2, dense=True)

    print(ans_3)




