import numpy as np
from ostrich.mathutils import cartesian_prod


def describe_cartesian_prod():

    def it_should_give_the_same_order_as_flattening():
        """
        The order of array.flatten's output should always align with the ordering given
        by cartesian product, when the product is interpreted as coordinates for the
        values of the flattened array
        """
        xs = np.linspace(1, 10, 10)
        ys = np.linspace(1, 10, 10)
        zs = np.linspace(1, 10, 10)

        fs = np.random.rand(10, 10, 10)

        flat_func = fs.flatten()

        coord_inds = []
        for x in (xs, ys, zs):
            coord_inds.append(np.linspace(0, x.size - 1, x.size, dtype=int))

        coord_inds = tuple(cartesian_prod(*coord_inds).T)
        what_flat_func_should_be = fs[coord_inds]

        assert np.all(flat_func == what_flat_func_should_be)
