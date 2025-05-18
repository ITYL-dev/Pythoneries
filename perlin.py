import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a, b, t):
    return a + t * (b - a)


def perlin(grid_shape: tuple = (10, 10), discretization: tuple = (10, 10)):
    assert len(grid_shape) == len(discretization)
    ndim = len(grid_shape)
    assert ndim == 1 or ndim == 2

    gradient_grid = (
        2 * np.random.rand(*[grid_shape[d] + 1 for d in range(ndim)], ndim) - 1
    )

    noise_grid = np.zeros([grid_shape[d] * discretization[d] + 1 for d in range(ndim)])
    grid = np.stack(
        np.meshgrid(
            *[
                np.linspace(0, grid_shape[d] - 1, grid_shape[d] * discretization[d] + 1)
                for d in range(ndim)
            ],
            indexing="ij"
        ),
        axis=-1,
    )

    if ndim == 1:
        grid.squeeze()
        gradient_grid.squeeze()
        for i in tqdm(range(grid_shape[0])):
            corners = np.zeros((2, 1))
            corners[0] = grid[i * discretization[0]]
            corners[1] = grid[(i + 1) * discretization[0]]
            for m in range(discretization[0]):
                point = grid[i * discretization[0] + m]

                dots = np.zeros(2)
                dots[0] = np.dot(point - corners[0], gradient_grid[i])
                dots[1] = np.dot(point - corners[1], gradient_grid[i + 1])

                local_point = (point - corners[0])[0]
                faded_point = fade(local_point)
                noise_grid[i * discretization[0] + m] = lerp(
                    dots[0], dots[1], faded_point
                )

    elif ndim == 2:
        # gradient_grid = gradient_grid / np.expand_dims(
        #     np.sqrt(np.square(gradient_grid).sum(axis=-1)), -1
        # )
        for i in tqdm(range(grid_shape[0])):
            for j in range(grid_shape[1]):
                corners = np.zeros((4, 2))
                corners[0] = grid[i * discretization[0], j * discretization[1]]
                corners[1] = grid[(i + 1) * discretization[0], j * discretization[1]]
                corners[2] = grid[i * discretization[0], (j + 1) * discretization[1]]
                corners[3] = grid[
                    (i + 1) * discretization[0], (j + 1) * discretization[1]
                ]
                for m in range(discretization[0]):
                    for n in range(discretization[1]):
                        point = grid[
                            i * discretization[0] + m, j * discretization[1] + n
                        ]

                        dots = np.zeros(4)
                        dots[0] = np.dot(point - corners[0], gradient_grid[i, j])
                        dots[1] = np.dot(point - corners[1], gradient_grid[i + 1, j])
                        dots[2] = np.dot(point - corners[2], gradient_grid[i, j + 1])
                        dots[3] = np.dot(
                            point - corners[3], gradient_grid[i + 1, j + 1]
                        )

                        local_point = point - corners[0]
                        faded_point = fade(local_point)

                        interp_i_1 = lerp(dots[0], dots[1], faded_point[0])
                        interp_i_2 = lerp(dots[2], dots[3], faded_point[0])
                        noise_grid[
                            i * discretization[0] + m, j * discretization[1] + n
                        ] = lerp(interp_i_1, interp_i_2, faded_point[1])

    return grid, noise_grid


if __name__ == "__main__":

    grid1, noise1 = perlin((20,), (1000,))
    grid2, noise2 = perlin((10, 10), (20, 20))

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(grid1, noise1)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(grid2[..., 0], grid2[..., 1], noise2)
    ax.set_aspect("equal")
    ax = fig.add_subplot(1, 3, 3)
    imshow = ax.imshow(noise2)
    plt.colorbar(imshow)
    plt.show()

    noise1 = np.abs(noise1)
    noise2 = np.abs(noise2)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(grid1, noise1)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(grid2[..., 0], grid2[..., 1], noise2)
    ax.set_aspect("equal")
    ax = fig.add_subplot(1, 3, 3)
    imshow = ax.imshow(noise2)
    plt.colorbar(imshow)
    plt.show()

    noise1 = 1 - noise1
    noise2 = 1 - noise2

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(grid1, noise1)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(grid2[..., 0], grid2[..., 1], noise2)
    ax.set_aspect("equal")
    ax = fig.add_subplot(1, 3, 3)
    imshow = ax.imshow(noise2)
    plt.colorbar(imshow)
    plt.show()