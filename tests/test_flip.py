import torch

from ser.transforms import flip


def test_flip():
    starting_array = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    expected_array = torch.tensor(
        [
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1],
        ]
    )

    do_flip = flip()
    torch.equal(do_flip(starting_array),expected_array)
