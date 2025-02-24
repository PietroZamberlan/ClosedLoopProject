import os

from pystim.euler import generate as generate_euler
# from pystim.dg import generate as generate_gratings


def prepare(args):

    _ = args

    base_path = '/tmp/pystim/experiments/latest'

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    # # Prepare checkerboard.
    # checkerboard_path = os.path.join(base_path, 'checkerboard')
    # checkerboard_args = {'path': checkerboard_path}
    # generate_checkerboard(checkerboard_args)

    # Prepare Euler full-field.
    euler_path = os.path.join(base_path, 'euler_full_field')
    euler_args = {'path': euler_path}
    generate_euler(euler_args)

    # TODO uncomment the following lines.
    # # Prepare drifting gratings.
    # gratings_path = os.path.join(base_path, 'drifting_gratings')
    # gratings_args = {'path': gratings_path}
    # generate_gratings(gratings_args)

    # Prepare flashed images.
    flashes_path = os.path.join(base_path, 'flashed_images')
    flashed_args = {'path': flashes_path}
    generate_flashed_images(flashed_args)

    return
