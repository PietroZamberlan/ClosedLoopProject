import argparse

from pystim.test import generate as generate_test
from pystim.square import generate as generate_square
from pystim.euler import generate as generate_euler
from pystim.dg import generate as generate_dg
from pystim.fi import generate as generate_fi
from pystim.fi_comp import generate as generate_fi_comp
from pystim.fi_merge import generate as generate_fi_merge
from pystim.fipwc import generate as generate_fipwc
from pystim.fipwfc import generate as generate_fipwfc
from pystim.fipwrc import generate as generate_fipwrc

from pystim.experiments.latest import prepare as prepare_latest

from pystim.utils import list_stimuli, configure, initialize, reinitialize


def print_list_stimuli(args):

    _ = args

    stimuli = list_stimuli()
    if len(stimuli) > 0:
        lines = ["    - {}".format(s) for s in stimuli]
        message = '\n'.join(lines)
        print(message)

    return


def defaults(args):

    _ = args

    raise NotImplementedError  # TODO replace.


def main():

    parser = argparse.ArgumentParser(description="A Python 3.6 module to generate visual stimuli.")
    parser.set_defaults(func=lambda args_: parser.print_usage())
    subparsers = parser.add_subparsers(title='positional arguments')

    subparser_config = subparsers.add_parser('config', help='config help')
    subparser_config.set_defaults(func=configure)

    subparser_init = subparsers.add_parser('init', help='init help')
    subparser_init.set_defaults(func=initialize)

    subparser_reinit = subparsers.add_parser('reinit', help='reinit help')
    subparser_reinit.set_defaults(func=reinitialize)

    subparser_list = subparsers.add_parser('list', help='list help')
    subparser_list.set_defaults(func=print_list_stimuli)

    subparser_generate = subparsers.add_parser('generate', help="generate help")
    subparser_generate.set_defaults(func=lambda args_: subparser_generate.print_usage())
    subparsers_generate = subparser_generate.add_subparsers(title="positional arguments")

    # Test.
    subparser_generate_test = subparsers_generate.add_parser('test', help='test help')
    subparser_generate_test.set_defaults(func=generate_test)

    # Square.
    subparser_generate_square = subparsers_generate.add_parser('square', help='square help')
    subparser_generate_square.set_defaults(func=generate_square)

    # Euler's full-field.
    subparser_generate_euler = subparsers_generate.add_parser('euler', help='euler help')
    subparser_generate_euler.set_defaults(func=generate_euler)

    # Drifting gratings.
    subparser_generate_dg = subparsers_generate.add_parser('dg', help='drifting gratings help')
    subparser_generate_dg.set_defaults(func=generate_dg)

    # Flashed images.
    subparser_generate_fi = subparsers_generate.add_parser('fi', help="flashed images")
    subparser_generate_fi.set_defaults(func=generate_fi)

    # Flashed images composition.
    subparser_generate_fi_comp = subparsers_generate.add_parser('fi_comp', help="flashed images composition")
    subparser_generate_fi_comp.set_defaults(func=generate_fi_comp)

    # Flashed images merging.
    subparser_generate_fi_merge = subparsers_generate.add_parser('fi_merge', help="flashed images merging")
    subparser_generate_fi_merge.set_defaults(func=generate_fi_merge)

    # Flashed images perturbed with checkerboard.
    subparser_generate_fipwc = subparsers_generate.add_parser('fipwc', help="flashed images perturbed with checkerboards")
    subparser_generate_fipwc.set_defaults(func=generate_fipwc)

    # Flashed images perturbed with frozen checkerboards.
    subparser_generate_fipwfc = subparsers_generate.add_parser('fipwfc', help="flashed images perturbed with frozen checkerboards")
    subparser_generate_fipwfc.set_defaults(func=generate_fipwfc)

    # Flashed images perturbed with random checkerboards.
    subparser_generate_fipwrc = subparsers_generate.add_parser('fipwrc', help="flashed images perturbed with random checkerboards")
    subparser_generate_fipwrc.set_defaults(func=generate_fipwrc)

    subparser_prepare = subparsers.add_parser('prepare', help="prepare help")
    subparser_prepare.set_defaults(func=lambda args_: subparser_prepare.print_usage())
    subparsers_prepare = subparser_prepare.add_subparsers(title="positional arguments")

    subparser_prepare_latest = subparsers_prepare.add_parser('latest', help="latest help")
    subparser_prepare_latest.set_defaults(func=prepare_latest)

    subparser_check = subparsers.add_parser('check', help="check help")
    subparser_check.set_defaults(func=defaults)

    args = parser.parse_args()
    args.func(args)
    # parser.print_usage()

    return


if __name__ == '__main__':

    main()
