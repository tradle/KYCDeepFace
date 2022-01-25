#
#   Run this in parent folder using:
#
#   $ python -m bin.pytorch-to-onnx
#
import torch
import argparse

from core.model import MobileFacenet
from core import credits

DEFAULT_WIDTH = 96
DEFAULT_HEIGHT = 112
DEFAULT_COLORS = 3
DEFAULT_DEVICE = 'cpu'

def main ():
    parser = argparse.ArgumentParser()
    parser.description = 'Converts a Pytorch model (.pth) into an ONNX model (.onnx)'
    parser.epilog = credits.slim
    parser.add_argument('-o', '--output',   type = str,  required = True, help = 'output file path')
    parser.add_argument('-i', '--input',    type = str,  required = True, help = 'input file path')
    parser.add_argument('-x', '--width',    type = int,  default = DEFAULT_WIDTH,  help = 'image width')
    parser.add_argument('-y', '--height',   type = int,  default = DEFAULT_HEIGHT, help = 'image height')
    parser.add_argument('-c', '--colors',   type = int,  default = DEFAULT_COLORS, help = 'color channels')
    parser.add_argument('-d', '--device',   type = str,  default = DEFAULT_DEVICE, help = 'pytorch device, see: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device')
    parser.add_argument('-q', '--quiet',    type = bool, default = False, help = 'skip default output')
    parser.add_argument('-n', '--no-check', type = bool, default = False, help = 'do not check for file name extensions')
    args = parser.parse_args()

    if args.input == args.output:
        print('Error: Input needs to be different from output. (%s)' % args.input)
        exit(1)
    
    if not args.no_check:
        if not args.input.endswith('.pth'):
            print('Error: Input (%s) needs to end with .pth (disable this error using --no_check)' % args.input)
            exit(1)
        if not args.output.endswith('.onnx'):
            print('Error: Output (%s) needs to end with .onnx (disable this error using --no_check)' % args.output)
            exit(1)

    print(args)

def quiet ():
    return

def run (args):
    out = quiet if args.quiet else print

    out('Loading torch model from %s' % args.input)
    ckpt = torch.load(args.input, map_location=args.device)

    net = MobileFacenet()
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    out(net)

    out('Exporting onnx model to %s' % args.output)
    torch.onnx.export(
        net,
        torch.randn(1, args.colors, args.height, args.width).to(torch.device(args.device)),
        args.output,
        verbose = False,
        input_names = ['input'],
        output_names = ['classes']
    )

main()
