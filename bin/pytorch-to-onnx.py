#
#   Run this in parent folder using the shell script bin/pytorch-to-onnx
#
import argparse

from core.credits import slim as epilog

DEFAULT_DEVICE = 'cpu'
DEFAULT_OUT_NAME = 'output'

def main ():
    parser = argparse.ArgumentParser(
        description='Converts a Pytorch model (.pth) into an ONNX model (.onnx)',
        epilog=epilog
    )
    parser.add_argument('-i', '--input',    type = str,  required = True, help = 'input file path')
    parser.add_argument('-o', '--output',   type = str,  required = False, help = 'output file path, defaults to same as input with .onnx ending')
    parser.add_argument('-d', '--device',   type = str,  default = DEFAULT_DEVICE, help = 'pytorch device, see: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device')
    parser.add_argument('-q', '--quiet',    type = bool, default = False, help = 'skip default output')
    parser.add_argument('-n', '--no-check', type = bool, default = False, help = 'do not check for file name extensions')
    parser.add_argument('-m', '--out-name', type = str,  default = DEFAULT_OUT_NAME, help = 'Output nme for for the onnx model')
    args = parser.parse_args()

    if args.input == args.output:
        print('Error: Input needs to be different from output. (%s)' % args.input)
        exit(1)
    
    if not args.no_check:
        if not args.input.endswith('.pth'):
            print('Error: Input (%s) needs to end with .pth (disable this error using --no_check)' % args.input)
            exit(1)
        if args.output and not args.output.endswith('.onnx'):
            print('Error: Output (%s) needs to end with .onnx (disable this error using --no_check)' % args.output)
            exit(1)

    if not args.output:
        if args.input.endswith('.pth'):
            args.output = args.input[:-4] + '.onnx'
        else:
            args.output = args.input + '.onnx'

    run(args)

def quiet ():
    return

def hash (f):
    import hashlib
    hasher = hashlib.sha1()
    with open(f, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def mobile_facenet_impl (args):
    from core.model import MobileFacenet
    import torch
    state_dict = torch.load(args.input, map_location=args.device)
    net = MobileFacenet()
    net.load_state_dict(state_dict['net_state_dict'])
    net.eval()
    return net

def export (args, net, output):
    import torch
    torch.onnx.export(
        model        = net,
        args         = net.random_input().to(torch.device(args.device)),
        f            = output,
        verbose      = False,
        # Currently the system supports only a single input/output
        input_names  = ['input'],
        output_names = [args.out_name]
    )
    return hash(output)

def run (args):
    import os
    out = quiet if args.quiet else print

    out('Loading torch model from %s' % args.input)
    impl = mobile_facenet_impl(args)

    out(impl)

    out('Exporting onnx model to %s' % args.output)
    
    out_hash = export(args, impl, args.output)

    verify_output = '%s.verify' % args.output

    out('Exporting verify onnx model to %s' % verify_output)

    verify_hash = export(args, impl, verify_output)

    out('Verifying the output.')

    assert out_hash == verify_hash, 'Verification failed, it seems to be a dynamic model!\n  %s != %s' % (out_hash, verify_hash)

    out('Cleaning up verify onnx model %s' % verify_output)

    os.unlink(verify_output)

    out('Done.')

main()
