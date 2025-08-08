import argparse
import logging
import sys
import os
import json

def bin_gen(label_file, model_file, output_file):
    page_align = 1024
    labels = []
    with open(label_file, 'r') as label_fd:
        while True:
            line_str = label_fd.read().splitlines()
            if len(line_str) == 0:
                break
            labels.append(line_str)
    label_fd.close()

    model_file_stat = os.stat(model_file)

    json_data = {}
    json_data['model_size'] = model_file_stat.st_size
    json_data['model_offset'] = 1024000
    json_data['class'] = labels[0]

    w = json.dumps(json_data)
    w = w + '\0'
 
    tflite_offset = (len(w) + page_align) & ~(page_align - 1)
    print(tflite_offset)
    json_data['model_offset'] = tflite_offset

    w = json.dumps(json_data)
    w = w + '\0'

    with open(output_file, 'wb') as bin_file:
        bin_file.write(w.encode('ascii'))
        bin_file.seek(tflite_offset, 0)
        model_fd = open(model_file, 'rb')
        print(model_fd)
        print(bin_file)
        m = model_fd.read()
        bin_file.write(m)

def _main(argv):
    """model and labels packer"""

    parser = argparse.ArgumentParser(
        prog="Model_Labels_Packer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pack model and labels to binary file",
        epilog=__doc__,
        # Help action will be added later, after all subparsers are created,
        # so it doesn't interfere with the creation of the dynamic subparsers.
        add_help=True,
    )
    parser.add_argument("-l", "--label_file",required = True, help="input lables.txt file")
    parser.add_argument("-m", "--model_file",required = True, help="input model file")
    parser.add_argument("-o", "--output_file", help="output bin file")
    args = parser.parse_args()

    label_file = args.label_file
    output_file = args.output_file
    model_file = args.model_file

    if output_file == None:
        output_file =  os.path.splitext(os.path.basename(model_file))[0]
        output_file = output_file + '.bin'

    print(label_file)
    print(model_file)
    print(output_file)
    bin_gen(label_file, model_file, output_file)

def main():
    sys.exit(_main(sys.argv[1:]))

if __name__ == "__main__":
    main()
