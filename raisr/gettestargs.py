import argparse

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", help="Use file as filter")
    parser.add_argument("-p", "--plot", help="Visualizing the process of RAISR image upscaling", action="store_true")
    parser.add_argument("-s", "--scaling", help="Set scaling factor")
    parser.add_argument('-w', '--write', action='store_true', help="Write HR images to disk")
    args = parser.parse_args()
    return args
