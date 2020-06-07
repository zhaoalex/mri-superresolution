import argparse

def gettrainargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qmatrix", help="Use file as Q matrix")
    parser.add_argument("-v", "--vmatrix", help="Use file as V matrix")
    parser.add_argument("-s", "--scaling", help="Set scaling factor")
    parser.add_argument("-p", "--plot", help="Plot the learned filters", action="store_true")
    parser.add_argument("-d", "--done_file", help="Specify file of training files to skip")
    args = parser.parse_args()
    return args
