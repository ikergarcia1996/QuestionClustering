import argparse
import itertools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    args = parser.parse_args()

    lines = open(args.input_file, 'r', encoding='utf8').readlines()
    lines = [x.rstrip().replace('\t', ' ') for x in lines]
    c = list(itertools.combinations(lines, 2))

    with open(args.output_file, 'a+', encoding='utf8') as file:
        for pair in set(c):
            print(pair[0] + '\t' + pair[1], file=file)


if __name__ == '__main__':
    main()
