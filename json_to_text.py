import argparse
import glob
import json
import logging
import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    args = parser.parse_args()

    with open(args.output_file, 'w') as out_file:

        files = glob.iglob(args.input_directory+'/**/*.json', recursive=True)

        num_files = 0
        for file in files:
            num_files += 1
        print('Total files: ' + str(num_files))
        total_words = 0

        for i_file, filename in enumerate(glob.iglob(args.input_directory+'/**/*.json', recursive=True)):
            if i_file % 100 == 0:
                string = '<' + str(datetime.datetime.now()) + '>  ' + 'Generating corpus: ' + str(
                    int(100 * i_file / num_files)) + '% . Number of words int he corpus: ' + str(total_words)
                print(string, end="\r")

            try:
                with open(filename) as json_file:
                    j = json.load(json_file)['text']
                    print(j, file=out_file)
                    total_words += len(j.split(' '))
            except:
                logging.warning('Error in file: ' + str(filename) + ' skipping file.')

        string = '<' + str(
            datetime.datetime.now()) + '>  ' + 'Generating corpus: 100% . Number of words in the corpus: ' + str(
            total_words)
        print(string)

if __name__ == '__main__':
    main()
