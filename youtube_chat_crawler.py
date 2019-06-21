import requests
import re
import time
import argparse


def youtube_crawler():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chat_url', required=True, type=str) #https://www.youtube.com/live_chat?is_popout=1&v=' URL del chat de youtube cuando le das a expandir
    parser.add_argument('-o', '--output_file', type=str, default='Dataset/leidas.txt')
    args = parser.parse_args()

    session = requests.Session()
    session.headers[
        'User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
    i = 0

    with open(args.output_file, 'a+') as file:
        while (True):
            i += 1
            response = session.get(args.chat_url)
            html = response.text
            # preguntas = re.findall(r'#PREGUNTA(.*?)}', html)
            preguntas = re.findall(r'\"liveChatTextMessageRenderer\":{"message\":{\"simpleText\":(.*?)}', html)
            print(i, end="\r")
            for pregunta in preguntas:
                print(pregunta, file=file)

            time.sleep(0.2)


if __name__ == '__main__':
    youtube_crawler()

