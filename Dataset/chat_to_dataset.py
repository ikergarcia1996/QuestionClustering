questions = open('youtube_chat.txt', 'r').readlines()
with open('question_dataset.txt', 'w+') as file:
    for s in set(questions):
        print(s.rstrip()[1:-1], file=file)
