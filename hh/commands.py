import fire


class HeadHunter:
    def __init__(self):
        print ('Running hh')

    def process_data(self):
        print('Processing data')

def main():
    fire.Fire(HeadHunter)
