import argparse

class CmdLineParser(object):


    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Optional flags")

        self.parser.add_argument('pos_arg',type=str,
                                 help='Required local config file')

        self.parser.add_argument('--model_tag',type=str,
                                 help='tag for saved model')

        self.args = self.parser.parse_args()
        return

    def getParsedArgs(self):
        return self.args