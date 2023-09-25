import os
from torch import load, no_grad

class Chatter():
    def __init__(self, model_path) -> None:
        self.model = self.load_ckpt(model_path)

    def load_ckpt(self, model_pth):
        # lstm = LSTM()
        lstm = load(model_pth)
        return lstm

    
    def loop(self):
        self.model.evalmode = True
        # self.model.eval()
        with no_grad():
            while True:
                usr_input = input('>')
                bot_res = self.model.forward(usr_input)
                print('bot:'+bot_res)

def main():
    pth = './ckpt\E2final.pth'
    chatter = Chatter(pth)
    chatter.loop()

if __name__ == '__main__':
    main()