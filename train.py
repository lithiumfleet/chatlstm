from torch import nn, optim, cuda, Tensor, save
from torch.utils.data import DataLoader
from model import LSTM
from dataset import TextDataLoader

class ChatTrainer():
    def __init__(self, model:LSTM, train_dl:DataLoader) -> None:
        self.model = model
        self.train_dl = train_dl
        self.optimizer = optim.SGD(model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.batch_size = train_dl.batch_size
        self.max_batch_idx = len(self.train_dl)//self.batch_size
        self.dev = 'gpu' if cuda.is_available() else 'cpu'
        self.ckpt_pth = './ckpt/'
        self.epoches = 10

    def train(self):
        self.model.train()
        for epoch in range(1,self.epoches+1):
            print('epoch:{}'.format(epoch))
            for batch_idx, (x, y) in enumerate(self.train_dl):
                batch_idx += 1
                x, y = x.to(self.dev), y.to(self.dev)
                batch_loss =  None
                for ix, iy in zip(x,y):
                    pred = self.model.forward(ix)
                    if not batch_loss is None:
                        batch_loss += self.loss_fn(pred, iy)
                    else:
                        batch_loss = self.loss_fn(pred, iy)
                batch_loss.backward()
                self._print_log(mode='batchinfo', batch_idx=batch_idx, batch_loss=batch_loss.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
            self.save_model(epoch=epoch)
        self._print_log('endinfo')
    
    def _print_log(self, mode:str, **kwargs):
        if mode.startswith('batch'):
            batch_idx, batch_loss = kwargs['batch_idx'], kwargs['batch_loss']
            templete = '[train info]    batch:{}    batch_loss={:>7f}'.format(batch_idx,batch_loss)
            print(templete)
        elif mode.startswith('save'):
            epoch, save_dir = kwargs['epoch'], kwargs['save_dir']
            templete = '[save info] epoch:{}    model ckpt save to {}'.format(epoch,save_dir)
            print(templete)
        elif mode.startswith('end'):
            templete = '[end of train]'
            print(templete)
        else:
            print('error info mode.')
    
    def save_model(self, epoch):
        name = 'E{}final.pth'.format(epoch)
        save(self.model, self.ckpt_pth+name)
        self._print_log('saveinfo', epoch=epoch, save_dir=self.ckpt_pth)

def main():
    chatlstm = LSTM()
    train_ds = TextDataLoader()
    chat_trainer = ChatTrainer(chatlstm, train_ds)
    chat_trainer.train()

if __name__ == '__main__':
    main()
    