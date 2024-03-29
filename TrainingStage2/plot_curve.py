import datetime
import matplotlib.pyplot as plt


def plot_loss_and_lr(train_loss, learning_rate):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)
        fig.savefig('/content/drive/MyDrive/TrainingStage2/save_weights/loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP05, mAP0595):
    try:
        x1 = list(range(len(mAP05)))
        plt.plot(x1, mAP05, label='mAP@0.5')
        x2 = list(range(len(mAP0595)))
        plt.plot(x2, mAP0595, label='mAP@0.5:0.95')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP05))
        plt.legend(loc='best')
        plt.savefig('/content/drive/MyDrive/TrainingStage2/save_weights/mAP.png')
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)
