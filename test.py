import os
import argparse
import cv2
import torch
from torch import nn

from dataset import MyData
from models.baseline import BSL
from utils import save_tensor_img
from config import Config


def main(args):
    # Init model
    config = Config()

    device = torch.device(config.device)
    model = BSL().to(device)
    print('Testing with model {}'.format(args.ckpt))
    state_dict = torch.load(args.ckpt)

    model.to(device)
    model.load_state_dict(state_dict)

    model.eval()

    for testset in args.testsets.split('+'):
        print('Testing {}...'.format(testset))
        saved_root = os.path.join(args.pred_dir, testset)
        data_loader_test = torch.utils.data.DataLoader(
            dataset=MyData(data_root=os.path.join(config.data_root_dir, config.dataset, testset), image_size=config.size, is_train=False),
            batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
        for batch in data_loader_test:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            label_paths = batch[2]
            with torch.no_grad():
                scaled_preds = model(inputs)[-1]

            os.makedirs(saved_root, exist_ok=True)

            for idx_sample in range(scaled_preds.shape[0]):
                res = nn.functional.interpolate(
                    scaled_preds[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE),
                    mode='bilinear',
                    align_corners=True
                ).sigmoid()
                save_tensor_img(res, os.path.join(saved_root, label_paths[idx_sample].replace('\\', '/').split('/')[-1]))   # test set dir + file name


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--testsets',
                        default='DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4',
                        type=str,
                        help="Options: 'DIS-VD', 'DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'")
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--pred_dir', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)