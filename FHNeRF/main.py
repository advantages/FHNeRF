import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
import numpy as np
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=1000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
# parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=1)
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
torch.set_printoptions(precision=5,sci_mode=False)


image_path = sorted(glob(os.path.join("../test_new/", "*.*")))

for i in range(len(image_path)):
# for i in range(min_id, max_id + 1):
    print(f'Image {i}')


    img = np.load(image_path[i]).astype(np.float)
    img = transforms.ToTensor()(img).float().to(device, dtype)

    img_=img

    func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=50,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        # final_activation=torch.nn.ReLU(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)
    
#     # Set up training
    trainer = Trainer(func_rep, lr=args.learning_rate,img_s1=img_.shape[1],img_s2=img_.shape[2])
    print("im_shape:",img.shape)
    coordinates, features = util.to_coordinates_and_features(img)

    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=func_rep, image=img)
    print(f'Full precision bpp: {fp_bpp:.5f}')


    trainer.train(coordinates, features, num_iters=args.num_iters)

    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')


    # Save best model
    torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')




