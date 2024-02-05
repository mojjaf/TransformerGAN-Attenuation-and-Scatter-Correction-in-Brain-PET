import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str,default='/home/mojjaf/Attenuation Correction/Code',
                        help="project directory")
    parser.add_argument("--data_dir", type=str,default='/home/mojjaf/Attenuation Correction/Multidim_vols/all_data',
                        help="all image dir")
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--output_channel", type=int, default=3,
                        help="output image dimension, 1= 2D or 3=2.5D")
    parser.add_argument("--input_channel", type=int, default=3,
                        help="input image dimension, 1=2D or 3=2.5D")
    parser.add_argument("--nb_epochs", type=int, default=2,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--folds", type=int, default=5,
                        help="number of folds")
    parser.add_argument("--data_modality", type=str, default='multi',
                        help="input modality; 'single', or 'multi' is expected")
    parser.add_argument("--network", type=str, default='Pix2Pix',
                        help="network architecture, 'Pix2Pix', 'AG-Pix2Pix','Swin-GAN','ViT-GAN',")
    parser.add_argument("--dataset", type=str, default="Brain_PETMR",
                        help="checkpoint dir")
   
    args, unknown = parser.parse_known_args()

    return args 
