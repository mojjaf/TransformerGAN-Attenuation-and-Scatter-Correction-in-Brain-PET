#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

print("\n * GPU CONFIGURATION* \n")

############################## Define the number of Physical and Virtual GPUs #############

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[3:4], 'GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20000),
             tf.config.LogicalDeviceConfiguration(memory_limit=20000)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
        print(e)


print("\n * GPU Setup compelted... * \n")


################################# load libraries #######################################
import sys

import os 
import numpy as np 
import datetime
from matplotlib import pyplot as plt
from utils import load_data,inference_GAN
from sklearn.model_selection import KFold
from networks import get_resnet_generator, get_discriminator
from keras_unet_collection import models
from AttentionGAN import AG_Pix2Pix_Generator
sys.path.insert(0, '/home/mojjaf/Attenuation Correction/Code/Pix2Pix/transunet/')
from transunet import TransUNet
from arg_parser import get_args
from tensorflow import keras


# In[3]:

################################# set up the experiment ################################

args = get_args()
model_for_training=args.network
project_dir = args.main_dir
PATH=args.data_dir
IMG_WIDTH = args.image_size
IMG_HEIGHT = args.image_size
INPUT_CHANNELS=args.input_channel
BATCH_SIZE =args.batch_size
OUTPUT_CHANNELS = args.output_channel
data_path = args.data_dir
num_epochs=args.nb_epochs
fold=args.folds
image_modality=args.data_modality
datadir = os.listdir(data_path)
kfold_CV=str(fold)
data_type=args.dataset
experiment_model= "Experiment_"+data_type+"_"+image_modality+"_"+model_for_training+"_"+kfold_CV+"Fold"
print('Input and Output Channels:',INPUT_CHANNELS, OUTPUT_CHANNELS)
print('Image Dimension:',IMG_HEIGHT, IMG_WIDTH)
experiment = "/experiments/"+experiment_model+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"  # 
model_name = "/saved_models"+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/" 
print(f"\nExperiment: {experiment}\n")

experiment_dir = project_dir+experiment

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir) 
    
visual_dir=experiment_dir+"/training_visuals"
if not os.path.exists(visual_dir):
    os.makedirs(visual_dir) 
    
models_dir = experiment_dir+model_name
if not os.path.exists(models_dir):
    os.makedirs(models_dir) 
    
output_preds_2D = experiment_dir+'predictions/'

if not os.path.exists(output_preds_2D):
    os.makedirs(output_preds_2D)


################################# set up the network and loss functions ################################

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = loss_object(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = loss_object(tf.ones_like(real), real)
    fake_loss = loss_object(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) 

def calc_ssim_loss(real_image, fake_image):

    return 0.5*(1 - tf.image.ssim(fake_image, real_image, max_val=2.0)[0])


class cGAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        lambda_gen=100,
        lambda_structure=0.5,
    ):
        super().__init__()
        self.gen_G = generator
        self.disc_X = discriminator
        self.lambda_L1 = lambda_gen
        self.lambda_ssim=lambda_structure
        

    def call(self, inputs):
        return (
            self.disc_X(inputs),
            self.gen_G(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        disc_X_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.generator_L1_loss = keras.losses.MeanAbsoluteError()
        self.discriminator_loss_fn = disc_loss_fn

    def train_step(self, batch_data):
        # x is PET Domain X and y is PET Domain Y [Domain is either a tracer or noise condition]
        real_x, real_y = batch_data
        # For Pix2Pix, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images to the corresponding discriminator.
        # 3. Calculate the generators total loss (adversarial + L1 + SSIM)
        # 4. Calculate the discriminator loss
        # 5. Update the weights of the generator
        # 6. Update the weights of the discriminator
        # 7. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Generator output for translating PET D1 to PET D2
            fake_y = self.gen_G(real_x, training=True)
            # Discriminator output
            disc_real_out = self.disc_X([real_x,real_y], training=True)
            disc_gen_out = self.disc_X([real_x,fake_y], training=True)
            
            ##GENERATOR LOSS CALCULATIONS
            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(disc_gen_out)
            # Generator L1 loss
            gen_loss_L1 = self.generator_L1_loss(fake_y, real_y) * self.lambda_L1
          
            
    
            # Total generator loss
            total_gen_loss = gen_G_loss + gen_loss_L1

            # Discriminator loss
            disc_loss = self.discriminator_loss_fn(disc_real_out, disc_gen_out)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_gen_loss, self.gen_G.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_loss, self.disc_X.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )

        return {
            "Generator_loss": total_gen_loss,
            "Discriminator_loss": disc_loss,
        }
    def test_step(self, val_data):
        real_x, real_y = val_data
        fake_y = self.gen_G(real_x, training=True)
        disc_real_out = self.disc_X([real_x,real_y], training=True)
        disc_gen_out = self.disc_X([real_x,fake_y], training=True)
        
        gen_G_loss = self.generator_loss_fn(disc_gen_out)
        # Generator L1 loss
        gen_loss_L1 = self.generator_L1_loss(fake_y, real_y) * self.lambda_L1

        # Total generator loss
        total_gen_loss = gen_G_loss + gen_loss_L1 

        # Discriminator loss
        disc_loss = self.discriminator_loss_fn(disc_real_out, disc_gen_out)
        return {
            "Generator_loss": total_gen_loss,
            "Discriminator_loss": disc_loss}


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(visual_dataset_mac.take(self.num_img)):
            prediction = self.model.gen_G(img,training=True)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.float32)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.float32)
            #print(np.shape(img),np.shape(prediction))
            ax[i, 0].imshow(img[...,0],cmap='gray')
            ax[i, 1].imshow(prediction[...,0],cmap='gray')
            ax[i, 0].set_title("Target image")
            ax[i, 1].set_title("Translated image (Synthetic)")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.utils.array_to_img(prediction)
            prediction.save(visual_dir+"/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1))
        plt.show()
        plt.close()

################################# set up the model training and testing ################################

fold_number=1
kf = KFold(n_splits = fold,shuffle = True, random_state = 9134)

for train_sub, test_sub in kf.split(datadir):
    print('fold running:',fold_number)
    train_cases = [datadir[i] for i in train_sub]
    test_cases = [datadir[i] for i in test_sub]
    print("\n * Test Subjects * \n")
    print(test_cases)
    print("\n * Train Subjects * \n")
    print (train_cases)
   
    
    input_image_train,real_image_train=load_data(data_path, train_cases,image_modality,augment=True)
    
    print("\n * Train dataset is successfully loaded. * \n")
    print("\n * Total number of train samples: ",np.shape(input_image_train))
    
    input_image_test,real_image_test=load_data(data_path, test_cases,image_modality,augment=False)
    print("\n * Test dataset is successfully loaded. * \n")
    print("\n * Total number of test samples: ", np.shape(input_image_test))
    BUFFER_SIZE = len(input_image_train)
    BATCH_SIZE = 1
    
    AUTOTUNE = tf.data.AUTOTUNE
    with tf.device('/device:cpu:0'):
        
        datasetx = tf.data.Dataset.from_tensor_slices((input_image_train))

        datasety = tf.data.Dataset.from_tensor_slices((real_image_train))
        train_dataset = tf.data.Dataset.zip((datasetx, datasety))

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)

        datasetxVal = tf.data.Dataset.from_tensor_slices((input_image_test))
        datasetyVal = tf.data.Dataset.from_tensor_slices((real_image_test))
        val_dataset = tf.data.Dataset.zip((datasetxVal, datasetyVal)).batch(BATCH_SIZE)
        
        MAC_img = tf.data.Dataset.from_tensor_slices(real_image_test[50:60,...])
        visual_dataset_mac = tf.data.Dataset.zip((MAC_img)).batch(BATCH_SIZE)

    print("\n * Visualization dataset successfully created. * \n")

    
    print("\n * dataset compelted for training in fold:",fold_number)
    tf.keras.backend.clear_session()
    #########################################################
    print("\n * Creating the Generative Model. * \n")

    if model_for_training=='Pix2Pix':
        gen_G = get_resnet_generator(name="generator_G")
    elif model_for_training=='AG-Pix2Pix':
        gen_G=AG_Pix2Pix_Generator()
    elif model_for_training=='Swin-GAN':
        gen_G=models.swin_unet_2d((256, 256, 3), filter_num_begin=32, n_labels=3, depth=4, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Tanh', shift_window=True, name='swin_unet')
        
    elif model_for_training=='ViT-GAN':
        gen_G=TransUNet(image_size=256,patch_size=16,num_classes=3, pretrain=False)
    else:
        print('Network architecture is unknown. Defult networks are Pix2Pix, AG-Pix2Pix, Swin-GAN, and ViT-GAN')
    
    disc_X = get_discriminator(name="discriminator_X")
    ###########################################
    print("\n * Training the Model. This will take some time ... * \n")
    # Create gan model
    gan_model = cGAN(generator= gen_G, discriminator=disc_X)

    # Compile the model
    gan_model.compile(
        gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
        disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    # Callbacks
    plotter = GANMonitor()
    checkpoint_filepath = models_dir+"/model_checkpoints/pix2pix_checkpoints.{epoch:03d}"
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath) 

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True)

    
    #####################################
    gan_model.fit(train_dataset,epochs=num_epochs,validation_data=val_dataset,callbacks=[plotter],)

    gan_model.gen_G.save_weights(models_dir+'genG_'+str(fold_number) + '.h5')
    print("\n * MODEL G SAVED * \n")
    ####
    print("\n * training compelted. Saving the model...* \n")
    #########################################################
    print("\n * Now Testing the Model. This will take some time ... * \n")
    outpath=output_preds_2D+"/"+"fold_"+str(fold_number)

    inference_GAN(data_path, test_cases,gan_model,outpath, image_modality)
    #####################################
    ##DO TESTING
    print("\n * training compelted...* \n")
     
    print("\n * results saved. Going to next fold... * \n")

    fold_number+=1
    
print("\n * Kfold Cross Validation Completed Successfully ... * \n")
    

