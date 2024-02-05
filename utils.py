import os 
import numpy as np 
import numpy as np
import tensorflow as tf
from arg_parser import get_args
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import random
import ntpath

args = get_args()
IMG_HEIGHT=args.image_size
IMG_WIDTH=args.image_size
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS = args.output_channel



def path_leaf(path):
    #get the file name from the path
    head, tail = ntpath.split(str(path))
    return tail or ntpath.basename(head)

def load_png(image_file):
    #Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image)
    return image

def resize(input_image, real_image, height, width):
    #Resizes the input and target tensors (in 3D/4D formats only) to the given height and width using K-nearest neighbor method
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def normalize(input_image):
    #normalizes images from 0-255 to -1 and 1
    input_image = (input_image / 127.5) - 1
    return input_image
def denormalize(normalized_image):
    denormalized_image = ((normalized_image + 1) / 2) * 255
    return denormalized_image

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    #example: orig_image =interval_mapping(norm_image, 0.0, 1.0, 0, 255).astype('uint8')
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def load_npy(path):
    #reads numpy saved data
    image=np.load(path)
    image=np.transpose(image, axes=[1,2,0])
    return tf.cast(image, tf.float32)


def random_crop(input_image, real_image):
    #randomly crops images 
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

def random_rotation_pair(image1, image2, max_angle):
    #applying random rotation to the image
    image1=np.asarray(image1)
    image2=np.asarray(image2)
    # Generate a random rotation angle within the specified range
    angle = random.uniform(-max_angle, max_angle)

    # Get the height and width of the images
    height, width = image1.shape[0],image1.shape[1]

    # Define the rotation center as the center of the images
    center = (width // 2, height // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation on both images
    rotated_image1 = cv2.warpAffine(np.asarray(image1), rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    rotated_image2 = cv2.warpAffine(np.asarray(image2), rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    return rotated_image1, rotated_image2

@tf.function()
def augmentation(input_image, real_image):
    #data augmentation by randomly deciding to miror or rotate the image. if random_gen is more than 0.5 does flip the image otherwise rotates 25 degrees.
  # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)
    random_gen=random.uniform(0, 1)

    if random_gen > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    else:
        #Random rotation
        input_image, real_image=random_rotation_pair(input_image, real_image, 25)

    return input_image, real_image



def load_data(directory, patient_list, modality,augment=False,):

    #Data loader based on the patient list. Augmentation can be enabled in training mode. mode referes to the input image modality, either single or multi modal training
    input_image=[]
    real_image=[]   

    for patient in (patient_list):
        print('Subject:', patient)
        nac = load_npy(os.path.join(directory,patient,patient+'_nac.npy'))
        water = load_npy(os.path.join(directory,patient,patient+'_water.npy'))
        fat = load_npy(os.path.join(directory,patient,patient+'_fat.npy'))
        mac = load_npy(os.path.join(directory,patient,patient+'_mac.npy'))
        #print(np.shape(nac))
        zlen=mac.shape[2]
        for plane in range(zlen):
            _image_nac = nac[:, :, plane]
            _image_water = water[:, :, plane]
            _image_fat = fat[:, :, plane]
            _image_mac=mac[:,:,plane]

            if np.min(_image_mac)==0 and np.max(_image_mac)==0:
                continue
            if np.min(_image_nac)==0 and np.max(_image_nac)==0:
                continue
            if np.min(_image_water)==0 and np.max(_image_water)==0:
                continue
            if np.min(_image_fat)==0 and np.max(_image_fat)==0:
                continue
           
            _image_nac = normalize(_image_nac)
            _image_water = normalize(_image_water)
            _image_fat = normalize(_image_fat)
            _image_mac = normalize(_image_mac)

            if modality=='multi':
                inp=np.stack((_image_nac,_image_water,_image_fat),axis=-1)
                    
                input_image.append(inp)
            else:
                inp=np.stack((_image_nac,_image_nac,_image_nac),axis=-1)
                input_image.append(inp)
            real=np.stack((_image_mac,_image_mac,_image_mac),axis=-1)
            real_image.append(real)
            
            if augment:
                inp,real= augmentation(inp, real)
                input_image.append(inp)
                real_image.append(real)
                
            
        del nac, water, fat, mac 
    input_image = tf.cast(np.asarray(input_image), tf.float32)
    real_image = tf.cast(np.asarray(real_image), tf.float32)
    
    return input_image,real_image

            

def inference_GAN(directory, patient_list,model,outpath, modality):
  

    for patient in (patient_list):
        print('Subject:', patient)
        nac = load_npy(os.path.join(directory,patient,patient+'_nac.npy'))
        water = load_npy(os.path.join(directory,patient,patient+'_water.npy'))
        fat = load_npy(os.path.join(directory,patient,patient+'_fat.npy'))
        mac = load_npy(os.path.join(directory,patient,patient+'_mac.npy'))
        zlen=mac.shape[2]
        slice_num=1
        outpath=outpath+"/"+patient+"/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for plane in range(zlen):
            _image_nac = nac[:, :, plane]
            _image_water = water[:, :, plane]
            _image_fat = fat[:, :, plane]
            _image_mac=mac[:,:,plane]

            if np.min(_image_mac)==0 and np.max(_image_mac)==0:
                continue
            if np.min(_image_nac)==0 and np.max(_image_nac)==0:
                continue
            if np.min(_image_water)==0 and np.max(_image_water)==0:
                continue
            if np.min(_image_fat)==0 and np.max(_image_fat)==0:
                continue
           
            _image_nac = normalize(_image_nac)
            _image_water = normalize(_image_water)
            _image_fat = normalize(_image_fat)
            _image_mac = normalize(_image_mac)

            if modality=='multi':
                real_A=np.stack((_image_nac,_image_water,_image_fat),axis=-1)
                    
            else:
                real_A=np.stack((_image_nac,_image_nac,_image_nac),axis=-1)
                
            
            real_B=np.stack((_image_mac,_image_mac,_image_mac),axis=-1)
            
            
            fake_B = model.gen_G(np.expand_dims(real_A, axis=0), training=True)  
            proto_tensor_A = tf.make_tensor_proto(real_A)
            real_A=tf.make_ndarray(proto_tensor_A)
            proto_tensor_B = tf.make_tensor_proto(real_B)
            real_B=tf.make_ndarray(proto_tensor_B)
            proto_tensor_fB = tf.make_tensor_proto(fake_B)
            fake_B=tf.make_ndarray(proto_tensor_fB)
            fake_B=np.squeeze(fake_B, axis=0)
            #fake_B=np.squeeze(fake_B, axis=-1)
            denormalize

            real_A = denormalize(real_A[:,:,0]).astype('uint8')
            real_B = denormalize(real_B).astype('uint8') 
            fake_B = denormalize(fake_B).astype('uint8') 

            #real_A = interval_mapping(real_A[:,:,0], -1.0, 1.0, 0, 255).astype('uint8')
            #real_B = interval_mapping(real_B, -1.0, 1.0, 0, 255).astype('uint8') 
            #fake_B = interval_mapping(fake_B, -1.0, 1.0, 0, 255).astype('uint8') 

            real_A = Image.fromarray(np.array(real_A))
            real_B = Image.fromarray(np.array(real_B))
            fake_B = Image.fromarray(np.array(fake_B))
            
            outpathA=outpath+ patient+"_"+str(slice_num)+'_real_A'
            outpathB=outpath+ patient+"_"+str(slice_num)+'_real_B'
            outpath_fake=outpath+ patient+"_"+str(slice_num)+'_fake_B'
            real_A.save(f"{outpathA}.png" )
            real_B.save(f"{outpathB}.png")
            fake_B.save(f"{outpath_fake}.png")
           
            
            slice_num=slice_num+1
        print('All slices are saved for subject:', patient)
        print(f"total numer of files saved: {slice_num}")

        del nac, water, fat, mac 
            
    return print('Testing phase is successfully Completed')

                
