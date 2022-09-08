import colorsys
import math

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.measure import inertia_tensor, perimeter, shannon_entropy
from skimage.filters import gaussian, median, sobel_v
from sklearn.metrics import mean_squared_error
from skimage.transform import resize
    
import time
import matplotlib.pyplot as plt
from PIL import Image

import imagehash
import imageio
import os
# from lshash import lshash


import torch.nn as nn
from torch import tensor, double, float64, flatten
import torch.nn.functional as F
import torch
import torchvision.models as torch_models
from torchvision import datasets, transforms as T

from cppn_neat.util import get_best_solution_from_all_runs


def save_image(individual, width, height, color_mode):
    img = individual.get_image(height, width, color_mode)
    img = np.clip(img, 0, 1)
    if(color_mode == 'L'):
        plt.imsave(f'saved_imgs/img_{time.time()}.png', arr=np.asarray(img), format='png')
    elif(color_mode == "HSL"):
        img = hsv2rgb(img)
        plt.imsave(f'saved_imgs/img_{time.time()}.png', arr=np.asarray(img), format='png')
    else:
        plt.imsave(f'saved_imgs/img_{time.time()}.png', arr=np.asarray(img), format='png')

    # plt.imsave(f'saved_imgs/img_{time.time()}.png', arr=np.asarray(img), format='png')

def avg_pixel_distance(train_images, candidate_images):
    # calculate the average pixel difference
    diffs = np.abs(train_images - candidate_images)
    
    if len(candidate_images.shape) > 3:
        # color
        avgs = np.nanmean(diffs, axis=(1, 2, 3))   # TODO hacky, why are there NaN pixels?
    else:
        avgs = np.nanmean(diffs, axis=(1, 2))   # TODO hacky, why are there NaN pixels?
    
    return avgs
    
def hybrid_pixel_distance(img0, img1):
    return .8*avg_pixel_distance(img0, img1) + .2 * max_pixel_distance(img0, img1)

def max_pixel_distances(train_images, candidate_images):
    # img0_flat = img0.flatten()
    # img1_flat = img1.flatten()
    # calculate the max pixel difference
    diff = np.subtract(train_images, candidate_images)  # elementwise 
    diff = np.abs(diff)
    max = np.max(diff, axis=(1,2))  
    return max


k = 50 # hash size (10)
L = 5  # number of tables (5)
def local_sensitivity_hashing(train, candidate):
    # note that uniform planes are random and therefore this algorithm is non-deterministic 
    lsh = lshash.LSHash(hash_size=k, input_dim=len(train.flat), num_hashtables=L)
    lsh.index(train.flat)
    lsh.index(candidate.flat)
    distances = []
    for i, table in enumerate(lsh.hash_tables):
        binary_hash = lsh._hash(lsh.uniform_planes[i], candidate.flat)
        distance = lshash.LSHash.hamming_dist(list(table.keys())[0], binary_hash) # table.keys()[0] is train image
        distances.append(distance)
            
    return np.mean(distances) / k

def dhash_v_images(imgs, hash_size: int) -> np.ndarray:
    if len(imgs[0].shape) > 2:
        # color image, convert to greyscale TODO
        R, G, B = imgs[:,:,:,0], imgs[:,:,:,1], imgs[:,:,:,2]
        imgs = 0.2989 * R + 0.5870 * G + 0.1140 * B
    resized = np.zeros((imgs.shape[0], hash_size + 1, hash_size))
    for n,i in enumerate(imgs):
        resized[n,:,:] = resize(imgs[n,:,:], (hash_size + 1, hash_size), anti_aliasing=True)

	# compute differences between rows
    diff = resized[:,1:, :] > resized[:,-1:, :]
    hashed = diff
    return hashed

def dhash_h_images(imgs, hash_size: int) -> np.ndarray:
    """ 
    Calculate the dhash signature of a given image 
    Args:
        image: the image (np array) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2
    
    Returns:
        Image signature as Numpy n-dimensional array
    """
    
    if len(imgs[0].shape) > 2:
        # color image, convert to greyscale TODO
        R, G, B = imgs[:,:,:,0], imgs[:,:,:,1], imgs[:,:,:,2]
        imgs = 0.2989 * R + 0.5870 * G + 0.1140 * B
    resized = np.zeros((imgs.shape[0], hash_size + 1, hash_size))
    for n,i in enumerate(imgs):
        resized[n,:,:] = resize(imgs[n,:,:], (hash_size + 1, hash_size), anti_aliasing=True)

	# compute differences between columns
    diff = resized[:,:, 1:] > resized[:,:, :-1]
    hashed = diff
    return hashed

def difference_hashing(train, candidates, hash_size):
    train_images = np.expand_dims(train, 0) # add "batch" dim
    hashes0h = dhash_h_images(train_images, hash_size)
    hashes1h = dhash_h_images(candidates, hash_size)
    hashes0v = dhash_v_images(train_images, hash_size)
    hashes1v = dhash_v_images(candidates, hash_size)
    diff_h = np.sum(hashes0h != hashes1h, axis=(1,2)) / len(hashes0h[0].flat)
    diff_v = np.sum(hashes0v != hashes1v, axis=(1,2)) / len(hashes0v[0].flat)
    
    return (diff_v+diff_h)/2

def clip01(img):
    return np.clip(img, 0, 1)

def diff_mean_square_error(train, candidates):
    mse = (np.square(train - candidates)).mean(axis=(1,2)) # mean over pixels
    return mse 

def diff_shannon_entropy(img0, img1):
    return abs(shannon_entropy(img0) - shannon_entropy(img1))

def diff_inertia_tensor(img0, img1):
    return abs(inertia_tensor(img0) - inertia_tensor(img1))

def diff_perimeter(img0, img1):
    return abs(perimeter(img0) - perimeter(img1))

def diff_sobel_feature_set(img0, img1):
    return diff_feature_set(sobel_v(img0), sobel_v(img1))

"""To compare two images and calculate fitness, each is de-
fined by a feature set that includes the grayscale value at each pixel
location (at N1 × N2 pixels) and the gradient between adjacent
pixel values. The candidate feature set is then scaled to correspond
with the normalized target feature set (Woolley and Stanley, 2011)."""
def diff_feature_set(train_image, candidate_images):
    # c = candidate, t = target
    # d(c,t) = 1 −e^(−α|c−t|)
    # α=5 (modulation parameter)
    # error = average(d(c,t))
    # fitness = 1 − err(C, T)^2
    batch_size = candidate_images.shape[0]
    train_images = np.expand_dims(train_image, 0) # add "batch" dim
    h = candidate_images.shape[1]
    w = candidate_images.shape[2]
    alpha = 5
    if(len(candidate_images[0].shape) < 3):
        # greyscale
        [gradient0x, gradient0y] = np.gradient(train_images, axis=(1,2)) # Y axis, X axis
        [gradient1x, gradient1y] = np.gradient(candidate_images, axis=(1,2))
        
        c = np.array([train_images.reshape(1, h*w), gradient0x.reshape(1, h*w ), gradient0y.reshape(1, h*w)])
        t = np.array([candidate_images.reshape(batch_size, h*w), gradient1x.reshape(batch_size, h*w), gradient1y.reshape(batch_size, h*w)])
        
        c = np.transpose(c, (1, 0, 2))# batch dimension first
        t = np.transpose(t, (1, 0, 2))
        diffs = c-t
        diffs = diffs.reshape(batch_size, 3*h*w) # flatten diffs (3 diffs, height, width)
        D = 1 - np.exp(-alpha * np.abs(diffs))
        diffs = np.mean(D, axis =1)

    else:
        [Y0, X0, C0] = np.gradient(train_images, axis=(1, 2, 3)) # gradient over all axes (Y axis, X axis, color axis)
        [Y1, X1, C1] = np.gradient(candidate_images, axis=(1,2, 3))
        flat_dim = h*w*3 # flatten along all dimensions besides batch dim
        c = np.array([train_images.reshape(1, flat_dim), Y0.reshape(1, flat_dim ), X0.reshape(1, flat_dim), C0.reshape(1, flat_dim)])
        t = np.array([candidate_images.reshape(batch_size, flat_dim), Y1.reshape(batch_size, flat_dim), X1.reshape(batch_size, flat_dim), C1.reshape(batch_size, flat_dim)])
        c = np.transpose(c, (1, 0, 2)) # batch dimension first
        t = np.transpose(t, (1, 0, 2)) 
        
        diffs = np.abs(t-c)
        
        diffs = diffs.reshape(batch_size, 4*3*h*w) # flatten diffs (4 diffs, 3 color channels, height, width)
        D = 1 - np.exp(-alpha * np.abs(diffs))
        diffs = np.mean(D, axis =1)

    return diffs


class Net(nn.Module):
    def __init__(self, n_channels, height, width):
        super().__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(n_channels, 100, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 60, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.conv2(x)
        return x

    def linear_input_neurons(self, height, width):
        def size_after_relu(x):
            x = self.pool(F.tanh(self.conv1(x.float())))
            x = self.pool(F.tanh(self.conv2(x.float())))
            return x.size()
        size = size_after_relu(torch.rand(1, 1, height, width))
        m = 1
        for i in size:
            m *= i
        return int(m)



def diff_cnn_trained(train, candidates):
    batch_size = candidates.shape[0]
    in_channels = 3 if (len(train.shape)>2) else 1
    # trained_net = torch_models.regnet_y_400mf(pretrained=True)
    # trained_net = torch_models.mobilenet_v3_small (pretrained=True)
    # trained_net = torch_models.vgg16(pretrained=True)
    # trained_net = torch_models.alexnet(pretrained=True)
    # trained_net = torch_models.inception_v3(pretrained=True)
    trained_net = torch_models.efficientnet_b0(pretrained=True)

    # remove output layer:
    # trained_net = torch.nn.Sequential(*(list(trained_net.children())[:-2]))
    trained_net = torch.nn.Sequential(*(list(trained_net.children())[:-1]))

    if(in_channels < 3):
        # grayscale to rgb
        train = np.stack((train,)*3, axis=-1) 
        candidates = np.stack((candidates,)*3, axis=-1)
    
    # pytorch wants color channel as axis 0
    train = train.transpose(2, 0, 1)
    candidates = candidates.transpose(0, 3, 1, 2)
    train = np.expand_dims(train, 0) # convert to batch
    
    train = tensor(train.astype(np.float32))
    candidates = tensor(candidates.astype(np.float32))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train = train.to(device)
    candidates = candidates.to(device)
    trained_net = trained_net.to(device)

    # train = train.reshape((1, 3, 224, 224))
    # candidate = candidate.reshape((1, 3, 224, 224))

    # train = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]).forward(train)
    # candidate = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]).forward(candidate)

    trained_net.eval()
    with torch.no_grad():
        train = trained_net(train)
        candidates = trained_net(candidates)
    
    criterion = nn.MSELoss(reduction="none")
    losses = criterion(candidates, train)
    losses = losses.reshape((batch_size, losses.shape[1]*1*1))
    losses = torch.mean(losses, dim=1).cpu().detach().numpy()
    return losses 

    # train =     train.cpu().detach().numpy()
    # candidate = candidate.cpu().detach().numpy()
    # return np.max(np.abs(train - candidate))
    

def diff_cnn(train, candidates):
    train = np.expand_dims(train, axis=0) # turn image into batch
    
    in_channels = 3 if (len(train.shape)>3) else 1
    height = train.shape[1]
    width  = train.shape[2]
    batch_size = candidates.shape[0]
    
    if(width<15):
        # TODO not great
        width*=2
        height*=2
        train = resize(train, (1, height, width))
        candidates = resize(candidates, (batch_size, height, width))

    if in_channels > 1:
        # put color channel first
        train = train.transpose(0, 3, 1, 2)
        candidates = candidates.transpose(0, 3, 1, 2)
    else:
        train = np.expand_dims(train, axis=0) # add color dim
        candidates = np.expand_dims(candidates, axis=1) # add color dim
        
    train = tensor(train.astype(np.float32))
    candidates = tensor(candidates.astype(np.float32))

    net = Net(in_channels, height, width)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = train.to(device)
    candidates = candidates.to(device)
    net = net.to(device)
    train = net(train)
    candidates = net(candidates)

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # loss = criterion(candidate, train).cpu().detach().numpy()
    error_criterion = nn.MSELoss(reduction='none')
    losses = torch.mean(error_criterion(candidates, train), dim=(1,2,3)).detach().cpu().numpy() # compute loss for each image
    return losses

#Inertia tensor? https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.inertia_tensor
# perimeter? https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.perimeter
# shannon_entropy #https://scikit-image.org/docs/dev/api/skimage.measure.html#shannon-entropy   

def load_image(path, color_mode):
    '''Load an image to a Numpy array
    parameters:
    path: path of image to load
    
    returns:
    data: array of pixel data
    '''
    if(color_mode == 'L'):
        image = Image.open(path).convert(color_mode)
    else:
        image = Image.open(path).convert('RGB')
    
    data = np.asarray(image, dtype=np.float64)
    data = np.divide(data, 255.0)
    if(color_mode=='HSL'):
        data = rgb2hsv(data)
    
    return data

def normalize_0_to_1(image):
    image = np.divide(image, np.max(image))
    # image = np.multiply(255.0, image)
    # values_norm = [255.0 * (x - 0) / (np.max(image)-0) for x in image]
    return image

def normalize_neg1_to_1(image):
    min_value = np.min(image)
    max_value = np.max(image)
    image = np.subtract(np.divide(2*(image-min_value), max_value-min_value),1.0)
    return image

def normalize(image, lower, upper):
    min_value = np.min(image)
    max_value = np.max(image)
    image = (upper-lower) * (image - min_value)/(max_value-min_value) + lower
    return image

def picbreeder_normalize(image):
    # from picbreeder: function outputs range between [-1,1]. 
    # However, ink level is darker the closer the output is to zero. Therefore, an output of either -1 or 1 produces white.
    # image = normalize(image, -1.0, 1.0)
    # for p in image: 
    #     if(p==-1 or p == 1): p = 1
    return image

def show_sobel(img):
    plt.imshow(sobel_v(img))

def show_image(img, color_mode, ax = None, title=None):
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if(color_mode == 'L'):
        if(ax==None):
            plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    elif(color_mode == "HSL"):
        img = hsv2rgb(img)
        if(ax==None):
            plt.imshow(img, vmin=0, vmax=1)
        else:
            ax.imshow(img)
    else:
        if(ax==None):
            plt.imshow(img, vmin=0, vmax=1)
        else:
            ax.imshow(img)
    if title is not None:
        plt.title(title)
    # if(ax==None):
    #     plt.show()

def save_current_run_progress_image(img, color_mode="L", name="", directory="./current_run"):
    path = directory+"/img-"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path+=name if name!="" else str(time.time)
    path+=".png"
    if(color_mode == 'L'):
        plt.imsave(path, img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    elif(color_mode == "HSL"):
        img = clip01(img)
        img = hsv2rgb(img)
        plt.imsave(path, img, vmin=0, vmax=1)
    else:
        img = clip01(img) 
        plt.imsave(path, img, vmin=0, vmax=1)

def show_images(imgs, color_mode="L", titles=[], height=10, title=None):
    num_imgs = len(imgs)
    # plt.figure(figsize=(20, num_imgs //1.8))
    # plt.figure(figsize=(20, num_imgs*1))
    # columns = num_imgs//4
    # rows = 4 
    fig = plt.figure(figsize=(20, height))
    # fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for i, image in enumerate(imgs):
        # plt.subplot(rows, columns, i + 1)
        # plt.gca().set_title(f'{i}')
        # plt.tight_layout()
        ax = fig.add_subplot(num_imgs//5 +1, 5, i+1)
        if(len(titles)> 0):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"{i}")
        show_image(image, color_mode)
    if title is not None:
        plt.title(title)
    plt.show()

def show_target_and_trained(all_runs, train_path, color_mode, res=[-1,-1]):
    best = get_best_solution_from_all_runs(all_runs)[0]
    train_image = load_image(train_path, color_mode)
    if(-1 in res): res = train_image.shape
    start_end_imgs = [train_image, best.get_image(res[0],res[1], color_mode)]
    show_images(start_end_imgs, color_mode, titles=["Training Image", f"Result: {best.fitness:.3f}"])


def show_images_over_training(results, color_mode, n_shown=9, res = [100,100]):
    total_generations = len(results)
    SHOW_EVERY = math.ceil(total_generations/min(n_shown, total_generations))
    imgs = [results[i].get_image(res[0],res[1], color_mode) for i in range(0, total_generations, SHOW_EVERY)]
    show_images(imgs, color_mode,  titles=[f"gen{i}: {results[i].fitness:.3f}" for i in range(0, total_generations, SHOW_EVERY)])

def images_to_gif(imgs, name, fps = 1):
    imageio.mimsave('./saved_imgs/gifs/'+name, imgs, fps=fps)


def image_grid(array, color_mode, x_title, y_title):
    n_channels = 1
    if len(array.shape) > 4:
        # color
        n_channels = 3
    num_rows = array.shape[0]
    num_cols = array.shape[1]
    height = array.shape[2]
    width = array.shape[3]
    
    array = np.flip(np.copy(array), 0)
    
    img_grid = (array.reshape(num_rows, num_cols, height, width, n_channels)
              .swapaxes(1,2)
              .reshape(height*num_rows, width*num_cols, n_channels))
    
    result = np.array(np.copy(img_grid))
        
    # result = np.flip(result, 0)
    # for i, r in enumerate(result[0:,:,:]):
        # result[i] = np.flip(r)
        
    fig = plt.figure(figsize=(20., 20.), frameon=False)
    # result = np.flip(result, 1)
    plt.tick_params(direction='out', length=6, width=1, colors='r',
               grid_color='r', grid_alpha=0.2, labelsize=0)
    x=np.linspace(0, num_cols*width+width/2, num_cols+1)
    y=np.linspace(0, num_rows*height+height/2, num_rows+1)
    
    # x = [-1, width-.5, 2*width-.5, 3*width-.5]
    x = [i * width - .5 if i>0 else -1 for i in range(num_cols)]
    y = [i * height - .5 if i>0 else -1 for i in range(num_rows)]
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    
    if(color_mode == 'L'):
        plt.imshow(result, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    elif(color_mode == "HSL"):
        result = hsv2rgb(result)
        plt.imshow(result, vmin=0, vmax=1)
    else:
        plt.imshow(result, vmin=0, vmax=1)
    
    plt.show()

# Other distance functions to try:
def diff_l1norm(train, candidate):
    return sum(abs(train - candidate))

def diff_euclidean(train, candidate):
    diff = np.array(train) - candidate
    return np.sqrt(np.dot(diff, diff))

def diff_euclidean_square(train, candidate):
    diff = np.array(train) - candidate
    return np.dot(diff, diff)

def diff_euclidean_centered(train, candidate):
    diff = np.mean(train) - np.mean(candidate)
    return np.dot(diff, diff)

def diff_cosine(train, candidate):
    return 1 - float(np.dot(train, candidate)) / ((np.dot(train, train) * np.dot(candidate, candidate)) ** 0.5)
