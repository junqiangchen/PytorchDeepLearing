import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import Tensor
import SimpleITK as sitk


def plot_result(model_dir, H_train, H_validation, H_train_name, H_validation_name, labelname):
    PLOT_PATH = os.path.sep.join([model_dir, H_train_name + "_" + H_validation_name + "plot.png"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H_train, label=H_train_name)
    plt.plot(H_validation, label=H_validation_name)
    plt.title(H_train_name + "," + H_validation_name + " on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel(labelname)
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)


def save_images3d(pdmask: Tensor, gtmask: Tensor, size, path, pixelvalue=255.):
    images = pdmask.detach().cpu().squeeze().numpy()
    gtmask = gtmask.detach().cpu().squeeze().numpy()
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image
    cv2.imwrite(path + "pdmask.bmp", np.clip(merge_img * pixelvalue, 0, 255).astype('uint8'))

    merge_mask = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(gtmask):
        i = idx % size[1]
        j = idx // size[1]
        merge_mask[j * h:j * h + h, i * w:i * w + w] = image
    cv2.imwrite(path + "gtmask.bmp", np.clip(merge_mask * pixelvalue, 0, 255).astype('uint8'))


def save_images2d(pdmask: Tensor, gtmask: Tensor, path, pixelvalue=255.):
    pdmask = pdmask.detach().cpu().squeeze().numpy()
    gtmask = gtmask.detach().cpu().squeeze().numpy()
    if np.max(gtmask) == 1:
        pdmask[pdmask > 0.5] = 1
        pdmask[pdmask < 0.5] = 0
    cv2.imwrite(path + "pdmask.bmp", np.clip(pdmask * pixelvalue, 0, 255).astype('uint8'))
    cv2.imwrite(path + "gtmask.bmp", np.clip(gtmask * pixelvalue, 0, 255).astype('uint8'))


def save_images2dregression(src: Tensor, pdmask: Tensor, gtmask: Tensor, path, pixelvalue=255.):
    src = src.detach().cpu().squeeze().numpy()
    pdmask = pdmask.detach().cpu().squeeze().numpy()
    gtmask = gtmask.detach().cpu().squeeze().numpy()
    if np.max(gtmask) == 1:
        pdmask[pdmask > 0.5] = 1
        pdmask[pdmask < 0.5] = 0
    cv2.imwrite(path + "src.png", np.clip(src * pixelvalue, 0, 255).astype('uint8'))
    cv2.imwrite(path + "pdmask.png", np.clip(pdmask * pixelvalue, 0, 255).astype('uint8'))
    cv2.imwrite(path + "gtmask.png", np.clip(gtmask * pixelvalue, 0, 255).astype('uint8'))


# show CNN feature visulation
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """

    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output, prob, label = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = label
            print(f"category id: {target_category}")

        target_category = [target_category] * input_tensor.size(0)
        assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
