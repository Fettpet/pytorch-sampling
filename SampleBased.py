# implementation of the sample based approch https://arxiv.org/abs/1511.05067

import torch.nn as nn
import torch
from Network.Blocks.Sampling import SamplingLayer
import configparser
import numpy as np
from os.path import join
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SampleBased(nn.Module):
    """
    This variable stores the binary potentials. The binary potential are a tensor of the following shape
    [nbNeigbors, nbClasses, nbClasses]
    """

    # The PCD methode needs the current labeling of all trainingsexamples
    current_labeling = None

    def __init__(
            self,
            net,
            nb_classes,
            neighborhood,
            train_mode="Jointly",
            debug_dir=None,
            sample_steps=1
    ):
        """
        Creates the sample based approach
        :param net: The network used to define the unaries.
        :param nb_classes: number of different segments
        :param neighborhood: define the neighbor hood. It is a list of distances. Each distance is a tuple [x, y].
        :param train_mode: With the train mode it is possible to set the trained parts. Possible values are 1. Jointly:
        Both, the CRF and CNN are trained jointly. 2. CRF: The CRF is trained. The values of the CNN are constant. 3.
        CNN: The CNN is trained. The CRF is constant.
        Starting at the current point the algorithm connects the current and the pixel x rows and y cols away.
        """
        super(SampleBased, self).__init__()

        self.net = net
        if train_mode.lower() == "crf":
            for param in self.net.parameters():
                param.requires_grad = False
        self.nb_classes = nb_classes
        self.neighborhood = neighborhood
        self.nb_neighbors = len(neighborhood)
        self.samples = {}

        self.debug_dir = debug_dir
        self.counter_debug = 0

        if train_mode.lower() == "cnn":
            requires_grad = False
        else:
            requires_grad = True

        # main diagonal is 1, rest 0

        self.binary_potentials = torch.nn.Parameter(
            torch.zeros(
                self.nb_neighbors,
                self.nb_classes,
                self.nb_classes,
                device="cuda:0"
            ),
            requires_grad=requires_grad
        )

        self.sampler = SamplingLayer(
            neighborhood=neighborhood
        )
        self.steps = 0
        self.sample_steps = sample_steps

    def forward(
            self,
            x,
            name
    ):
        self.steps += 1
        unaries = self.net(x)
        if name in self.samples.keys():
            sample = self.samples[name]
        else:
            shape = list(unaries.shape)
            shape[1] = 1

            sample = np.random.randint(
                0,
                self.nb_classes,
                size=shape
            ).squeeze(axis=1)

        unaries = unaries.cuda(0)
        result, next_sample = self.sampler(
            unaries=unaries,
            binaries=self.binary_potentials,
            sample=torch.Tensor(sample).int(),
            sample_steps=self.sample_steps
        )

        next_sample_numpy = next_sample.cpu().detach().numpy()

        self.samples[name] = next_sample_numpy

        if self.debug_dir is not None and self.steps % 10 == 0:
            # store the image x
            img_name = join(self.debug_dir, "oct.png")
            img = x.cpu().detach().numpy()
            img = np.swapaxes(
                np.swapaxes(
                    img,
                    0,
                    2
                ),
                1,
                3
            )
            cv2.imwrite(
                img_name,
                np.squeeze(img)
            )
            # store the unaries
            img_name = join(self.debug_dir, "unaries.png")
            img = self.predict_to_image(torch.argmin(unaries, dim=1))
            cv2.imwrite(
                img_name,
                np.squeeze(img)
            )
            # store the result
            img_name = join(self.debug_dir, "result.png")
            img = self.predict_to_image(result)
            cv2.imwrite(
                img_name,
                np.squeeze(img)
            )

            # previous sample
            img_name = join(self.debug_dir, "sample.png")
            img = self.predict_to_image(sample)
            cv2.imwrite(
                img_name,
                np.squeeze(img)
            )

        return result

    def compare_sample_result(
            self,
            sample,
            result
    ):
        sample = self.predict_to_image(sample)[0]
        result = self.predict_to_image(result.detach().cpu().numpy())[0]
        filename = join(self.debug_dir, str(self.counter_debug) + ".png")
        self.counter_debug += 1
        plt.subplot(
            2,
            1,
            1
        )

        plt.imshow(sample)
        plt.subplot(
            2,
            1,
            2
        )

        plt.imshow(result)

        plt.savefig(filename)
        plt.clf()

    def predict_to_image(
            self,
            predicted
    ):
        """
        transform the label into a image with 3 color channels
        :param predicted:
        :return:
        """
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.cpu().detach().numpy()

        if len(predicted.shape) == 4:
            if predicted.shape[1] > 1:
                labels = np.argmax(
                    predicted,
                    axis=1
                )
            else:
                labels = np.squeeze(
                    predicted,
                    axis=1
                )
        else:
            labels = predicted

        # get the size of the output
        result = np.zeros([labels.shape[0], 3, labels.shape[1], labels.shape[2]], dtype=np.int)
        pixel_values = [[(i * 481) % 255, (i * 127) % 255, (i * 343) % 255] for i in range(self.nb_classes)]
        pixel_values[0:5] = [[0, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 255, 255]]

        for i in range(self.nb_classes):
            result[:, 0, :, :][labels == i] = pixel_values[i][0]
            result[:, 1, :, :][labels == i] = pixel_values[i][1]
            result[:, 2, :, :][labels == i] = pixel_values[i][2]
        result = np.swapaxes(
            np.swapaxes(
                result,
                0,
                2
            ),
            1,
            3
        )
        return result

    def crf_dict(self):
        return {"binary_potentials": self.binary_potentials}

    def cnn_dict(self):
        return self.net.state_dict()

    def cnn_parameters(self):
        return self.net.parameters()

    def crf_parameters(self):
        return [
            self.binary_potentials
        ]

    def load_parameter(
            self,
            cnn_path=None,
            crf_path=None
    ):
        if cnn_path is not None:
            self.net.load_state_dict(torch.load(cnn_path))
        if crf_path is not None:
            self.binary_potentials = torch.load(crf_path)["binary_potentials"]
            print(self.binary_potentials)


def make_sample_based_config(
        dataset,
        config_file,
        use_gpu=False,
        train_mode="Jointly",
        inference=False,
        network_type=None
):
    output_channels = dataset.get_number_labels()

    config = configparser.ConfigParser()
    config.read(config_file)

    # the import need to be here to avoid recursion
    import Network.Factory as Factory
    if inference:
        config = config["Inference"]
    else:
        config = config["Train"]

    if network_type is None:
        network_type = config["network"]

    network = Factory.network_factory(
        dataset=dataset,
        network_type=network_type,
        use_config=True,
        use_gpu=use_gpu
    )
    dataset_type = dataset.label_type

    dataset.label_type = "Label"

    dataset.label_type = dataset_type

    # set the sample steps
    sample_steps = int(config["sample_steps"])

    # transform the string to a list of tuples
    neighborhood = config["neighborhood"]

    x = neighborhood.replace("[", "").replace("]", "")
    neighborhood = [[int(y), int(z)] for y, z in zip(x.split(",")[0::2], x.split(",")[1::2])]

    return SampleBased(
        net=network,
        nb_classes=output_channels,

        neighborhood=neighborhood,
        train_mode=train_mode,
        sample_steps=sample_steps
    )


def make_sample_based(
        dataset,
        network_type="testing",
        train_mode="Jointly",
        neighborhood=[[1, 0], [-1, 0], [0, 1], [0, -1]],
        use_gpu=True,
        inference=False
):

    output_channels = dataset.get_number_labels()

    # the import need to be here to avoid recursion
    import Network.Factory as Factory

    network = Factory.network_factory(
        dataset=dataset,
        network_type=network_type,
        use_config=False
    )

    if inference:
        sample_steps=100
    else:
        sample_steps=1

    return SampleBased(
        net=network,
        nb_classes=output_channels,
        neighborhood=neighborhood,
        train_mode=train_mode,
        sample_steps=sample_steps
    )