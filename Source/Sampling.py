import torch
import torch.nn as nn
import Sampling_gpu
import numpy as np


class Sampling(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            unaries,
            binaries,
            sample,
            neighborhood,
            sample_steps,
            use_debug=False
    ):
        """
        This is the forward step of the sampling. The idea is described in https://arxiv.org/abs/1511.05067
        We use gibs sample
        :param ctx: A container for storage.
        :param unaries: The unaries. This is a tensor with the shape [nb_examples, nb_classes, rows, cols]
        :param binaries: This is a tensor with shape [len(neighborhood), nb_classes, nb_classes]
        :param sample: The starting point of the sampling. It must have the shape [nb_examples, rows, cols]
        :param neighborhood: The neighborhood of a single pixel.
        :param sample_steps: How many steps are sampled
        :param use_debug: Set to true to get additional debugging informations.
        :return: The sampled image.
        """
        # some checks

        assert unaries.shape[2] == sample.shape[1]
        assert unaries.shape[3] == sample.shape[2]
        assert unaries.shape[0] == sample.shape[0]

        if torch.cuda.is_available():
            neighborhood = [x for l in neighborhood for x in l]
            unaries = unaries.cuda(0)
            binaries = binaries.cuda(0)
            sample = sample.cuda(0)

            if use_debug:
                # at::Tensor const & unaries,
                # at::Tensor const & binaries,
                # at::Tensor & sample,
                # std::vector < int > neighborhood,
                # int const sample_steps
                result, sample, random_number_tensor, min_value_tensor, energies_tensor, probs_tensor = Sampling_gpu.forward_debug(
                        unaries,
                        binaries,
                        sample,
                        neighborhood,
                        sample_steps
                    )
                return result, sample, random_number_tensor, min_value_tensor, energies_tensor, probs_tensor
            else:
                sample_result, next_sample = Sampling_gpu.forward(
                    unaries,
                    binaries,
                    sample,
                    neighborhood,
                    sample_steps
                )

                ctx.save_for_backward(
                    next_sample,
                    unaries,
                    binaries
                )
                ctx.neighborhood = neighborhood

                return sample_result, next_sample

        else:
            raise ValueError

    @staticmethod
    def backward(
            ctx,
            labels,
            next_sample
    ):
        """

        The backward has the form -[yd = y^] + [y' = y^]. This is the equation for the unaries. yd is the train example,
        y^ is the label and y' is the prediction. We calculate -1 to the label yd and +1 to the label y'.
        The binaries are more complicate. The error is summed over all possible possitions. Lets assume a singe pixel
        (r, c). We use the second connection which is (r+1, c). The pixel (r, c) has the label 1 and (r+1,c) = 2 in the
        trainingsdata. The calculation gives pixel (r, c) the label 3 und (r+1, c) = 2.

        The error for this particular pixel is
        binary_grad[2][1][2] += -1
        binary_grad[2][3][2] += 1

        A problem are the labels. Currently my algorithm works with labels like [batch][Label][rows][cols] which gives
        some proberbility for each label. The labels which are used here have the shape [batch][rows][cols]. I need a
        transformation to do this. A way to do this is to adapt the base reader.
        :param labels: shape [batch][rows][cols].
        :param next_sample: not needed
        :return:
        """

        # read some variables
        sample, unaries, binaries = ctx.saved_variables
        labels = torch.argmax(labels, dim=1)
        sample_numpy = sample.cpu().detach().numpy()
        labels_numpy = labels.cpu().detach().numpy()
        if torch.cuda.is_available():

            unary_grad, binary_grad = Sampling_gpu.backward(
                unaries,
                binaries,
                sample.int(),
                labels.int(),
                ctx.neighborhood
            )

            return unary_grad, binary_grad, torch.zeros_like(sample), None, None
        else:
            raise ValueError


class SamplingLayer(nn.Module):

    def __init__(
            self,
            neighborhood
    ):
        super().__init__()
        self.neighborhood = neighborhood

    def forward(
            self,
            unaries,
            binaries,
            sample,
            sample_steps
    ):
        return Sampling.apply(
            unaries,
            binaries,
            sample,
            self.neighborhood,
            sample_steps
        )
