# Learning Type-Aware Embeddings for Fashion Compatibility

fashion-compatibility contains a [PyTorch](http://pytorch.org/) implementation for our [paper](https://arxiv.org/pdf/1803.09196.pdf).  If you find this code or our dataset useful in your research, please consider citing:

    @inproceedings{VasilevaECCV18FasionCompatibility,
    Author = {Mariya I. Vasileva and Bryan A. Plummer and Krishna Dusad and Shreya Rajpal and Ranjitha Kumar and David Forsyth},
    Title = {Learning Type-Aware Embeddings for Fashion Compatibility},
    booktitle = {ECCV},
    Year = {2018}
    }

This code was tested on an Ubuntu 16.04 system using Pytorch version 0.1.12.  It is based on the [official implementation](https://github.com/andreasveit/conditional-similarity-networks) of the [Conditional Similarity Networks paper](https://arxiv.org/abs/1603.07810).


## Usage

You can download the Polyvore Outfits dataset including the splits and questions for the compatibility and fill-in-the-blank tasks from [here (6G)](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing).  The code assumes you unpacked it in a dictory called `data`, but if you choose a different directory simply set the `--datadir` argument.  You can see a listing and description of the model options with:

```sh
    python main.py --help
```

## Using a pre-trained model

We have provided a pre-trained model for the nondisjoint data split which you can download [here (11M)](https://drive.google.com/file/d/1JrRgM_EaLQqLw1CNjM65XnTm9rZyLRgj/view?usp=sharing).  This model learns diagonal projections from the general embedding to a type-specific compatibility space which is L2 normalized after appying the projection.  You can test this model using:

```sh
    python main.py --test --l2_embed --resume runs/nondisjoint_l2norm/model_best.pth.tar
```

This code includes some minor modifications resulting in better perfromance than the version used for our camera ready.  For example, our pre-trained model should provide a compatibility AUC of 0.88 and fill-in-the-blank accuracy of 57.6, which is a little better than the 0.86 AUC/55.3 accuracy for our best model reported in our paper.

## Training a new model

To train the pre-trained model above we used the following command.

```sh
    python main.py --name {your experiment name} --learned --l2_embed
```

By default the code outputs the results on the test set after training. However, if you wanted to re-run the test for many settings you have to use the same flags during testing as you had during training.  For example, if you trained with the `--use_fc` to train fully connected type-specific embeddings rather than a (diagonal) mask, at test time you would use:

```sh
   python main.py --test --use_fc --resume runs/{your experiment name}/model_best.pth.tar
```

