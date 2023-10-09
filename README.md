# BayesTune: Bayesian Sparse Deep Model Fine-tuning (NeurIPS 2023) -- A Bayesian Approach to PEFT

Deep learning practice is increasingly driven by powerful foundation models (FM), pre-trained at scale and then fine-tuned for specific tasks of interest. A key property of this workflow is the efficacy of performing sparse or parameter-efficient fine-tuning, meaning that by updating only a tiny fraction of the whole FM parameters on a downstream task can lead to surprisingly good performance, often even superior to a full model update. However, it is not clear what is the optimal and principled way to select which parameters to update. Although a growing number of sparse fine-tuning ideas have been proposed, they are mostly not satisfactory, relying on hand-crafted heuristics or heavy approximation. In this paper we propose a novel Bayesian sparse fine-tuning algorithm: we place a (sparse) Laplace prior for each parameter of the FM, with the mean equal to the initial value and the scale parameter having a hyper-prior that encourages small scale. Roughly speaking, the posterior means of the scale parameters indicate how important it is to update the corresponding parameter away from its initial value when solving the downstream task. Given the sparse prior, most scale parameters are small a posteriori, and the few large-valued scale parameters identify those FM parameters that crucially need to be updated away from their initial values. Based on this, we can threshold the scale parameters to decide which parameters to update or freeze, leading to a principled sparse fine-tuning strategy. To efficiently infer the posterior distribution of the scale parameters, we adopt the Langevin MCMC sampler, requiring only two times the complexity of the vanilla SGD. Tested on popular NLP benchmarks as well as the VTAB vision tasks, our approach shows significant improvement over the state-of-the-arts (e.g., 1% point higher than the best SOTA when fine-tuning RoBERTa for GLUE and SuperGLUE benchmarks).  For the details, please see [1].

<p align="center">
  <img align="middle" src="./figs/sgld_eq.png"/>
</p>

---

## Set up environment
```
>> conda env create -f environment.yml
>> conda activate recmixvae
>> conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Usage examples
We provide the demo codes that run on the MNIST dataset. It should be straightforward to modify the codes for other datasets (e.g., "datasets.py"). 
For the MNIST dataset, it can automatically download the dataset, and place it in the data folder you have specified. 

The first step is to train the VAE model, where the learned VAE encoder model will be used to initialise our mixture inference component models. 
To train the VAE model,
```
>> python train_vae.py
```
You can consult the source code for alternative training options (command line arguments). During training, the model files will be saved in the checkpoint folder at every training epoch. 

Then you can evaluate the trained VAE model using "eval_vae.py" with the model's epoch number. 
For instance, to evaluate the likelihood on the test data wrt the VAE model after the training epoch 999 (perhaps chosen by validation performance),
```
>> python eval_vae.py --ckpt_load_epoch 999 --task loglik
```
Or, to measure the test inference time,
```
>> python eval_vae.py --ckpt_load_epoch 999 --task time
```

Now, we train the recursive mixture. Among other options, the path to the trained VAE model and the mixture order (the number of mixture components) are the most important. For instance, with the epoch-999th model as initialization and the mixture order 5, we can start the mixture training by:
```
>> python train_rme.py --num_comps 5 --init_vae_path ./checkpoints/MNIST/VAE/MNIST.VAE.0/999.pkl
```

Similar to VAE, you can evaluate the trained mixture model using "eval_rme.py". 
For instance, to evaluate the likelihood on the test data wrt the mixture model after the training epoch 4 (perhaps chosen by validation performance),
```
>> python eval_rme.py --ckpt_load_epoch 4 --task loglik
```
Or, to measure the test inference time,
```
>> python eval_rme.py --ckpt_load_epoch 4 --task time
```

## Acknowledgements
This code is built from: [https://github.com/yookoon/VLAE](https://github.com/yookoon/VLAE)


## References
[1] Kim, Minyoung and Hospedales, Timothy, "BayesTune: Bayesian Sparse Deep Model Fine-tuning", *Advances in Neural Information Processing Systems.* 2023.


## Citation
If you found this library useful in your research, please cite:
```
@article{kim_hospedales_neurips2023,
  title={{BayesTune: Bayesian Sparse Deep Model Fine-tuning}},
  author={Kim, Minyoung and Hospedales, Timothy},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

