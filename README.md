<div align="justify">
This repository contains the code for the ECE-60131 project : "A diffusion-guided framework for speech synthesis from neural data". 
This project aims to generate mel spectrograms directly from ECoG signals using diffusion models combined with Diffusion Posterior Sampling (DPS). A pretrained Whisper ASR encoder provides the likelihood guidance, helping steer the diffusion process toward acoustically intelligible mels consistent with the input brain activity.
</div>

## Methodology
<div align="justify">
Speech synthesis from neural recordings (stereoEEG/ECoG) is a challenging problem which has drawn researchers from both the Neuroscience and Machine Learning research communities in recent years. As depicted in the workflow diagram, a discriminative learning approach can be adopted; however, due to a lack of training data, producing intelligible speech from neural data is incredibly difficult. The problem is further pronounced when the workflow expects a model to learn structured speech from mere low-frequency brain recordings. 
</div>
<br>
<center>
<img src="https://github.com/shahed517/SSIP/blob/main/brain_to_voice_illustration.png" 
     alt="Speech Synthesis Workflow" 
     width="600" 
     height="500">
</center>

<div align="justify">
The problem can be substantially simplified if we take help from a generative prior that has been trained on thousands of hours worth of speech data. We, therefore, adopt a generative approach to speech synthesis, where we decompose the problem of speech synthesis as a sampling from a posterior distribution which requires the knowledge of 1) a diffusion model trained on mel spectrograms, and 2) a likelihood model that maps speech to mel spectrograms. And since we use a pretrained diffusion model, the current methodology only requires us to train a likehood model p(X<sub>eeg</sub> | X<sub>mel</sub>). Then the <i>Diffusion Posterior Sampling</i> (<i>Chung et.al.</i>, ICLR 23) approach is used to estimate the likelihood gradients at different timesteps of the diffusion sampling procedure.
<br>
The end result is that during inference, mel spectrograms are gradually denoised while being pushed toward features consistent with the ECoG input. The project also explores the option to incorporate a Whisper pretrained encoder to learn the likelihood model.

</div>

## Dataset
<p>The ECoG dataset (one subject only) used in this study was made publicly available by <i>Chung et.al.</i> from their recent study from 2023. The first three cells of the <i>finkerlab_likelihood.ipynb</i> notebook contains code to process the data as required. You can donwload the dataset from <a href="https://data.mendeley.com/datasets/fp4bv9gtwk/2">here</a></p>

## Reproduce Results
You can install the dependencies from the <i>requirements.txt</i> file using the following: 
```bash
pip install -r requirements.txt
```
The code to generate the weights for the likelihood model are in the <i>finkerlab_likelihood.ipynb</i> notebook, while the code to generate samples for any arbitrary ECoG from the dataset is in the <i>diffusion_model.ipynb</i> notebook. The pretrained weights for both the likelihood model and unconditional diffusion model can be found in the /checkpoints directory of the repository.

## References
1. Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. (2022). Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687.
2. Chen, X., Wang, R., Khalilian-Gourtani, A., Yu, L., Dugan, P., Friedman, D., ... & Flinker, A. (2024). A neural speech decoding framework leveraging deep learning and speech synthesis. Nature Machine Intelligence, 6(4), 467-480.

