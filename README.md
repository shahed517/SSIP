# SSIP
<div align="justify">
This repository contains the code for the ECE-60131 project : "A diffusion-guided framework for speech synthesis from neural data". 
This project aims to generate mel spectrograms directly from stereoEEG signals using diffusion models combined with Diffusion Posterior Sampling (DPS). A pretrained Whisper ASR encoder provides the likelihood guidance, helping steer the diffusion process toward acoustically intelligible mels consistent with the input brain activity.
</div>

## Methodology
<div align="justify">
Speech synthesis from neural recordings (stereoEEG/ECoG) is a challenging problem in neuroscience which has drawn researchers from both the Neuroscience and Machine Learning communities in recent years. As depicted in the workflow diagram, a discriminative learning approach can be adopted, however due to a lack of training data, producing intelligible speech from neural data is incredibly difficult. The problem is further pronounced when the workflow expects a model to learn structured speech from mere low-frequency brain recordings. 
</div>

<center>
<img src="https://github.com/shahed517/SSIP/blob/main/brain_to_voice_illustration.png" 
     alt="Speech Synthesis Workflow" 
     width="600" 
     height="500">
</center>

<div align="justify">
The problem can be substantially alleviated if we take help from a generative prior that has been trained on thousands of hours worth of speech data. We adopt a generative approach to speech synthesis, where we decompose the problem of speech synthesis as a sampling from a posterior distribution which requires the knowledge of 1) a diffusion model trained on mel spectrograms, and 2) a likelihood model that maps speech to mel spectrograms. And since we use a pretrained diffusion model, the current methodology only requires us to train a likehood model $ p(X_{eeg}|X_{mel}) $. Then the Diffusion Posterior Sampling (<i>Chung et.al.</i>, ICLR 23) approach is used to estimate the likelihood gradients at different timesteps of the diffusion sampling procedure.
</div>

## Dataset
<p>I have used the ECoG dataset made publicly available by Flinkerlab. The first three cells of the <i>finkerlab_likelihood.ipynb</i> notebook contains code to process the data as required. You can donwload the dataset from <a href="https://data.mendeley.com/datasets/fp4bv9gtwk/2">here</a></p>

## Reproduce Results
<p>You can install the dependencies from the <i>requirements.txt</i> file using the following: 
```bash
pip install -r requirements.txt
```
</p>

Key ideas are summarized as follows:
* Train an unconditional diffusion model for mel spectrograms.
* Use DPS to condition the generation on stereoEEG by maximizing Whisper-based likelihood.
* At inference, mel spectrograms are gradually denoised while being pushed toward features consistent with the stereoEEG input.

