# SSIP

This repository contains the code for the ECE-60131 project : "A diffusion-guided framework for speech synthesis from neural data". 
This project aims to generate mel spectrograms directly from stereoEEG signals using diffusion models combined with Diffusion Posterior Sampling (DPS). A pretrained Whisper ASR encoder provides the likelihood guidance, helping steer the diffusion process toward acoustically intelligible mels consistent with the input brain activity.

![Caption describing the figure](https://github.com/shahed517/SSIP/blob/main/brain_to_voice_illustration.png)

Key ideas are summarized as follows:
* Train an unconditional diffusion model for mel spectrograms.
* Use DPS to condition the generation on stereoEEG by maximizing Whisper-based likelihood.
* At inference, mel spectrograms are gradually denoised while being pushed toward features consistent with the stereoEEG input.

