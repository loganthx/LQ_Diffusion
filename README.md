# LQ_Diffusion
Sampling Liquid Crystal Textures with Denoising Diffusion Probabilistic Models

Using Improved Diffusion as benchmark, we tested on our novel dataset liquid crystals. We discovered that the linear schedules lead to better losses later in training for our dataset.

Generated Textures (linear schedule):
![Generated Textures (linear schedule)](results/linear.png)

Generated Textures (cosine schedule):
![Generated Textures (cosine schedule)](results/cos.png)

Generated Textures (logistic schedule):
![Generated Textures (logistic schedule)](results/logistic.png)

