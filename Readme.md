### Simulating Crop Yield Prediction with Minimal Samples Using Synthetic Data Generation

For this task, I employ Downstream Feedback GAN (DSF-GAN), a generative adversarial network architecture based on a conditional sampling GAN, aimed to generate increased utility synthetic samples which works especially well for small datasets. To test and evaluate this method, I propose (a) simulating a small dataset environment, by under-sampling a small percentage of a full dataset, (b) training the DSF-GAN on the available subset, (c) generating a sufficient number of synthetic samples to enhance the prediction power of the downstream model, and (d) evaluate performance of the model trained on the augmented dataset, with the one trained on the full dataset.

To use this repo:
1. please clone the DSF-GAN repo (https://github.com/oriel9p/downstream-task-feedback-GAN)
2. replace `main.py` with `experiments.py`.
3. Place crop yield dataset csv in the datasets folder
4. Select number of epochs and batch size
5. Run the script