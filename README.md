<img width="1167" alt="image" src="https://github.com/mojjaf/TransformerGAN-Attenuation-and-Scatter-Correction-in-Brain-PET/assets/55555705/d3129cf2-b872-41cb-b324-1d386f113e22">


This GitHub project introduces a generative context-aware deep learning framework designed to directly produce photon attenuation and scatter corrected (ASC) PET images from non-attenuation and non-scatter corrected (NASC) images. The framework employs conditional generative adversarial networks (cGAN), trained on either single-modality NASC or multi-modality NASC+MRI input data. Four cGAN models, including Pix2Pix, attention-guided cGAN (AG-Pix2Pix), vision transformer cGAN (ViT-GAN), and shifted window transformer cGAN (Swin-GAN), are designed and evaluated using retrospective 18F-fluorodeoxyglucose (18F-FDG) full-body PET images from 33 subjects. The unique strength of this work lies in the gold standard provided by each patient undergoing both PET/CT and PET/MRI scans on the same day, enabling a comprehensive investigation of ASC in PET imaging.


If you use this code for your research, please cite our paper.

M. Jafaritadi, E. Anaya, G. Chinn and C. S. Levin, "Multi-Modal Generative Vision Transformer for Attenuation and Scatter Correction in Brain PET," 2023 IEEE Nuclear Science Symposium, Medical Imaging Conference and International Symposium on Room-Temperature Semiconductor Detectors (NSS MIC RTSD), Vancouver, BC, Canada, 2023, pp. 1-1, doi: 10.1109/NSSMICRTSD49126.2023.10338712.
