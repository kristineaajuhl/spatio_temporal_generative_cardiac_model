# Spatio-temporal neural distance fields for conditional generative modeling of the heart
Repository linked to publication: "Spatio-temporal neural distance fields for conditional generative modeling of the heart" accepted for MICCAI 2024. 

Authors: Kristine Sørensen, Paula Diez, Jan Margeta, Yasmin El Youssef, Michael Pham, Jonas Jalili Pedersen, Tobias Kühl, Ole de Backer, Klaus Kofoed, Oscar Camara, and Rasmus Paulsen

contact: Kristine Sørensen - kajul@dtu.dk

![](https://github.com/kristineaajuhl/spatio_temporal_generative_cardiac_model/blob/main/normal_cropped.gif) ![](https://github.com/kristineaajuhl/spatio_temporal_generative_cardiac_model/blob/main/reconstruction_v2.png)

## Data Preparation
Data preparation code is coming up

To train the network a .npz file with coordinate-distance pairs for every shape at every time frame is required.
Below is an example-script preparing a set of pairs from a collection of surfaces. All surfaces are aligned to a common template, scaled with a common scale-factor to fit within the unit-sphere and the point-distance samples are sampled using the method from [NUDF](https://github.com/kristineaajuhl/NUDF).
The code runs on 16 cores to speed up the sampling time. 

```
python sample_distances_time.py
```

## Training
![](https://github.com/kristineaajuhl/spatio_temporal_generative_cardiac_model/blob/main/pipeline.png)
Training procedures are coming up

```
python train.py --exp_dir "Path/to/experiment/folder/" 
```

## Testing
Test procedures are coming up

```
python test.py --exp_dir "Path/to/experiment/folder/" --resume "epoch_to_test" --task "test_task"
```

Test tasks: 
- sequence_completion_training: Reconstruct the sequence from training set based on optimized latent vector (l. 66 specified which number in the training set to be reconstructud)
- sequence_completion_test: Reconstruct test sequences from single timestep
- sequence_generation: Generate unique sequence with specified demography (l. 158-160) specifies the demography)
- sequence_generation_similarconditions: Generate unique sequences matching the demographic of the test set

## Cite

```bibtex
@article{sorensen2024,
  title={Spatio-temporal neural distance fields for conditional generative modeling of the heart},
  author={Kristine Sørensen, Paula Diez, Jan Margeta, Yasmin El Youssef, Michael Pham, Jonas Jalili Pedersen, Tobias Kühl, Ole de Backer, Klaus Kofoed, Oscar Camara, and Rasmus Paulsen},
  journal={XXX},
  year={2024}
}
