

# GENERATION TO ONE ANOTHER DOMAIN: Audio ⟷ Image (Pytorch)
### GCT634-Final-Project
----------
# Model structure: Adversarial Conditional VAE(AC-VAE):



![model_sturcture](./figs/model_sturcture.png)

# Results:

## Audio to Image:

![A2I_output](./figs/A2I_output.png)



## Image to Audio:

![I2A_output](./figs/I2A_output.png)



## Visualization latent space:

<p float="left">
  <img src="./figs/A2I_visualization.png" width="500" />
  <img src="./figs/I2A_visualization.png" width="500" /> 

</p>



# Usage:

First, put dataset in ```<Code_path>/dataset/```

Dataset Link: https://www.cs.rochester.edu/~cxu22/d/vagan/

The results will save in ```<Code_path>/experiment/```

## Run Audio to Image:

```
python trainA2I.py --name <save_result_name>
```
## Run Image to Audio:

```
python trainA2I.py --name <save_result_name>
```

## Run Cross Modal Generation:

```
python train_CrossModal.py --name <save_result_name>
