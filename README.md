

## GENERATION TO ONE ANOTHER DOMAIN (A2I & I2A)
### GCT634-Final-Project
----------
# Model structure(AC-VAE):



![model_sturcture](./figs/model_sturcture.png)

# Results:

## Audio to Image:

![A2I_output](./figs/A2I_output.png)



## Image to Audio:

![I2A_output](./figs/I2A_output.png)



## Visualization latent space:

<img src="./figs/A2I_visualization.png" alt="A2I_visualization" style="zoom:45%;" /> <img src="C:\Users\rlgns\OneDrive\문서\GitHub\GCT634-Final-Project\figs\I2A_visualization.png" alt="I2A_visualization" style="zoom: 45%;" />



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
```
