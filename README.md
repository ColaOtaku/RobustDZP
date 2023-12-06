# RobustDZP
The official code repository for "Uncertainty-aware Decisioning for Robust Delivery Zone Partition in Last-mile Logistics"

### Code Description

- prediction
  - data
  - results 
  - common.py
    - for dataset generation; experimental and model hyperparameter settings 
  - model.py
    - for model implements
  - train.py
    - for model training
- sampling
  - sampling_module.ipynb
- partition
  - partition.ipynb 


### Data Description
Due to the company's data protection policy, we can not release the data. Instead, we offer the description of the data here:

**data/attendance**
> courier attendance
> 
> character: binary
> 
> shape: N_couriers,T_length
>
> sample:

| Couriers_ID\Time | 20220801 | 20220802 |
|------------------------------------------|----------|----------|
| 0 | 1 [on-duty]     | 0  [off-duty]    |

**data/aoi_order**
> aoi ID; aoi-level daily volume
> 
> character: ranging from [0,400];  average 14
> 
> shape: N_AOIs,T_length + 1
>
> sample: 

| AOI_ID\Time | 20220801 | 20220802 |
|------------------------------------------|----------|----------|
| 7262716338xxxxxxxx734D6C | 214      | 245      |

**data/aoi_poi_infos**
> aoi contextual
>
> sample:

| AOI_ID\POI category | Retail | Residence | ...| 
|------------------------------------------|----------|----------|----------|
| 7262716338xxxxxxxx734D6C | 0      | 2      | |
