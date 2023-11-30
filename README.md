# RobustDZP
The official code repository for "Uncertainty-aware Decisioning for Robust Delivery Zone Partition in Last-mile Logistics"



### Data Description
Due to the company's data protection policy, we can not release the data. Instead, we offer the description of the data here:

**data/attendance.npy**
> courier attendance
> character: binary
> shape: N_couriers,T_length

**data/aoi_order.csv**
> aoi ID; aoi-level daily volume
> character: ranging from [0,400];  average 14
> shape: N_AOIs,T_length + 1 
