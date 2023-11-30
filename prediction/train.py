from common import *
from model import *

criterion = nn.L1Loss()
mse = nn.MSELoss()

DP = DataProcessing()
data =DP.generate_seq_data(mode=args.poi_type) # order,poi,t_cov
DG = DatasetGenerator(*data,args.ratio,args.hist_len,args.pred_len,args.step,args.batch_size, args.mask_event)
trainset,testset = DG.generate_dataset()
device = torch.device(args.device)

args.nn_params  = get_nn_params('OCAP')

labels = torch.zeros(299) # only one AOI group
model = OCAP(args.nn_params).to(args.device)
optims = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(args.epoch+1):
    for data in trainset:
        data = [item.to(device) for item in data]
        lookback,y,dynamic_feature,static_feature,hist_atten,real_atten = data
        optims.zero_grad()
        of,af,oa,aa = model(lookback,dynamic_feature,labels)
        loss = args.nn_params['lambda'][0]*criterion(oa,y) + args.nn_params['lambda'][1]*criterion(aa,real_atten) + args.nn_params['lambda'][2]*criterion(of,lookback) + args.nn_params['lambda'][3]*criterion(af,hist_atten) 
        loss.backward()
        optims.step()
    
    if epoch>=int(args.epoch *0.7):
        with torch.no_grad():
            for data in testset:
                data = [item.to(device) for item in data]
                lookback,y,dynamic_feature,static_feature,hist_atten,real_atten = data
                _,_,order,atten = model(lookback,dynamic_feature,labels)
            pred_order = DP._order_ch_denormalize(order.cpu())
            real_order = DP._order_ch_denormalize(y.cpu())

            print(f"{epoch} Order Loss: {COLOR_RED}{mse(pred_order,real_order).item():.06f}{COLOR_RESET}")
            print(f"{epoch} Order MAPE: {COLOR_RED}{mape(pred_order,real_order,0.255):.06f}{COLOR_RESET}")

            pred_atten = DP._atten_ch_denormalize(atten.cpu())
            real_atten = DP._atten_ch_denormalize(real_atten.cpu())

            print(f"Atten Loss: {COLOR_GREEN}{mse(pred_atten,real_atten).item():.06f}{COLOR_RESET}")
            print(f"Atten Loss: {COLOR_GREEN}{mape(pred_atten,real_atten,0).item():.06f}{COLOR_RESET}")

            # torch.save(model.state_dict(), 'results/ckpt/'+args.model+'_'+str(epoch)+'.pth')