import torch
def map_twofold_to_swinunetr_swinvit(src_sd):
    
    
    WANTED_PREFIXES = (
    "module.swinViT.", "module.encoder1.", "module.encoder2.", "module.encoder3.", "module.encoder4.", "module.encoder10.",
    "module.decoder1.", "module.decoder2.", "module.decoder3.", "module.decoder4.", "module.decoder5.", "module.out."
)
    change_count=0
    mapped={}  
    for k, v in src_sd.items():
        if k.startswith("module.encoder.net"):
            
            nk = "module.swinViT."+k[len("module.encoder.net."):]        
        elif k.startswith("module.encoder."):
            nk = "module." + k[len("module.encoder."):]

        elif k.startswith("module.recon.decoder"):
            nk= "module." + k[len("module.recon."):]
        
        elif k.startswith("module.recon.out"):
            nk="module." + k[len("module.recon."):]    
        else:
            nk = k

        if any(nk.startswith(p) for p in WANTED_PREFIXES):
            mapped[nk] = v
            change_count += 1

    return mapped


def load_2fold_swin(swinunetr_model, path, rank):
    
    
    obj=torch.load(path,map_location="cpu")
    if "model_state" in obj:
        src_sd = obj.get("model_state",obj)
    else:
        src_sd = obj.get("model",obj)
    mapped = map_twofold_to_swinunetr_swinvit(src_sd)
    tgt_sd=swinunetr_model.state_dict()
    
    loadable={}
    shape_mismatch=[]
    name_miss_in_target=[]
    for k,v in mapped.items():
        if k in tgt_sd:           
            if tgt_sd[k].shape == v.shape:
                loadable[k]=v
            else:
                shape_mismatch.append((k,tuple(v.shape),tuple(tgt_sd[k].shape)))
        else:
            name_miss_in_target.append(k)

    if rank == 0:
        print(f"Target model (SwinUNETR) has {len(tgt_sd)} parameters in state_dict.")
        print(len(mapped),"mapped lenght")
        print(len(loadable),"loadable's key length")
    msg = swinunetr_model.load_state_dict(loadable, strict=False)

    if rank==0:
        print("shape_mismatch", shape_mismatch)
        print("name miss in target", name_miss_in_target)
        print("len of name miss in target", name_miss_in_target)
    return swinunetr_model


