I as understand:
the basic of dreambooth is:
x_i ~ instance_images
x_c ~ class_images

x = concat(x_i, x_c)   
p = concat(p_i, p_c) 

c  = TextEncoder(p)  
 pred = UNet(z_t, t, c) 

pred_i, pred_c = split(pred) 
tgt_i,  tgt_c  = split(target)

so if we want to use DreamBooth + ControlNet Fine-tuning
we should:
x_i, pose_i ~ instance_images
x_c, pose_i ~ class_images (the same pose)

pred = UNet(z_t, t, c, pose_i) (so basically we will teach the controlnet to output the instance instead of a general class, AND ON THE SAME POSE as input -> the model will understand that with this pose, a specific prompt, I should output that instance)
