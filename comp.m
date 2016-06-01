clear all;
close all;

I = imread('./TestData/jpg/PIA20676_modest.jpg');
[height,width,~] = size(I);
I1 = I(:,:,1);
I2 = I(:,:,2);
I3 = I(:,:,3);
% I1 = rgb2gray(I);
r=4;
h=4;
I1=blkM2vc(I1,[r r]); 
I2=blkM2vc(I2,[r r]); 
I3=blkM2vc(I3,[r r]); 
net_c = feedforwardnet(1);
net_c.layers{1}.size = h;
net_c.trainparam.epochs=30; 
net_c.trainparam.goal=1e-5; 
[net_s,tr]=train(net_c,I1,I1); 

a=sim(net_s,I1); 
I1_compressed=vc2blkM(a,r,height,width);
a=sim(net_s,I2); 
I2_compressed=vc2blkM(a,r,height,width);
a=sim(net_s,I3); 
I3_compressed=vc2blkM(a,r,height,width);

RGBImage = cat(3,I1_compressed,I2_compressed,I3_compressed);
% imshow(I1)
%hold;
imshow(RGBImage)
% path = strcat('./Results/FFNN/jpg/Test1/h=',int2str(h),'/r=',int2str(r),'/elephant_baby.jpg');

% imwrite(RGBImage,path);