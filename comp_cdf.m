clear all;
close all;

I = imread('./TestData/jpg/elephant_baby_4.jpg');
[height,width,~] = size(I);
% I1 = rgb2gray(I);
I1 = I(:,:,1);
I2 = I(:,:,2);
I3 = I(:,:,3);

r=4;

I1_resized=blkM2vc(I1,[r r]);
ncdf_r = imgnormcdf(I1);
I1_resized = vc2cdf(I1_resized*255,ncdf_r);

I2_resized=blkM2vc(I2,[r r]); 
ncdf_g = imgnormcdf(I2);
I2_resized = vc2cdf(I2_resized*255,ncdf_g);

I3_resized=blkM2vc(I3,[r r]); 
ncdf_b = imgnormcdf(I3);
I3_resized = vc2cdf(I3_resized*255,ncdf_b);

net_c = feedforwardnet(1);
net_c.layers{1}.size = 4;
net_c.trainparam.epochs=30; 
net_c.trainparam.goal=1e-5; 
[net_s,tr]=train(net_c,I1_resized,I1_resized); 
 
a=sim(net_s,I1_resized); 

a = a';
output = zeros(size(a));
for i=1:size(a,1)
    output(i,:) = quantiz(a(i,:),ncdf_r);
end
I1_compressed=vc2blkM(a',r,height,width);

a=sim(net_s,I2_resized); 
a = a';
output = zeros(size(a));
for i=1:size(a,1)
    output(i,:) = quantiz(a(i,:),ncdf_g);
end
I2_compressed=vc2blkM(a',r,height,width);

a=sim(net_s,I3_resized); 
a = a';
output = zeros(size(a));
for i=1:size(a,1)
    output(i,:) = quantiz(a(i,:),ncdf_b);
end
I3_compressed=vc2blkM(a',r,height,width);
% imshow(I1_compressed)
RGBImage = cat(3,I1_compressed,I2_compressed,I3_compressed);
%imshow(I)
%hold;
imshow(RGBImage)
imwrite(RGBImage,'./Results/FFNN/jpg/elephant_baby_cdf.jpg');
