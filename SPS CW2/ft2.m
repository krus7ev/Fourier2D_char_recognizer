function [] = ft2(im)
%clear all;  close all;

s1 = imread(im); 

f1 = fft2(double(s1));      %do fft in 2d
q1 = fftshift(f1);          %center u=0 v=0
Magq1 = abs(q1);            %magnitude spectrum
Phaseq1 = angle(q1);        %phase spectrum

figure(1);
F = mat2gray(log(Magq1+1));    %view FT
imshow(F,[]);
%Colorbar;                   %?

w1 = ifft2(ifftshift(q1));   %do inverse fft 2d
%figure(2);  
%imagesc(w1);
end