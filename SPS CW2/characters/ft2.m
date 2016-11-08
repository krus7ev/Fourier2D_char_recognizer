function [] = ft2(im)

s1 = imread(im); 

f1 = fft2(double(s1));      %do fft in 2d
q1 = fftshift(f1);          %center u=0 v=0
Magq1 = abs(q1);            %magnitude spectrum

figure(1);
F = mat2gray(log(Magq1+1));  %view FT
hold on

imshow(F,[]);

rectangle('Position',[310,250,20,50], 'EdgeColor', 'w');

line([400 450], [110 160]);
line([430 480], [110 160]);
line([400 430], [110 110]);
line([450 480], [160 160]);

hold off

end