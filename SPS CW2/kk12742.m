function [] = kk12742()
Ss = zeros(10,2);
Ts = zeros(10,2);
Vs = zeros(10,2);

for i = 1 : 10
    %read a test image of each class
    S = imread(strcat('S',int2str(i),'.GIF')); 
    T = imread(strcat('T',int2str(i),'.GIF'));
    V = imread(strcat('V',int2str(i),'.GIF')); 
    
    %apply 2d fft to each, shifting u=0 v=0, take log of the magnitude
    %spectrum
    ftS = log(abs(fftshift(fft2(S))) + 1);
    ftT = log(abs(fftshift(fft2(T))) + 1);
    ftV = log(abs(fftshift(fft2(V))) + 1);
    
    
end

end