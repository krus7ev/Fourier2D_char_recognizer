function [ ] = importImg()
featS = zeros(10, 2);
featV = zeros(10, 2);
featT = zeros(10, 2);

for i = 1 : 10
    %Read a test image of each class
    S = imread(strcat('S',int2str(i),'.GIF'));
    V = imread(strcat('V',int2str(i),'.GIF'));
    T = imread(strcat('T',int2str(i),'.GIF'));
   
    %Fourier transform them
    ft2S = log(abs(fftshift(fft2(S))) + 1);
    ft2V = log(abs(fftshift(fft2(V))) + 1);
    ft2T = log(abs(fftshift(fft2(T))) + 1);
    
    %Store infromation about both features on all three samples S V and T
    featS(i,1) = Feature (ft2S,1);
    featS(i,2) = Feature (ft2S,2);
    featV(i,1) = Feature (ft2V,1);    
    featV(i,2) = Feature (ft2V,2);
    featT(i,1) = Feature (ft2T,1);
    featT(i,2) = Feature (ft2T,2);   
end

%% Sample A
A = imread('A1.GIF');
sampleA_Imgfft = log(abs(fftshift(fft2(A))) + 1);
sampleA_ImgF(1,1) = Feature (sampleA_Imgfft,1);
sampleA_ImgF(1,2) = Feature (sampleA_Imgfft,2);
    
%% Sample B
B = imread('B1.GIF');
sampleB_Imgfft = log(abs(fftshift(fft2(B))) + 1);
sampleB_ImgF(1,1) = Feature (sampleB_Imgfft,1);
sampleB_ImgF(1,2) = Feature (sampleB_Imgfft,2);

%% Sample S
sampleS_Img = imread('S22.GIF');
sampleS_Imgfft = log(abs(fftshift(fft2(sampleS_Img))) + 1);
sampleS_ImgF(1,1) = Feature (sampleS_Imgfft,1);
sampleS_ImgF(1,2) = Feature (sampleS_Imgfft,2);

%% Sample V
sampleV_Img = imread('V22.GIF');
sampleV_Imgfft = log(abs(fftshift(fft2(sampleV_Img))) + 1);
sampleV_ImgF(1,1) = Feature (sampleV_Imgfft,1);
sampleV_ImgF(1,2) = Feature (sampleV_Imgfft,2);

%% Sample T
sampleT_Img = imread('T23.GIF');
sampleT_Imgfft = log(abs(fftshift(fft2(sampleT_Img))) + 1);
sampleT_ImgF(1,1) = Feature (sampleT_Imgfft,1);
sampleT_ImgF(1,2) = Feature (sampleT_Imgfft,2);


% Decrease magnitude to better represent on the plot
smallerSampleA_ImgF = sampleA_ImgF/1e4;
smallerSampleB_ImgF = sampleB_ImgF/1e4;
smallerSampleS_ImgF = sampleS_ImgF/1e4;
smallerSampleT_ImgF = sampleT_ImgF/1e4;
smallerSampleV_ImgF = sampleV_ImgF/1e4;

smallerSp = featS/1e4;
smallerTp = featT/1e4;
smallerVp = featV/1e4;

smallerData= [smallerSp;smallerTp;smallerVp];

hold on;
x = 0.1:0.01:2.5;
y = 0.1:0.01:2;
[X Y] = meshgrid(x, y);
X = X(:);
Y = Y(:);

group = [repmat(1,10,1);repmat(2,10,1);repmat(3,10,1)];

% Nearest neighbour classifier, using closest 3 neighbours to check
class = knnclassify([X Y], smallerData, group,3); 

% paint the plot according to nearest neighbour classifier
gscatter(X,Y,class, 'rbg', '.',10); 
legend('S','T','V');

% restrict axis to not show unused space on the plot
axis([0.1 2.5 0.1 2]); 
         
% Classify sample using nearest neighbour classifier in function
% findCluster
findCluster(smallerData, smallerSampleS_ImgF, 1);
findCluster(smallerData, smallerSampleT_ImgF, 1);
findCluster(smallerData, smallerSampleV_ImgF, 1);

findCluster(smallerData, smallerSampleA_ImgF, 2);
findCluster(smallerData, smallerSampleB_ImgF, 3);

%% Function to pick out the feature
function [ power ] = Feature( Array,whichFeature )
[a,b] = size(Array);
power = 0;
    % Box on the vertical line in the Fourier Space
    if(whichFeature == 1)
        for a = 101:1:150
            for b = 310:1:330
                power = power + Array(a,b).^2;
            end
        end
    end
    
    % Box on the diagonal line in the Fourier Space
    if(whichFeature == 2)
        w1=0;
        for a = 101:1:150
            for b = 390+w1:1:420+w1
                power = power + Array(a,b).^2;
            end
            % variable which makes a rhomb shape for the box
            w1=w1+1;
        end
    end

end

%% Function to find to which cluster a point belongs to using the training
%% set
function [ class ] = findCluster( training , sample, letter )
    group = [repmat(1,10,1);repmat(2,10,1);repmat(3,10,1)];
    class = knnclassify(sample, training, group,3);
    
    S_TF(:,1) = training(1:10,1);
    S_TF(:,2) = training(1:10,2);
    
    T_TF(:,1) = training(11:20,1);
    T_TF(:,2) = training(11:20,2);
    
    V_TF(:,1) = training(21:30,1);
    V_TF(:,2) = training(21:30,2);
    
    scatter(S_TF(:,1),S_TF(:,2),'kv');%S
    scatter(T_TF(:,1),T_TF(:,2),'go');%T
    scatter(V_TF(:,1),V_TF(:,2),'rd');%V
    if(letter == 2) %A
        scatter(sample(:,1),sample(:,2),'m^','LineWidth',7,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0 0 0]);
        
    elseif (letter == 3) %B
        scatter(sample(:,1),sample(:,2),'m+','LineWidth',7,'MarkerEdgeColor',[1 1 1],'MarkerFaceColor',[1 1 1]);
    
    else
        if(class == 1)%S
            scatter(sample(:,1),sample(:,2),'ms','LineWidth',2.75,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[1 1 1]);
        end
        if(class == 2)%T
            scatter(sample(:,1),sample(:,2),'cs','LineWidth',2.75,'MarkerEdgeColor',[0 1 1],'MarkerFaceColor',[1 1 1]);
        end
        if(class == 3)%V
            scatter(sample(:,1),sample(:,2),'ws','LineWidth',2.75,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 1 1]);
        end
    end
end

end