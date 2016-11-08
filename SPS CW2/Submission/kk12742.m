function [] = kk12742()
% For each class of the training data set create a 10 by 2 vector to store 
% information about the two features chosen using Fourier Space analysis
featS = zeros(10,2);
featV = zeros(10,2);
featT = zeros(10,2);

%% Populate the vectors with data obtained from the 2D FFT
for i = 1 : 10
    % Read a test image of each class
    S = imread(strcat('S',int2str(i),'.GIF'));
    V = imread(strcat('V',int2str(i),'.GIF'));
    T = imread(strcat('T',int2str(i),'.GIF'));
    
    % Apply the 2D FFT to it, center it, take log of (magnitude+1)
    % for perceptual scaling
    ft2S = log(abs(fftshift(fft2(S)))+1);
    ft2V = log(abs(fftshift(fft2(V)))+1);
    ft2T = log(abs(fftshift(fft2(T)))+1);
    
    % Extract information about both featuress for each test image
    featS(i,1) = extractFeat(ft2S,1);
    featS(i,2) = extractFeat(ft2S,2);
    featV(i,1) = extractFeat(ft2V,1);    
    featV(i,2) = extractFeat(ft2V,2);
    featT(i,1) = extractFeat(ft2T,1);
    featT(i,2) = extractFeat(ft2T,2);    
end    

%% Decrease data magnitude of all numeric feature data by a factor of 10^4 
% to for easier plot representation
featS = featS/10^4;
featT = featT/10^4;
featV = featV/10^4;

% Shove all the training data in one place
trainDat = [featS; featT; featV];

%% Read and transform testing samples
testA = imread('A1.GIF');
testA = log(abs(fftshift(fft2(testA)))+1);
testA_F(1,1) = extractFeat(testA,1);
testA_F(1,2) = extractFeat(testA,2);
testA_F = testA_F/10^4;

testB = imread('B1.GIF');
testB = log(abs(fftshift(fft2(testB)))+1);
testB_F(1,1) = extractFeat(testB,1);
testB_F(1,2) = extractFeat(testB,2);
testB_F = testB_F/10^4;

testS = zeros(13,2);
testT = zeros(13,2);
testV = zeros(13,2);
for i = 1 : 13
    S = imread(strcat('S',int2str(i+10),'.GIF'));
    ft2S = log(abs(fftshift(fft2(S)))+1);
    testS(i,1) = extractFeat(ft2S,1)/10^4;
    testS(i,2) = extractFeat(ft2S,2)/10^4;
    
    T = imread(strcat('T',int2str(i+10),'.GIF'));
    ft2T = log(abs(fftshift(fft2(T)))+1);
    testT(i,1) = extractFeat(ft2T,1)/10^4;
    testT(i,2) = extractFeat(ft2T,2)/10^4;
    
    V = imread(strcat('V',int2str(i+10),'.GIF'));
    ft2V = log(abs(fftshift(fft2(V)))+1);
    testV(i,1) = extractFeat(ft2V,1)/10^4;
    testV(i,2) = extractFeat(ft2V,2)/10^4;
end

%% Classify and display graphically
hold on;

% Create a grid to plot features and decision boundaries on
[x,y] = meshgrid(0.1:0.01:2.5, 0.1:0.01:2);

% Create a group variable for the knn (k nearest neighbours) classifier
group = [ones(10,1);repmat(2,10,1);repmat(3,10,1)];

% Classify each point on the grid using the k-nearest neighbours classifier
% with k=3 (using test data to plot features against each other) 
decision = knnclassify([x(:),y(:)], trainDat, group, 1); 

% Alternatively use nearest centroid to classify training data
C(1,:) = mean(featS);
C(2,:) = mean(featV);
C(3,:) = mean(featT);

% Scatter the grid according to nearest neighbour classifier decision
% coloring classes areas in Magenta Yellow abd Cyan
gscatter(x(:),y(:),decision, 'myc', '.',23);

% Display of the decision boundary of the nearrest-centroid classifier
voronoi(C(:,1),C(:,2), 'k');

% Align axis to plot space and label accordingly
axis([0.1 2.5 0.1 2]);
xlabel('Feature 1');
ylabel('Feature 2');

% Plot S training data points as circles
scatter(featS(:,1),featS(:,2),'go');
% Plot V training data points as squares
scatter(featV(:,1),featV(:,2),'s');
% Plot T training data points as triangles
scatter(featT(:,1),featT(:,2),'^');

%for k = 1 : 10
%    text(featT(k,1),featT(k,2),['(', num2str(featT(k,1)), ',', num2str(featT(k,2)), ')']);
%end

% Plot test samples on the already classified grid

plotSample(testS,1);
plotSample(testV,2);
plotSample(testT,3);
plotSample(testA_F, 4);
plotSample(testB_F, 5);

% Add a legend to the plot
% NOTE!
% Based on prior knowledge and plot observations:
%   * Cluster with high feat1 magnitude and low feat2 => T (vertical rectangle)
%   * Cluster with high feat2 magnitude and low feat1 => T (diagonal rhomb)
%   * Cluster with high feat1 and feat2 magnitude     => S
legend('S space','T space','V space','centroid', 'Nearest centroid decision','Nearest centroid decision','Nearest centroid decision','Train S', 'Train V', 'Train T','Test S', 'Test V', 'Test T');

hold off;

%% Extract information about features 1 and 2
function [mag] = extractFeat(fSpace, feat)
mag = 0;
% Feature 1 - rectangle on the central vertical line of the Fourier Space
    if(feat == 1)
        for y = 251:1:300
            for x = 310:1:330
                mag = mag + fSpace(y,x).^2;
            end
        end
    end
    
% Feature 2 - Rhomboid on the diagonal line of the Fourier Space
    if(feat == 2)
        off = 0;
        for y = 111:1:160
            for x = 400+off:1:430+off
                mag = mag + fSpace(y,x).^2;
            end
% offset loop over x-axis to extract data from a rhomboid-shaped section         
            off=off+1;
        end
    end
end

%% Plot sample testing data on the aleready classified grid namig it accordingly
function [] = plotSample(sample, symbolClass) 
    % Check the symbol class
    % test A
    if(symbolClass == 4) 
        scatter(sample(:,1),sample(:,2),'x','MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0 0 0]);
        text(sample(:,1),sample(:,2),'Test A','VerticalAlignment','top','HorizontalAlignment','left','Color',[0 0 0]);
    % test B  
    elseif (symbolClass == 5)
        scatter(sample(:,1),sample(:,2),'+','MarkerEdgeColor',[1 1 1],'MarkerFaceColor',[1 1 1]);
        text(sample(:,1),sample(:,2),'Test B','VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1]);  
    % test S
    elseif (symbolClass == 1) 
            scatter(sample(:,1),sample(:,2),'p','MarkerEdgeColor',[0 1 0],'MarkerFaceColor',[0 1 0]);
    % test V   
    elseif(symbolClass == 2)
            scatter(sample(:,1),sample(:,2),'p','MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0]);
    %test T
    elseif(symbolClass == 3)
            scatter(sample(:,1),sample(:,2),'p','MarkerEdgeColor',[0 0 1],'MarkerFaceColor',[0 0 1]);
    end
end
end