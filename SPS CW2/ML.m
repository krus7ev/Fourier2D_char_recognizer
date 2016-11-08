function [] = ML(a,b,k)
%Load training data file into a mtarix and store relevant feature columns 
%in a new matrix (chosen with chooseFeatures).
in = load('kk12742.train.txt');
data = [in(:,a),in(:,b)];

%Load test data and store relevant feature columns in a new matrix.
in1 = load('kk12742.test.txt');
test = [in1(:,a),in1(:,b)];

%Run kmeans to classify each data point into one of the three identified
%clusters, outputing labels at respective indices, the centroids of the
%clusters and the within-cluster sums of point-to-centroid distances.
%Then store the indices in separate vectors.
[IDX, C, Sum] = kmeans(data,k);
k1 = find(IDX==1);
k2 = find(IDX==2);
k3 = find(IDX==3);

%Allocate data points to separate matrices depending on the cluster they 
%belong to.
C1 = [data(k1,1),data(k1,2)];
C2 = [data(k2,1),data(k2,2)];
C3 = [data(k3,1),data(k3,2)];

%Calculate the mean and covariance matrix for each cluster
mu1 = mean(C1);
sigma1 = cov(C1);

mu2 = mean(C2);
sigma2 = cov(C2);

mu3 = mean(C3);
sigma3 = cov(C3);

%Create a grid of scale 0.1 
x = 1:0.1:9;
[X Y] = meshgrid(x,x);


%Compute the probability density of the distribution of each cluster
%taken as normally distributed and scale it to the grid.
mv1 = mvnpdf([X(:),Y(:)],mu1,sigma1);
mv1 = reshape(mv1,size(X));

mv2 = mvnpdf([X(:),Y(:)],mu2,sigma2);
mv2 = reshape(mv2,size(X));

mv3 = mvnpdf([X(:),Y(:)],mu3,sigma3);
mv3 = reshape(mv3,size(X));

%Calculate the likelyhood for a random point to fall into any of
%the clusters.
P1 = (1/(2*pi*sqrt(det(sigma1))))*exp(-3);
P2 = (1/(2*pi*sqrt(det(sigma2))))*exp(-3);
P3 = (1/(2*pi*sqrt(det(sigma3))))*exp(-3);

%Calculate the three pairwise likelyhood ratios.
lr12 = mv1./mv2;
lr13 = mv1./mv3;
lr23 = mv2./mv3;

%Recalculate the probability density using identity covariance matrices
%rather than the actual computed ones in order to obtain decision
%boundaries identical to those of the K-means method.
mv11= mvnpdf([X(:),Y(:)],mu1);
mv11 = reshape(mv11,size(X));

mv21= mvnpdf([X(:),Y(:)],mu2);
mv21 = reshape(mv21,size(X));

mv31= mvnpdf([X(:),Y(:)],mu3);
mv31 = reshape(mv31,size(X));

%Recalculate pairwise likelyhoods accordingly.
lr121 = mv11./mv21;
lr131 = mv11./mv31;
lr231 = mv21./mv31;

hold on
%Plot training data points clusters, each in a diferent colour.
scatter(data(k1,1),data(k1,2),'r');
scatter(data(k2,1),data(k2,2),'g');
scatter(data(k3,1),data(k3,2),'b');

%Plot test data points.
scatter(test(:,1),test(:,2), 'k','x');

%Plot the countours of the normal distribution for each class such that 95%
%of the probability mass of each falls within it.
contour(X,Y,mv1, [P1 P1],'r');
contour(X,Y,mv2, [P2 P2],'g');
contour(X,Y,mv3, [P3 P3],'b');

%Plot the decision boundaries of the ML classifier for each pair of
%classes.
contour(X,Y,lr12, [1 1],'m');
contour(X,Y,lr13, [1 1],'m');
contour(X,Y,lr23, [1 1],'m');

%Plot the decision boundaries of the skewed ML classifier, that uses 
%identity covariance matrices rather than the acttual ones to behave like
%K-means.
contour(X,Y,lr121, [1 1],'c');
contour(X,Y,lr131, [1 1],'c');
contour(X,Y,lr231, [1 1],'c');

%Plot decision bounderies of K-means optimal clustering.
voronoi(C(:,1),C(:,2),'k');

hold off
end