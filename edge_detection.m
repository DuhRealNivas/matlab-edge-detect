% MATLAB Implementation for Face Edge Detection & Segmentation

% Step 1: Upload & Load Image
imageFile = 'image_nivas.jpg';
img = imread(imageFile);

% Convert image to grayscale
if size(img, 3) == 3
    imgGray = rgb2gray(img);
else
    imgGray = img;
end

% Apply Gaussian Blur to reduce noise while preserving edges
sigma = 1; % Standard deviation for Gaussian blur
blurredImage = imgaussfilt(imgGray, sigma);

% Display the original image and grayscale image
figure;
subplot(1,3,1); imshow(img); title('Original Image');
subplot(1,3,2); imshow(imgGray); title('Grayscale Image');
subplot(1,3,3); imshow(blurredImage); title('Blurred Image (Guassian) Sigma = 1');



% Step 2: Apply Edge Detection Techniques

% Apply Edge Detection on Gaussian Blurred Image of Sigma = 1
sobelEdges = edge(blurredImage, 'sobel');
cannyEdges = edge(blurredImage, 'canny');
prewittEdges = edge(blurredImage, 'prewitt');
logEdges = edge(blurredImage, 'log');
robertsEdges = edge(blurredImage, 'roberts');

% Display all edge detection results
figure;
subplot(2,3,1); imshow(blurredImage); title('Blurred Image (Gaussian)');
subplot(2,3,2); imshow(sobelEdges); title('Sobel Edge Detection');
subplot(2,3,3); imshow(cannyEdges); title('Canny Edge Detection');
subplot(2,3,4); imshow(prewittEdges); title('Prewitt Edge Detection');
subplot(2,3,5); imshow(logEdges); title('LoG (Laplacian of Gaussian) Edge Detection');
subplot(2,3,6); imshow(robertsEdges); title('Roberts Edge Detection');

% Apply Edge Detection Techniques with Varying Parameters

% Define different sigma values for Gaussian blur (Canny)
sigmaValues = [0.5, 1, 1.4, 2, 3, 4];
nSigma = length(sigmaValues);

% Define different kernel sizes for Sobel, Prewitt, and Roberts
kernelSizes = [3, 5, 7, 9, 11, 13];
nKernel = length(kernelSizes);

% Create a figure window to display results for Canny
figure;
for i = 1:nSigma
    % Apply Gaussian blur with varying sigma
    blurredImage = imgaussfilt(imgGray, sigmaValues(i));
    
    % Apply Canny edge detection
    cannyEdges = edge(blurredImage, 'canny');
    
    % Display results
    subplot(2,3,i);
    imshow(cannyEdges);
    title(['Canny (Sigma = ', num2str(sigmaValues(i)), ')']);
end

% Create a figure window to display results for Sobel, Prewitt, and Roberts
figure;
for i = 1:nKernel
    % Apply edge detection with different kernel sizes
    sobelEdges = edge(imgGray, 'sobel');
    prewittEdges = edge(imgGray, 'prewitt');
    
    % Display results in a properly structured layout
    subplot(2,6,i); % Adjusted layout
    imshow(sobelEdges);
    title(['Sobel (Kernel = ', num2str(kernelSizes(i)), ')']);

    subplot(2,6,nKernel + i);
    imshow(prewittEdges);
    title(['Prewitt (Kernel = ', num2str(kernelSizes(i)), ')']);

end

% Step 3: Face Segmentation using K-means, SVM, and DeepLabV3+

% Define different K values for K-means clustering
kValues = [2, 3, 4, 5]; % Experiment with different cluster numbers
nK = length(kValues);

% Convert image to double precision for clustering
imgDouble = im2double(imgGray);
pixels = imgDouble(:);

% Create a figure to display K-means segmentation results
figure;
for i = 1:nK
    % Apply K-means clustering
    [idx, ~] = kmeans(pixels, kValues(i));
    segmentedImg = reshape(idx, size(imgGray));
    
    % Display segmented results
    subplot(3, nK, i);
    imshow(segmentedImg, []);
    title(['K-means (K = ', num2str(kValues(i)), ')']);
end

% SVM Segmentation
% Prepare data for SVM training (Use K-means with K=2 as labels)
kmeansIdx = kmeans(pixels, 2);
labels = categorical(kmeansIdx(:)); % Convert labels to categorical format
features = double(imgGray(:));
svmModel = fitcsvm(features, double(labels), 'KernelFunction', 'linear');
predictedLabels = predict(svmModel, features);
svmSegmentedImg = reshape(predictedLabels, size(imgGray));

% Display SVM segmentation results
subplot(3, nK, nK + 1);
imshow(svmSegmentedImg, []);
title('SVM Segmentation');



% MATLAB Implementation for Additional Features - Fix for Skin Color-Based Masking

% Read original RGB image 
imgRGB = imread('image_nivas.jpg');
imgYCbCr = rgb2ycbcr(imgRGB); % Convert to YCbCr color space
Y = imgYCbCr(:,:,1); % Luminance
Cb = imgYCbCr(:,:,2); % Chrominance Blue
Cr = imgYCbCr(:,:,3); % Chrominance Red

% Adjusted skin color range to improve face detection
skinMask = (Cb >= 80 & Cb <= 135) & (Cr >= 135 & Cr <= 180) & (Y > 50); 

% Apply mask to keep only skin regions (remove hair and shoulders)
faceOnly = imgGray;
faceOnly(~skinMask) = 0; % Keep only skin pixels

% Define different sigma values for Gaussian blur
sigmaValues = [0.5, 1, 2, 4, 6, 8, 10, 13];
nSigma = length(sigmaValues);

% Create a figure window to display improved colored edges
figure;
for i = 1:nSigma
    % Apply Gaussian blur with varying sigma
    blurredImage = imgaussfilt(faceOnly, sigmaValues(i));
    
    % Apply Canny edge detection to extract outer face boundary
    faceEdges = edge(blurredImage, 'canny');
    
    % Convert edges to 3-channel format for better visualization
    coloredEdges = cat(3, faceEdges, zeros(size(faceEdges)), zeros(size(faceEdges))); % Red outline
    
    % Display results
    subplot(3,3,i);
    imshow(coloredEdges);
    title(['Face Outline (Sigma = ', num2str(sigmaValues(i)), ')']);
end


% Super impose signature onto my original image
original_image = imread('image_nivas.jpg'); % Directly load image from file

signature_image = imread('signature_inv.png'); % Ensure the generated file exists
signature_image = imresize(signature_image, [50, 200]); % Resize if needed

% Convert the signature to binary (black/white)
signature_binary = imbinarize(rgb2gray(signature_image));

[orig_h, orig_w, orig_d] = size(original_image);
[sig_h, sig_w] = size(signature_binary);

% Define Signature Position (Top-Center)
[orig_h, orig_w, orig_d] = size(original_image);
[sig_h, sig_w] = size(signature_binary);

pos_x = round((orig_w - sig_w) / 2); % Centered horizontally
pos_y = 10; % 10 pixels from the top

% Convert original image to RGB if it's grayscale
if size(original_image, 3) == 1
    original_image = repmat(original_image, [1, 1, 3]);
end

% Overlay the Signature on the Original Image
overlayed_image = original_image;

% Superimpose white signature pixels onto the image
for i = 1:sig_h
    for j = 1:sig_w
        if signature_binary(i, j) == 1 % White signature pixels
            overlayed_image(pos_y + i, pos_x + j, :) = 255; % Set pixel to white
        end
    end
end

% Display and Save the Final Image
figure;
imshow(overlayed_image);
title('Original Image with Superimpose Signature');
