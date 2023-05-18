
% Eğitim veri setini oluştur
trainingData = struct('image', {}, 'label', {});

% Yabancı cisim içeren görüntülerin yollarını belirt
foreignObjectImages = {'1-1a.jpeg', '1-2a.jpeg','1-3a.jpeg'};
% Okunan resimleri saklamak için bir hücre dizisi oluşturduk
imagesforeign = cell(1, numel(foreignObjectImages));
% Yabancı cisim içeren resimleri okuma ve hücre dizisine kaydetme
for i = 1:numel(foreignObjectImages)
    % Resmi okuma
    imagesforeign = imread(foreignObjectImages{i});
    % Gri tonlamalı görüntüye dönüştürme
    grayImageforeign = im2gray(imagesforeign);
    % Kenar tespiti yapma
    edge_img = edge(grayImageforeign, 'Canny');
    %imshow(edge_img), title('Foreign Fotoğraf Kenar Tespiti');
    % Görüntüyü ikleştirme işlemi yaparak kontrastı artırrma
    enhanced_imgforeign = imadjust(grayImageforeign);
    % Binarizasyon işlemi yaparak yabancı cisim bölgelerini belirleme
    bw_imgforeign = imbinarize(enhanced_imgforeign, 'adaptive');
    % Büyük nesneleri baskıla
    bw_imgforeign = bwareaopen(bw_imgforeign, 5000);
    % Nesneleri etiketle ve özelliklerini çıkar
    ccforeign = bwconncomp(bw_imgforeign);
    statsforeign = regionprops(ccforeign, grayImageforeign, 'BoundingBox', 'Area', 'Perimeter', 'Circularity', 'MeanIntensity');

  % Yabancı cisimlerin özelliklerini analiz etmek için döngü
   for j = 1:ccforeign.NumObjects
    % Yabancı cismin boyutunu hesapla
    foreignwidth = statsforeign(j).BoundingBox(3);
    foreignheight = statsforeign(j).BoundingBox(4);

    % Yabancı cismin şeklini hesapla
    foreignshape = foreignheight / foreignwidth; % Oran olarak hesaplama, kareye yakınlık kontrolü yapıldı

    % Yabancı cismin renk dağılımını hesapla
    foreignintensity = statsforeign(j).MeanIntensity;

    %Sonuçları göster
   disp(['Foreign Yabancı Cisim ', num2str(j)]);
    disp(['Foreign Boyut: ', num2str(foreignwidth), ' x ', num2str(foreignheight)]);
    disp(['Foreign Şekil Oranı: ', num2str(foreignshape)]);
    disp(['Foreign Ortalama Yoğunluk: ', num2str(foreignintensity)]);
    disp('---------------------------');
   end
end

normalImages = {'0-1n.jpeg', '0-2n.jpeg'};
imagesnormal = cell(1, length(normalImages)); % hücre dizisi boyutunu değiştirme
for i = 1:length(normalImages)
    imagesnormal{i} = imread(normalImages{i}); % resmi hücre dizisine ekle
    grayImagennormal = im2gray(imagesnormal{i});    
    edge_img = edge(grayImagennormal, 'Canny');
    enhanced_imgnormal = imadjust(grayImagennormal);
    bw_imgnormal = imbinarize(enhanced_imgnormal, 'adaptive');
    bw_imgnormal = bwareaopen(bw_imgnormal, 5000);
    ccnormal = bwconncomp(bw_imgnormal);
    statsnormal = regionprops(ccnormal, grayImagennormal, 'BoundingBox', 'Area', 'Perimeter', 'Circularity', 'MeanIntensity');
    for j = 1:ccnormal.NumObjects
        normalwidth = statsnormal(j).BoundingBox(3);
        normalheight = statsnormal(j).BoundingBox(4);
        normalshape = normalheight / normalwidth;
        normalintensity = statsnormal(j).MeanIntensity;
    end
end

im_gray = imread('C:\Program Files\MATLAB\MatlabProje\anormal6.jpeg');
im_bw = imbinarize(im_gray);


% Yabancı cisim içeren görüntülere bir sınıf etiketi (1) ve yabancı cisim içermeyen 
% normal göğüs filmi görüntülerine başka bir sınıf etiketi (0) atadık.
% Bölge Özelliklerini Ölçme
stats = regionprops('table',im_bw,'Centroid','BoundingBox','Area');

% Sınırlayıcı Kutu Çizme
figure;imshow(im_bw);
hold on;
for i = 1:size(stats,1)
    if stats.Area(i) > 1000 % Bölge alanı kontrolü
        rectangle('Position',stats.BoundingBox(i,:),'EdgeColor','r','LineWidth',2);
    end
end

% Yabancı cisim içeren görüntülere sınıf etiketi atayarak veri setine ekle
for i = 1:numel(foreignObjectImages)
    img = imread(foreignObjectImages{i});
    data.image = img;
    data.label = 1; % Yabancı cisim sınıf etiketi
    trainingData = [trainingData, data];
end

% Normal göğüs filmi görüntülerine sınıf etiketi atayarak veri setine ekle
for i = 1:numel(normalImages)
    img = imread(normalImages{i});
    data.image = img;
    data.label = 0; % Normal sınıf etiketi
    trainingData = [trainingData, data];
end


% CNN modeli oluşturma
model = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(11, 96, "Stride", 4, "Padding", 0)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3, "Stride", 2, "Padding", 0)
    
    convolution2dLayer(5, 256, "Stride", 1, "Padding", 2)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3, "Stride", 2, "Padding", 0)
    
    convolution2dLayer(3, 384, "Stride", 1, "Padding", 1)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 384, "Stride", 1, "Padding", 1)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 256, "Stride", 1, "Padding", 1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3, "Stride", 2, "Padding", 0)
    
    fullyConnectedLayer(4096)
    reluLayer
    
    fullyConnectedLayer(4096)
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];
% Eğitim seçeneklerini belirleme
options = trainingOptions("adam", ...
    "MaxEpochs", 10, ...
    "MiniBatchSize", 32, ...
    "InitialLearnRate", 0.0001, ...
    "LearnRateSchedule", "piecewise", ...
    "LearnRateDropFactor", 0.1, ...
    "LearnRateDropPeriod", 5);

options = trainingOptions('sgdm', 'MaxEpochs', 10, 'MiniBatchSize', 16, 'InitialLearnRate', 0.001);

save('trainedNet.mat', 'model');

% Test veri kümesi yolu
testSetDir = 'C:\Program Files\MATLAB\matlabproje\test';

% Test veri kümesi nesnesini oluşturma
testSet = imageDatastore(testSetDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Eğitilmiş modeli yükle
loadedModel = load('trainedNet.mat');
net = loadedModel.model;
