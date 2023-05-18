addpath([matlabroot '\toolbox\images\images']);
  % Film yolu
    filmPath = 'C:\Program Files\MATLAB\MatlabProje\aa';
    
    % Film listesi
    filmList = dir([filmPath '*.jpg']);
    
    % Görüntü işleme için kullanılan değişkenler
    blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
        'AreaOutputPort', false, 'CentroidOutputPort', false, ...
        'MinimumBlobArea', 50);
    
    % Yabancı nesneler için renk tanımlaması
    foreignObjectColor = [255, 0, 0];
      for i = 1:length(filmList)
        % Film yolu
        filmPath = fullfile(filmList(i).folder, filmList(i).name);
        
        % Film yükleniyor
        film = imread(filmPath);
        
        % Görüntü işleme
        grayFilm = rgb2gray(film);
        threshold = graythresh(grayFilm);
        bwFilm = im2bw(grayFilm, threshold);
        labeledFilms = bwlabel(bwFilm);
        blobMeasurements = regionprops(labeledFilms, 'all');
        
        % Etiketleme için yeni bir görüntü oluşturulur.
        labeledImage = label2rgb(labeledFilms, @jet, [.5 .5 .5]);
        
        % Yabancı nesneleri bulmak için bir döngü kullanarak etiketlerin üzerinde dolaşın.
        for j = 1:length(blobMeasurements)
            % Yabancı nesne için renkli kutu çizme
            if (blobMeasurements(j).Extent < 0.9)
                labeledImage = insertShape(labeledImage, 'Rectangle', ...
                    blobMeasurements(j).BoundingBox, 'Color', foreignObjectColor, 'LineWidth', 3);
            end
        end
        
        % Etiketli görüntüyü kaydet
        imwrite(labeledImage, ['labeled_', filmList(i).name]);
      end
% Veri seti yolu
dataSetPath = 'C:\Program Files\MATLAB\MatlabProje\aa';

% imageDatastore nesnesi oluştur
imds = imageDatastore(dataSetPath, 'IncludeSubfolders', true, ...
        'FileExtensions', '.jpg', 'LabelSource', 'foldernames');

% Eğitim seçeneklerini ayarla
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 1, ...
        'InitialLearnRate', 1e-3, ...
        'MaxEpochs', 10);

% Eğitim verilerini kullanarak nesne tespiti algoritmasını eğitin
detector = trainFasterRCNNObjectDetector(imds, layers, options, ...
        'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange', [0.6 1], ...
        'NumStrongestRegions', 200, 'BoxPyramidScale', 1.2);
 % Eğitim ve test verisi oluşturun
    imds = imageDatastore(dataSetPath, 'IncludeSubfolders', true, ...
        'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
    
   


     % Eğitim parametrelerini belirleyin
    options = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir);
    
    % Nesne tespiti için eğitim yapın
    detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange', [0.6 1], ...
        'NumStrongestRegions', 200, 'BoxPyramidScale', 1.2);

   % Eğitilmiş modeli kaydedin
    save('trainedDetector.mat', 'detector');

    testData = imageDatastore('testImagesFolder');
    detector = load('trainedDetector.mat');
    results = detect(detector, testData);
    evaluationMetrics = evaluateDetectionPrecision(results, testData);
averagePrecision = evaluationMetrics.AveragePrecision;
