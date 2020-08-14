function [net, info] = trainconv3()
global datadir;
run matlab/vl_setupnn ;

datadir='/home/longpeng/project/LongPeng_ML_Course/projects/classification/matconvnet/conv3/mouth';
opts.expDir = fullfile('/home/longpeng/project/LongPeng_ML_Course/projects/classification/matconvnet/conv3/','imdb') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

if exist(opts.imdbPath,'file')
    imdb=load(opts.imdbPath);
else
    imdb=mydataset(datadir);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

net=simpleconv3();
net.meta.normalization.averageImage =imdb.images.data_mean ;
opts.train.gpus=1;

[net, info] = cnn_train(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

function fn = getBatch(opts)
% --------------------------------------------------------------------
    fn = @(x,y) getSimpleNNBatch(x,y) ;
end

function [images, labels]  = getSimpleNNBatch(imdb, batch)
    images = imdb.images.data(:,:,:,batch) ;
    labels = imdb.images.labels(1,batch) ;
    if opts.train.gpus > 0
        images = gpuArray(images) ;
    end
end
end