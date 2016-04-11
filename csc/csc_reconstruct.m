% Reconstruction from sparse data

clear;
close all;

%% Debug options
verbose = 'all';

%% Load image
%addpath('./image_helpers');
%CONTRAST_NORMALIZE = 'local_cn'; 
%ZERO_MEAN = 1;   
%COLOR_IMAGES = 'gray';   
%[b] = CreateImages('../datasets/test_images',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES); 
%signal = b(:,:,1);
%f = load('IMinfile.mat');
%b = f.IMin;
%signal = b;
batch = 100;
filters = 20;
b = loadMNISTImages('t10k-images-idx3-ubyte');
b = b(:,1:batch);
b = reshape(b, 28, 28, []);

l = loadMNISTLabels('t10k-labels-idx1-ubyte');
csvwrite('l.csv',l(1:batch))

signal = b;
%Sampling matrix
MtM = ones(size(signal(:,:,1)));
%MtM(1:2:end, 1:2:end) = 1;
%MtM(rand(size(MtM)) < 0.5 ) = 1;

%Subsample
%signal_sparse = signal + 0.05 * randn(size(signal));
signal_sparse = signal;
%signal_sparse( ~MtM ) = 0;

%% Load filters
%kernels = load('../learned_filters/filters_ours_obj1.26e04.mat');
kernels = load('../csc/filters_ours_obj202.mat');

d = kernels.d;
%Show kernels
if strcmp(verbose, 'brief ') || strcmp(verbose, 'all') 
    figure();
    sqr_k = ceil(sqrt(size(d,3))); pd = 1;
    psf_radius = floor(size(d,1)/2);
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_disp( floor(j/sqr_k) * (size(d,1) + pd) + pd + (1:size(d,1)) , mod(j,sqr_k) * (size(d,2) + pd) + pd + (1:size(d,2)) ) =  d(:,:,j + 1);
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title('Kernels used');
    
    figure();
    %subplot(1,2,1), imagesc( signal ), axis image, colormap gray, title('Original image');
    %subplot(1,2,2), imagesc( signal_sparse ), axis image, colormap gray, title('Subsampled image');
end

%% 1) Sparse coding reconstruction     
fprintf('Doing sparse coding reconstruction.\n\n')

lambda_residual = 5.0;
lambda = 2.0; %

verbose_admm = 'all';
max_it = [100];
tic();
%Z = zeros(32,32,20,100);\
Z = zeros(batch,20,32,32);
for i = 1:batch 
    [z, sig_rec] = admm_solve_conv2D_weighted_sparse_reconstruction(signal_sparse(:,:,i), d, MtM, lambda_residual, lambda, max_it, 1e-3, signal(:,:,i), verbose_admm); 
    for layer = 1:filters
        for height = 1:32
            for width = 1:32
                Z(i,layer,height,width) = z(height,width,layer); 
            end
        end
    end
end
tt = toc;

% maybe = zeros(28,28)
% for i = 1:20
%     maybe = maybe + conv2(reshape(Z(1,i,:,:),32,32), reshape(d(:,:,i),5,5),'valid')
% end
% print(maybe)
save('Z.mat','Z');

%Z = reshape(Z,1,batch*20*32*32);
%csvwrite('Z.csv',Z)
% %Show result
% if strcmp(verbose, 'brief ') || strcmp(verbose, 'all') 
%     figure();
% %     subplot(1,2,1), imagesc(signal), axis image, colormap gray; title('Orig');
% %     subplot(1,2,2), imagesc(sig_rec), axis image, colormap gray; title('Reconstruction');
%     subplot(1,2,1), imshow(signal,[]), axis image, colormap gray; title('Orig');
%     subplot(1,2,2), imshow(sig_rec,[]), axis image, colormap gray; title('Reconstruction');
% end
% 
% %Debug
% fprintf('Done sparse coding! --> Time %2.2f sec.\n\n', tt)
% 
% %Write stuff
% max_sig = max(signal(:));
% min_sig = min(signal(:));
% 
% %Transform and save
% signal_disp = (signal - min_sig)/(max_sig - min_sig);
% signal_sparse_disp = (signal_sparse - min_sig)/(max_sig - min_sig);
% signal_sparse_disp( ~MtM ) = 0;
% sig_rec_disp = (sig_rec - min_sig)/(max_sig - min_sig);
% 
% max_d = max(d_disp(:));
% min_d = min(d_disp(:));
% d_sc = (d - min_d)/(max_d - min_d);
% 
% sqr_k = ceil(sqrt(size(d,3))); pd = 1;
% psf_radius = floor(size(d,1)/2);
% d_disp = ones( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
% for j = 0:size(d,3) - 1
%     d_disp( floor(j/sqr_k) * (size(d,1) + pd) + pd + (1:size(d,1)) , mod(j,sqr_k) * (size(d,2) + pd) + pd + (1:size(d,2)) ) =  d_sc(:,:,j + 1);
% end
% 
% %Save stuff
% imwrite(signal_disp , 'signal.png','bitdepth', 16);
% imwrite(signal_sparse_disp ,'signal_sparse.png','bitdepth', 16);
% imwrite(sig_rec_disp ,'signal_reconstruction.png','bitdepth', 16);
% imwrite(d_disp ,'kernel.png','bitdepth', 16);
