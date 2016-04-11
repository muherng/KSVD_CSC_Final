% Learning from dense data for sparse convolutional coding

clear;
close all;

%% Debug options
verbose = 'all';

% %% Load the data
% addpath('./image_helpers');
% CONTRAST_NORMALIZE = 'local_cn'; 
% ZERO_MEAN = 1;   
% COLOR_IMAGES = 'gray';                         
% [b] = CreateImages('../datasets/Images/city_100_100',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
% 
% %One long dataset iterating over color if defined
% b = reshape(b, size(b,1), size(b,2), [] ); 

b = loadMNISTImages('train-images-idx3-ubyte');
normal_images = 480;
b = b(:,1:normal_images);
b = reshape(b, 28, 28, []);

large_images = normal_images/16;
c = zeros(28*4,28*4,large_images);
for i = 1:large_images
    for j = 1:4
        for k = 1:4
            c(((j-1)*28 + 1):j*28,((k-1)*28 + 1):k*28,i) = b(:,:,(i-1)*16 + 1 + (j-1)*4 + (k-1));  
        end
    end
end

%% Define the parameters
filter_number = 20;
kernel_size = [7, 7, filter_number];
lambda_residual = 1.0;
lambda = 1.0; %2.8

%% Do the reconstruction  
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d] kernels.\n\n', kernel_size(3), kernel_size(1), kernel_size(2) )

%Optim options
verbose_admm = 'all';
max_it = [100];
%tol = 1e-3;
tol = .001;

tic();

prefix = 'ours';
kernel_size = [7, 7, filter_number];
[ d, z, Dz, obj, iterations]  = admm_learn_conv2D_weighted(c, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, []);

tt = toc;

%Show result
if strcmp(verbose, 'brief ') || strcmp(verbose, 'all') 
    figure();    
    pd = 1;
    sqr_k = ceil(sqrt(size(d,3)));
    d_disp = zeros( sqr_k * [kernel_size(1) + pd, kernel_size(2) + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_disp( floor(j/sqr_k) * (kernel_size(1) + pd) + pd + (1:kernel_size(1)) , mod(j,sqr_k) * (kernel_size(2) + pd) + pd + (1:kernel_size(2)) ) = d(:,:,j + 1); 
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title('Final filter estimate');
end

%Save
save(sprintf('filters_%s_obj%3.3g.mat', prefix, obj), 'd', 'z', 'Dz', 'obj', 'iterations');

%Debug
fprintf('Done sparse coding learning! --> Time %2.2f sec.\n\n', tt)
