clear
bb=5; % block size
%RR=4; % redundancy factor
%K=RR*bb^2; % number of atoms in the dictionary
K = 2000;
sigma = 1;


b = loadMNISTImages('train-images-idx3-ubyte');
train_batch = 100; 
b = b(:,1:train_batch);
b = reshape(b, 28, 28, []);
IMin0 = b;

IMin = IMin0;
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)));

[IoutAdaptive,output] = denoiseImageKSVD(IMin,train_batch,sigma,K);

PSNROut = 20*log10(255/sqrt(mean((IoutAdaptive(:)-IMin0(:)).^2)));
figure;
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);
title('The dictionary trained on image set');