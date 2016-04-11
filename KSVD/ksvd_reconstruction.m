clear
bb=8; % block size
RR=4; % redundancy factor
K=RR*bb^2; % number of atoms in the dictionary

sigma = 1;
c = load('signal_sparse.mat');
IMin0 = c.signal_sparse;

IMin = IMin0;
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)));

[IoutGlobal,output] = denoiseImageGlobal(IMin, sigma,K);

PSNROut = 20*log10(255/sqrt(mean((IoutGlobal(:)-IMin0(:)).^2)));
figure;
subplot(1,2,1); imshow(IMin0,[]); title('sparse image');
%subplot(1,3,2); imshow(IMin,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
subplot(1,2,2); imshow(IoutGlobal,[]); title(strcat(['reconstruction with global dictionary, ',num2str(PSNROut),'dB']));
figure;
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);
title('The dictionary trained on patches from natural images');