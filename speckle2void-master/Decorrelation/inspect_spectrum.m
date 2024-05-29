function inspect_spectrum(input_file)


complex_SAR = load(input_file);
fields=fieldnames(complex_SAR);
%inphase = getfield(complex_SAR,fields{2});
%inquad = getfield(complex_SAR,fields{1});
img_complex = getfield(complex_SAR,fields{1});

[r,c]=size(img_complex);
 
fC = fft2(img_complex);
S = real(fC.*conj(fC));
% azimuth
temp1=sqrt(mean(fftshift(S), 1));
temp1 = temp1/max(temp1);
x1 = -1:2/(r-1):1;
figure;
plot(x1, temp1, 'o');
title('Spectrum Mean in Azimuth');
xlabel('Normalized Frequency');
ylabel('Amplitude');
%range
temp1=sqrt(mean(fftshift(S), 2));
temp1 = temp1/max(temp1);
x1 = -1:2/(c-1):1;
figure;
plot(x1, temp1, 'o');
title('Spectrum Mean in Range');
xlabel('Normalized Frequency');
ylabel('Amplitude');
    
end