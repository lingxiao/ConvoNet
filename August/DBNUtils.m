
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Module: Convolutional Deep Belief Networks Utils
%       
%        The detail of the algorithm is described in the following paper:
%        'Convolutional Deep Belief Networks for Scalable Unsupervised Learning of 
%         Hiearchical Representations'
% 
% Author: Xiao Ling
% Date  : June 4th, 2014
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

% Export Module Functions
function T = ConvDeepBeliefUtils()

    Pr         = Prelude();
	T.show     = @(a,b) displayNetwork(a,b);
    T.whiten   = @(a)   crbm_whiten_olshausen(a);

end

% -----------------------------------------------------------------------------------
% pre-process raw image
% -----------------------------------------------------------------------------------

function im_out = crbm_whiten_olshausen(im)

    if ~exist('D', 'var'), D = 16; end
    if size(im,3)>1, im = rgb2gray(im); end
    im = double(im);

    im = im - mean(im(:));
    im = im./std(im(:));

    N1 = size(im, 1);
    N2 = size(im, 2);

    [fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
    rho=sqrt(fx.*fx+fy.*fy)';

    f_0=0.4*mean([N1,N2]);
    filt=rho.*exp(-(rho/f_0).^4);

    If=fft2(im);
    imw=real(ifft2(If.*fftshift(filt)));

    im_out = imw/std(imw(:)); % 0.1 is the same factor as in make-your-own-images


    im_out = im_out-mean(mean(im_out));
    im_out = im_out/sqrt(mean(mean(im_out.^2)));
    im_out = sqrt(0.1)*im_out; % just for some trick??

end

% -----------------------------------------------------------------------------------
% Show
% -----------------------------------------------------------------------------------

function [h, array] = displayNetwork(A,str,opt_normalize, cols)

	figure;

    opt_normalize= true;
    opt_graycolor= true;
    opt_colmajor = false;

    if opt_graycolor, colormap(gray); end

    % compute rows, cols
    [L M]=size(A);
    sz=sqrt(L);
    buf=1;
    if ~exist('cols', 'var')
        if floor(sqrt(M))^2 ~= M
            n=ceil(sqrt(M));
            while mod(M, n)~=0 && n<1.2*sqrt(M), n=n+1; end
            m=ceil(M/n);
        else
            n=sqrt(M);
            m=n;
        end
    else
        n = cols;
        m = ceil(M/n);
    end

    array=-ones(buf+m*(sz+buf),buf+n*(sz+buf));

    if ~opt_graycolor
        array = 0.1.* array;
    end


    if ~opt_colmajor
        k=1;
        for i=1:m
            for j=1:n
                if k>M, 
                    continue; 
                end
                clim=max(abs(A(:,k)));
                if opt_normalize
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/clim;
                else
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/max(abs(A(:)));
                end
                k=k+1;
            end
        end
    else
        k=1;
        for j=1:n
            for i=1:m
                if k>M, 
                    continue; 
                end
                clim=max(abs(A(:,k)));
                if opt_normalize
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/clim;
                else
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz);
                end
                k=k+1;
            end
        end
    end

    if opt_graycolor
        h=imagesc(array,'EraseMode','none',[-1 1]);
    else
        h=imagesc(array,'EraseMode','none',[-1 1]);
    end
    axis image off

    drawnow;

	title(str);
end


