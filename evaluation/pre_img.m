function im_pad = pre_img(im,crop_size,mean_rgb)
    row = size(im,1);
    col = size(im,2);
    im_pad = single(im);
    if(size(im_pad,3) < 3)
        im_r = im_pad;
        im_g = im_pad;
        im_b = im_pad;
        im_pad = cat(3,im_r,im_g,im_b);
    end
    if(row < crop_size)
        im_r = padarray(im_pad(:,:,1),[crop_size-row,0],mean_rgb(1),'post');
        im_g = padarray(im_pad(:,:,2),[crop_size-row,0],mean_rgb(2),'post');
        im_b = padarray(im_pad(:,:,3),[crop_size-row,0],mean_rgb(3),'post');
        im_pad = cat(3,im_r,im_g,im_b);
    end
    if(col < crop_size)
        im_r = padarray(im_pad(:,:,1),[0,crop_size-col],mean_rgb(1),'post');
        im_g = padarray(im_pad(:,:,2),[0,crop_size-col],mean_rgb(2),'post');
        im_b = padarray(im_pad(:,:,3),[0,crop_size-col],mean_rgb(3),'post');
        im_pad = cat(3,im_r,im_g,im_b);
    end
    im_mean = zeros(crop_size,crop_size,3,'single');
    im_mean(:,:,1) = mean_rgb(1);
    im_mean(:,:,2) = mean_rgb(2);
    im_mean(:,:,3) = mean_rgb(3);
    im_pad = single(im_pad) - im_mean;
    im_pad = im_pad(:,:,[3 2 1]);
    im_pad = permute(im_pad,[2 1 3]);
end

