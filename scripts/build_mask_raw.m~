function [] = build_mask_raw()

ROADS_DIR = '../data/roads/ROADS';

dir_struct = dir(fullfile(ROADS_DIR, 'SegmentationClass', '*.png'));

for i = 1:numel(dir_struct)
    BW = rgb2gray(imread(fullfile(ROADS_DIR, 'SegmentationClass', dir_struct(i).name)));
    B = boundarymask(logical(BW));
    se = strel('disk', 5);
    J = imdilate(B, se);
    BW(BW == 255) = 1;
    BW(J) = 255;
    imwrite(BW, fullfile(ROADS_DIR, 'SegmentationClassRaw', dir_struct(i).name));
end

end