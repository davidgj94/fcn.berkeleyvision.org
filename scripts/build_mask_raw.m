function [] = build_mask_raw()

root_dir = '../data/roads/ROADS/SegmentationClass';
target_dir = '../data/roads/ROADS/SegmentationClassRaw';
dir_struct = dir(root_dir);

for i = 3:numel(dir_struct)
    BW_other = rgb2gray(imread(fullfile(root_dir, dir_struct(i).name, 'other.png')));
    full_mask = zeros(size(BW_other));
    full_mask(BW_other == 255) = 2;
    if exist(fullfile(root_dir, dir_struct(i).name, 'disconn.png'))
        BW_disconn = rgb2gray(imread(fullfile(root_dir, dir_struct(i).name, 'disconn.png')));
        full_mask(BW_disconn == 255) = 1;
    end
    if exist(fullfile(root_dir, dir_struct(i).name, 'disconn-short.png'))
        BW_disconn_short = rgb2gray(imread(fullfile(root_dir, dir_struct(i).name, 'disconn-short.png')));
        full_mask(BW_disconn_short == 255) = 3;
    end
    disp(strcat(dir_struct(i).name, '.png'));
    imwrite(uint8(full_mask), fullfile(target_dir, strcat(dir_struct(i).name, '.png')));
end

end