function [] = build_mask_raw()

root_dir = '../data/roads/ROADS/SegmentationClass';
target_dir = '../data/roads/ROADS/SegmentationClassRaw';
dir_struct = dir(root_dir);

for i = 3:numel(dir_struct)
    BW_disconn = rgb2gray(imread(fullfile(root_dir, dir_struct(i).name, 'disconn.png')));
    BW_other = rgb2gray(imread(fullfile(root_dir, dir_struct(i).name, 'other.png')));
    full_mask = zeros(size(BW_disconn));
    full_mask(BW_disconn == 255) = 1;
    full_mask(BW_other == 255) = 2;
    disp(strcat(dir_struct(i).name, '.png'));
    imwrite(uint8(full_mask), fullfile(target_dir, strcat(dir_struct(i).name, '.png')));
end

end