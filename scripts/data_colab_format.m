files = dir('C:\Users\tca11\Desktop\CNT\Projects\DeepResection\data\png\*');
files = files(3:end);
outDir = 'C:\Users\tca11\Desktop\CNT\Projects\DeepResection\data\Unet_trial';

test = [1:2 15:19];
validation = [3:4 20:24];
train = setdiff(1:length(files),[test validation]);

for i = train
folder = [files(i).folder filesep files(i).name];
subject2colab(folder, outDir, 0)
end

for i = test
folder = [files(i).folder filesep files(i).name];
subject2colab(folder, outDir, 1)
end

for i = validation
folder = [files(i).folder filesep files(i).name];
subject2colab(folder, outDir, 2)
end