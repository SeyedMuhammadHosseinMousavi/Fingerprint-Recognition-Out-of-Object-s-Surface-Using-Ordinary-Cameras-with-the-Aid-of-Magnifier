clc;
clear;;
path='correct';
fileinfo = dir(fullfile(path,'*.bmp'));
filesnumber=size(fileinfo);
fsize=filesnumber(1,1);
for i = 1 : fsize
images{i} = imread(fullfile(path,fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;
% Adjust
for i = 1:fsize   
sizee{i} = imadjust(images{i},[.2 .3 0; .6 .7 1],[]);
    disp(['Adjusting intensity value :   ' num2str(i) ]);
end;
% Histogram eq
for i = 1:fsize   
sizee{i} = histeq(images{i});
    disp(['Histogram eq :   ' num2str(i) ]);
end;
% Resize
for i = 1:fsize   
sizee{i} = imresize(images{i},[900 512]);
end;
%save to disk
for i = 1:fsize   
   imwrite(sizee{i},strcat('my_new',num2str(i),'.jpg'));
end