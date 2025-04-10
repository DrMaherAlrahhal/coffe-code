function Iout = preprocess_D(filename)

I = imread(filename);
if size(I,3)==1
    I_3D = cat(2,I,I);
else
    I_3D=I;
end
Iout=imresize(I_3D,[22,22]);
