%% Train %%
for i=1:11
    airfoil_data{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\', sprintf('dset_%d.mat', i-1)]);
end

gan4_cl_Cd = [];
gan4_latent = [];
for i=1:11
    gan4_cl_Cd= [gan4_cl_Cd;airfoil_data{i}.dset.Y];
    gan4_latent= [gan4_latent;airfoil_data{i}.dset.X];
end

    
save("Dataset\gan4_cl_Cd.mat", "gan4_cl_Cd")
save("Dataset\gan4_latent.mat", "gan4_latent")
