s = load('Z.mat');
z = s.Z;
total = 0;
total_mass = 0;
filters = 20;
all_mass = 0;
for digit = 1:100
    count = 0;
    mass = 0;
    all = zeros(1,10);
    for i = 1:filters
        c = 0;
        for j = 1:32
            for k = 1:32
                all_mass = all_mass + abs(z(digit,i,j,k));
                if z(digit,i,j,k) > 0.07 || z(digit,i,j,k) < -0.07
                    total_mass = total_mass + abs(z(digit,i,j,k));
                    count = count + 1;
                    c = c + 1;
                end
            end
        end
        all(i) = c;
    end
    total = total + count;
    %total_mass = total_mass + mass;
end
disp('TOTAL SPARSITY');
disp(total/100);
disp('CAPTURED MASS');
disp(total_mass);
disp('AVERAGE MASS');
disp(total_mass/total);
disp('TOTAL MASS');
disp(all_mass);