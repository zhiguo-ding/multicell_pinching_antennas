clear
%close all

ct = 10000; 
D = 20;
D_leng =4*D;  
height = 3; %d 
snrvec = [0:5 : 30];
sigmanoise = 10^(-10);
M=2; %the number of cells
f = 28e9; % 28 GHz
c = 3e8; % speed of light
lambda = c/f; % free space wavelength 
eta = (c/4/pi/f)^2;
R = 1;%target data rate
eps = 2^R-1;%
Rvec = [0.1 :0.1: 1];
K_max = 10; %the number of CE iterations
N_CE = 100; %the number of CE samples generated during each iteration
N_best = 10; %the number of elite samples
alpha = 0.7; %soothing parameter
CE_noise_base = 0.1;  %CE noise

%generate the locations of the starting and ending points of segments
temp1 = [0: D_leng/M: (M-1)*D_leng/M]';
seg_st = temp1 - D_leng/2;
temp2 = [D_leng/M: D_leng/M: D_leng]';
seg_end = temp2 - D_leng/2;

P_max=100;

%conventional antenna
conv_bs = [seg_st+D_leng/M/2 zeros(M,1) height*ones(M,1)];
for rveci = 1: length(Rvec)
    R = Rvec(rveci);
    eps = 2^R-1;%
    sum1=0;sum2=0;sum3=0;
    for cti = 1 : ct
        %generate the users' locations
        loc = zeros(M,3);    
        temp1xd = D_leng/M*rand(M,1);
        loc(:,1) = [-2*rand;2*rand];%[-1;0.5];%temp1xd + seg_st; % shift based on the starting point of segments
        loc(:,2) = D*rand(M,1)-D/2;  
        loc(:,3) = zeros(M,1);  

        %conventional antenna at the center
        power_conv_t(cti) = findpower(conv_bs(1,1),conv_bs(2,1),height,loc,eta,eps,sigmanoise); 
        if power_conv_t(cti)<0
            power_conv_t(cti) = P_max;
        end        

        %fixed choice 
        power_fix_t(cti) = findpower(loc(1,1),loc(2,1),height,loc,eta,eps,sigmanoise); 
        if power_fix_t(cti)<0
            power_fix_t(cti) = P_max;
        end  

        %cross-entropy method
        mu = loc(:,1)';%initial mean, we will use the users' x-locations 
        sigma = D_leng/4;% initial search radius, the initial one is very large
        for k = 1: K_max %K_max iterations
            samples = mu + sigma .* randn(N_CE, M);
            samples = max(samples,kron(seg_st',ones(N_CE,1)));%keep samples withint the range
            samples = min(samples,kron(seg_end',ones(N_CE,1)));%keep samples withint the range
            
            for n = 1: N_CE
                rewards(n) = -findpower(samples(n,1),samples(n,2),height,loc,eta,eps,sigmanoise);
                if rewards(n)>0 %infeasible case
                    rewards(n) = -1e6;
                end
            end
             
            %find those elite samples
            [idx, idy] = sort(rewards, 'descend');
            elite_samples = samples(idy(1:N_best), :);

            
            % update the mean and variance 
            mu_new = mean(elite_samples);
            sigma_new = std(elite_samples);
            
            mu = alpha * mu_new + (1 - alpha) * mu;%smoothing step 
            CE_noise = CE_noise_base * (0.9^k); %noise parameter update
            sigma = alpha * sigma_new + (1 - alpha) * sigma + CE_noise;
        end
        %find the best solution
        for n = 1: N_CE
            rewards(n) = findpower(samples(n,1),samples(n,2),height,loc,eta,eps,sigmanoise);
        end 
        optimal_samples_t(cti) = min(rewards);
        if optimal_samples_t(cti)<0
            optimal_samples_t(cti) = P_max;
        end  
  
        %pinching antennas, search based optimal
        stepz = D_leng/M/100;
        range1 = [seg_st(1): stepz: seg_end(1)]; %the search ranges
        range2 = [seg_st(2): stepz: seg_end(2)];
        for ix1 = 1 : length(range1)
            for ix2 = 1 : length(range2) 
                poweri(ix1,ix2) = findpower(range1(ix1),range2(ix2),height,loc,eta,eps,sigmanoise);  
                if poweri(ix1,ix2)<0
                    poweri(ix1,ix2) = inf;%avoid this is to be selected
                end
            end            
        end 
        [m, idx] = min(poweri(:));
        [row, col] = ind2sub(size(poweri), idx);
        power_search_t(cti) = poweri(row,col); 
        if power_search_t(cti)<0
            power_search_t(cti) = P_max;
        end  

        %just to maximize the ratio individually
        for ix1 = 1 : length(range1) 
            loc_pin1 = [range1(ix1) 0 height];  
            %the four channel gains
            d11z = sum((loc(1,:)-loc_pin1).^2);
            d21z = sum((loc(2,:)-loc_pin1).^2);
            ratiodistance1(ix1) = d21z/d11z;     
        end
        [mx, idxx] = max(ratiodistance1); 
        pin1_opt = range1(idxx);  
        for ix1 = 1 : length(range2) 
            loc_pin2 = [range2(ix1) 0 height];  
            %the four channel gains
            d12 = sum((loc(1,:)-loc_pin2).^2);
            d22 = sum((loc(2,:)-loc_pin2).^2);
            ratiodistance2(ix1) = d12/d22;     
        end
        [mx, idxx] = max(ratiodistance2(:)); 
        pin2_opt = range2(idxx);  
        power_ratio_t(cti) = findpower(pin1_opt,pin2_opt,height,loc,eta,eps,sigmanoise);
        if power_ratio_t(cti)<0
            power_ratio_t(cti) = P_max;
        end  

        %analytical expression for the ratio based optimization
        x1 = loc(1,1); x2 = loc(2,1); y1 = loc(1,2); y2 = loc(2,2);
        %user1's pinching antenna
        gamma1 = x2-x1; gamma2 = -(x2-x1)*(x1+x2)+y1^2-y2^2;
        gamma3 = (x2-x1)*x1*x2-(y1^2+height^2)*x2+(y2^2+height^2)*x1;
        pin_cand = [(-gamma2+sqrt(gamma2^2-4*gamma1*gamma3))/2/gamma1;
            (-gamma2-sqrt(gamma2^2-4*gamma1*gamma3))/2/gamma1];
        keepIndices = (pin_cand >= -D_leng/2 & pin_cand <= x1);%find index of feasible elements,-D_leng/2< .. <x1
        pin1_cand = pin_cand(keepIndices);
        pin1_cand = [pin1_cand -D_leng/2 x1]; %include the boundary points
        powerx1 = ((pin1_cand-x2).^2+y2^2+height^2)./...
            ((pin1_cand-x1).^2+y1^2+height^2);
        [temp1, idx1] = max(powerx1);
        pin1_rat = pin1_cand(idx1);
         %user2's pinching antenna %same stationary points
        keepIndices = (pin_cand <= D_leng/2 & pin_cand >= x2);%find index of feasible elements,-D_leng/2< .. <x1
        pin2_cand = pin_cand(keepIndices);
        pin2_cand = [pin2_cand D_leng/2 x2]; %include the boundary points
        powerx2 = ((pin2_cand-x1).^2+y1^2+height^2)./...
            ((pin2_cand-x2).^2+y2^2+height^2);
        [temp2, idx2] = max(powerx2);
        pin2_rat = pin2_cand(idx2);
        power_ratioaly_t(cti) = findpower(pin1_rat,pin2_rat,height,loc,eta,eps,sigmanoise); 
        if power_ratioaly_t(cti)<0
            power_ratioaly_t(cti) = P_max;
        end  


    end
    power_conv(rveci) = mean(power_conv_t);  
    power_fix(rveci) = mean(power_fix_t);  
    optimal_samples(rveci) = mean(optimal_samples_t);  
    power_search(rveci) = mean(power_search_t); 
    power_ratio(rveci) = mean(power_ratio_t); 
    power_ratioaly(rveci) = mean(power_ratioaly_t); 

end

plot(Rvec, 10*log10(power_conv)+30, Rvec,10*log10(power_fix)+30,'-s',Rvec,10*log10(optimal_samples)+30 ...
    ,Rvec,10*log10(power_search)+30,Rvec,10*log10(power_ratio)+30,Rvec,10*log10(power_ratioaly)+30)

%plot(Rvec, 10*log10(power_conv), Rvec,10*log10(power_fix),'-s',Rvec,10*log10(optimal_samples))

%plot(Rvec, power_conv, Rvec,power_fix,'-s', ...
%    Rvec,optimal_samples, '-o',Rvec, power_search,'-*')
legend('convtional', 'Fixed choice', 'Cross Entropy','Exahustive search', ...
    'ratio based simulation', 'ratio analytical')
  

function sumpower = overpower(h11,h12,h21,h22,eps,sigmanoise)
     sumpower = eps*sigmanoise*(h11+h22+eps*(h21 + h12))/...
         (h11*h22 - eps^2*h21*h12);
end

function power = findpower(pin1,pin2,height,loc,eta,eps,sigmanoise)
    loc_pin1 = [pin1 0 height];
    loc_pin2 = [pin2 0 height];
    %the four channel gains
    d11 = sum((loc(1,:)-loc_pin1).^2);
    h11 = eta / d11; % Calculate the channel gain
    d12 = sum((loc(1,:)-loc_pin2).^2);
    h12 = eta / d12; % Calculate the channel gain
    d21 = sum((loc(2,:)-loc_pin1).^2);
    h21 = eta / d21; % Calculate the channel gain
    d22 = sum((loc(2,:)-loc_pin2).^2);
    h22 = eta / d22; % Calculate the channel gain 
    power = overpower(h11,h12,h21,h22,eps,sigmanoise);
end
