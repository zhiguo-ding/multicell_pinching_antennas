clear
%close all

ct = 10000; 
D = 20;
D_leng =4*D;  
height = 3; %d 
snrvec = [0:5 : 30];
sigmanoise = 10^(-10);
N_row = 1; N_col = 2; %number of rows and cols in the grid
M=N_row*N_col; %the number of cells
f = 28e9; % 28 GHz
c = 3e8; % speed of light
lambda = c/f; % free space wavelength 
eta = (c/4/pi/f)^2;
R = 1;%target data rate
eps = 2^R-1;%
Rvec = [0.1 :0.1: 1 ];
K_max = 10; %the number of CE iterations
N_CE = 1000; %the number of CE samples generated during each iteration
N_best = 10; %the number of elite samples
alpha = 0.7; %soothing parameter
CE_noise_base = 0.1;  %CE noise
P_max = 100;

%find the coornidates of conventional base stations, they are also the
%center of each region
conv_bs = [];
for j = 1 : N_row 
    for i = 1: N_col
        xcord = (i-(N_col+1)/2)*D_leng/N_col;
        ycord = (j-(N_row+1)/2)*D/N_row;
        conv_bs_temp(j,i,:) = [xcord ycord height ]; %this is mainly for testing
        conv_bs = [conv_bs; [xcord ycord height ]];
    end
end
%generate the locations of the starting and ending points of segments
seg_st = conv_bs(:,1)-D_leng/N_col/2;
seg_end = conv_bs(:,1)+D_leng/N_col/2; 
 
for rveci = 1: length(Rvec)
    R = Rvec(rveci);
    eps = 2^R-1;%
    sum1=0;sum2=0;sum3=0;
    power_conv_t = zeros(ct,1);
    power_fix_t = zeros(ct,1);
    optimal_samples_t = zeros(ct,1);
    parfor cti = 1 : ct
        %generate the users' locations
        dx = D_leng/N_col;
        dy = D/N_row;           
        loc = [];
        loc_temp=[];
        for j = 1 : N_row 
            for i = 1: N_col
                centerx = squeeze(conv_bs_temp(j,i,:)); 
                tempxcd = dx*(rand-1/2);
                tempycd = dy*(rand-1/2); %these are still centered at (0,0)
                loc_temp(j,i,:) = [ [tempxcd tempycd] + centerx(1:2)' 0];
                loc = [loc ; [ [tempxcd tempycd] + centerx(1:2)' 0]];
            end
        end
        % %for the two-user special case
        % loc(:,1) = [-2*rand;2*rand];%[-1;0.5];%temp1xd + seg_st; % shift based on the starting point of segments
        % loc(:,2) = D*rand(M,1)-D/2;  
        % loc(:,3) = zeros(M,1);  


             
        %conventional antenna at the center        
        power_conv_t(cti) = findpower(conv_bs,N_row,N_col,M,height,loc,eta,eps,sigmanoise,P_max);

      
        %fixed choice 
        loc_pin_fix = [loc(:,1) conv_bs(:,2) height*ones(M,1)];
        power_fix_t(cti) = findpower(loc_pin_fix,N_row,N_col,M,height,loc,eta,eps,sigmanoise,P_max);

        %cross-entropy method
        %mu = conv_bs(:,1)';%initial mean, we will use the covnentiona base stations x-locations 
        mu = loc(:,1)';%initial mean, we will use the users' x-locations 
        sigma = dx/2;% initial search radius, the initial one is very large
        for k = 1: K_max %K_max iterations
            samples = mu + sigma .* randn(N_CE, M);
            samples = max(samples,kron(seg_st',ones(N_CE,1)));%keep samples withint the range
            samples = min(samples,kron(seg_end',ones(N_CE,1)));%keep samples withint the range
            
            rewards = [];
            for n = 1: N_CE
                loc_pin_ce = [samples(n,:)' conv_bs(:,2) height*ones(M,1)];
                rewards(n) = rewardpower(loc_pin_ce,N_row,N_col,M,height,loc,eta,eps,sigmanoise,P_max);
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
            loc_pin_ce = [samples(n,:)' conv_bs(:,2) height*ones(M,1)];
            rewards(n) = findpower(loc_pin_ce,N_row,N_col,M,height,loc,eta,eps,sigmanoise,P_max);
        end  
        optimal_samples_t(cti) = min(rewards); 

    end
    power_conv_t(power_conv_t == P_max)=[];
    power_fix_t(power_fix_t == P_max)=[];
    optimal_samples_t(optimal_samples_t == P_max)=[];

    power_conv(rveci) = mean(power_conv_t);  
    power_fix(rveci) = mean(power_fix_t);  
    optimal_samples(rveci) = mean(optimal_samples_t);  
    % power_search(rveci) = mean(power_search_t); 

end

plot(Rvec, 10*log10(power_conv)+30, Rvec,10*log10(power_fix)+30,'-s',Rvec,10*log10(optimal_samples)+30)
legend('convtional', 'Fixed choice', 'Cross Entropy' )
% plot(Rvec, power_conv, Rvec,power_fix,'-s', ...
%     Rvec,optimal_samples, Rvec, power_search)
% legend('convtional', 'Fixed choice', 'Cross Entropy','Exahustive search')
  

function power = findpower(loc_BS,N_row,N_col,M,height,loc,eta,eps,sigmanoise,P_max)

    for m = 1 : M
        D_mat(m,:) = sum((loc(m,:)-loc_BS).^2,2)';% a distance matrix
    end    
    H_mat = eta ./ D_mat; %the channel matrix
    hmmvec = diag(H_mat);  %the channel vector
    G_mat = H_mat./hmmvec;
    G_mat = G_mat - eye(M);
    a = eps*sigmanoise./hmmvec;

    power_ori = inv(eye(M)-eps*G_mat)*a;
    if min(power_ori)<0
        power = P_max;
    else
        power = sum(inv(eye(M)-eps*G_mat)*a);
    end
end

function reward = rewardpower(loc_BS,N_row,N_col,M,height,loc,eta,eps,sigmanoise,P_max)

    for m = 1 : M
        D_mat(m,:) = sum((loc(m,:)-loc_BS).^2,2)';% a distance matrix
    end    
    H_mat = eta ./ D_mat; %the channel matrix
    hmmvec = diag(H_mat);  %the channel vector
    G_mat = H_mat./hmmvec;
    G_mat = G_mat - eye(M);
    a = eps*sigmanoise./hmmvec;

    power_ori = inv(eye(M)-eps*G_mat)*a;
    if min(power_ori)<0
        reward = -1e6;
    else
        reward = -sum(power_ori);
    end
end