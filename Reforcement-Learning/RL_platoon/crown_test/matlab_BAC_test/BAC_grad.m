function [ grad ] = BAC_grad( episodes_store_cell, G, env_param, learning_param )
% calculate bayesian actor-critic gradient
%fisher information matrix
gamma = learning_param.gamma;
gamma2 = gamma^2;

nu = 0.1;
sig = 3;
sig2 = sig^2;
ck_xa = 1;
sigk_x = 1.3 * 0.25;   


% GLOBAL INITIALIZATION
T = 0;
m = 0;
statedic = [];
scr_dic = [];
invG_scr_dic = [];
alpha = [];
C = [];
Kinv = [];
k = [];
c = [];
z = [];

for l = 1:learning_param.train_episode_num    
%%%%%%%%%%%% INIT. PER TRIAL %%%%%%%%%%%%

    ISGOAL = 0;
    t = 1;
    T = T + 1;
    c = zeros(m,1);
    d = 0;
    s = inf;

    state = episodes_store_cell{l,1}(:,t);
    scroe = episodes_store_cell{l,2}(:,t);

    temp1 = state_kernel_kxx(state,env_param);
    invG_scr = G\scroe;
    
    temp2 = ck_xa * (scroe' * invG_scr);
    kk = temp1 + temp2;

    if (m > 0)     
        
        k = ck_xa * (scroe' * invG_scr_dic); % state-action kernel - fisherinformation kernel
        k = (k + state_kernel_kx(state,statedic,env_param))';
        
        a = Kinv * k;
        delta = kk - (k' * a);
    else
        k = [];
        a = [];
        delta = kk;
    end

    if ((m == 0) || (delta > nu))
        a_hat = a;
        h = [a; -gamma];
        a = [z; 1];
        alpha = [alpha; 0];
        C = [C, z; z', 0];        
        Kinv = [(delta * Kinv) + (a_hat * a_hat'), -a_hat;
                -a_hat'                          , 1] / delta;
        z = [z; 0];
        c = [c; 0];
        statedic = [statedic, state];
        scr_dic = [scr_dic, scroe];
        invG_scr_dic = [invG_scr_dic, invG_scr];        
        m = m + 1;
        k = [k; kk];
    end

%%%%%%%%%%%% END INITIALIZE %%%%%%%%%%%%

%%%%%%%%%%%% TIME LOOP %%%%%%%%%%%%%

    while (t < episodes_store_cell{l,3})
        state_old = state;
        k_old = k;
        kk_old = kk;
        a_old = a;
        c_old = c;
        s_old = s;
        d_old = d;
      
        r =  get_reward(state_old);        
        
        coef = (gamma * sig2) / s_old;

        if (ISGOAL == 1)   %%%%%%%%%%%% GOAL UPDATE %%%%%%%%%%%%
            dk = k_old;
            dkk = kk_old;
            h = a_old;
            c = (coef * c_old) + h - (C * dk);
            s = sig2 - (gamma * sig2 * coef) + (dk' * (c + (coef * c_old)));
            d = (coef * d_old) + r - (dk' * alpha);
        else   %%%%%%%%%%%% NON-GOAL UPDATE %%%%%%%%%%%% 
            state = episodes_store_cell{l,1}(:,t+1);
            scroe = episodes_store_cell{l,2}(:,t+1);
            
            if (state.isgoal)
                ISGOAL = 1;
                t = t - 1;
                T = T - 1;
            end
            
            
            temp1 = state_kernel_kxx(state,env_param);
              invG_scr  =  G\scroe;
            temp2 = ck_xa * (scroe' * invG_scr); 
            kk = temp1 + temp2;
            
            k = ck_xa * (scroe' * invG_scr_dic);
            k = (k + state_kernel_kx(state,statedic,env_param))';
            
            a = Kinv * k;
            delta = kk - (k' * a);

            dk = k_old - (gamma * k);
            d = (coef * d_old) + r - (dk' * alpha);
            
            if (delta > nu)   %%%%%%%%%% DELTA > NU %%%%%%%%%%
                h = [a_old; -gamma];
                dkk = (a_old' * (k_old - (2 * gamma * k))) + (gamma2 * kk);
                c = (coef * [c_old; 0]) + h - [C * dk; 0];
                s = ((1 + gamma2) * sig2) + dkk - (dk' * C * dk) + (2 * coef * c_old' * dk) - (gamma * sig2 * coef);
                alpha = [alpha; 0];
                C = [C, z; z', 0];
                statedic = [statedic, state];
                scr_dic = [scr_dic, scroe];
                invG_scr_dic = [invG_scr_dic, invG_scr];
                m = m + 1;
                Kinv = [(delta * Kinv) + (a * a'), -a;
                        -a'                      , 1] / delta;
                a = [z; 1];
                z = [z; 0];
                k = [k; kk];
            else   %%%%%%%%%% DELTA <= NU %%%%%%%%%%
                h = a_old - (gamma * a);
                dkk = h' * dk;
                c = (coef * c_old) + h - (C * dk);
                s = ((1 + gamma2) * sig2) + (dk' * (c + (coef * c_old))) - (gamma * sig2 * coef);
            end % if delta
        end % if goal
        
        % alpha update
        alpha = alpha + (c * (d / s));
        % C update
        C = C + (c * (c' / s));
        
        % update time counters
        t = t + 1;
        T = T + 1;
    end % while t
end % for trial

grad = ck_xa * (scr_dic * alpha);



end

