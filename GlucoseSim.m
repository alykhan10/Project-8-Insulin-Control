%% Simulation Parameters
T = 1;              % Time step (min)
N = 60;             % Prediction horizon (min)
simTime = 300;      % Total simulation time (min)
glucoseSetpoint = 5; % Desired glucose level (mmol/L)
time = 0:T:simTime;

%% Initialize States
G = 10;              % Initial glucose (mmol/L)
I = 15;              % Initial insulin (uU/mL)
x = [G; I];          % State vector
xHist = zeros(2, length(time));
uHist = zeros(1, length(time));

%% PID Controller Parameters
Kp = 1.5;
Ki = 0.01;
Kd = 0.2;
intError = 0;
prevError = 0;

%% System Dynamics (Simplified Bergman Minimal Model)
A = [-0.01, -0.02;
      0.01, -0.05];
B = [0; 1];

%% MPC Setup (Linear Model)
Q = eye(2);              % State penalty
R = 0.01;                % Input penalty
umin = 0; umax = 10;     % Insulin pump limits

%% Main Control Loop
for k = 1:length(time)
    G = x(1);
    I = x(2);
    xHist(:, k) = x;

    % ----- PID Controller -----
    error = glucoseSetpoint - G;
    intError = intError + error * T;
    derError = (error - prevError) / T;
    pidOutput = Kp * error + Ki * intError + Kd * derError;
    prevError = error;

    % ----- MPC Controller -----
    % Predict future states using linear model
    H = []; F = [];   % Build optimization matrices
    for i = 1:N
        Phi = A^i;
        Gamma = zeros(2, N);
        for j = 1:i
            Gamma(:, j) = A^(i-j) * B;
        end
        H = [H; Phi];
        F = [F; Gamma];
    end

    % Solve quadratic program: minimize (Hx + Fu - setpoint)'Q(Hx + Fu - setpoint) + u'Ru
    % For simplicity, use only 1-step preview control:
    % u_opt = argmin (x_k+1 - setpoint)'Q(x_k+1 - setpoint) + Ru^2

    u_mpc = quadprog(F' * Q * F + R, -F' * Q * (glucoseSetpoint - H * x), [], [], [], [], umin, umax);

    % Combine PID + MPC control actions
    if isempty(u_mpc)
        u = max(0, min(umax, pidOutput)); % fallback if QP fails
    else
        u = 0.5 * pidOutput + 0.5 * u_mpc(1); % weight both controllers
    end

    % Apply control input to system
    dx = A * x + B * u;
    x = x + dx * T;

    % Log insulin pump output
    uHist(k) = u;
end

%% Plot Results
figure;
subplot(2,1,1);
plot(time, xHist(1,:), 'b', 'LineWidth', 2);
hold on; yline([4, 6], '--r');
xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
title('Glucose Regulation');
legend('Glucose','Target Range');

subplot(2,1,2);
plot(time, uHist, 'k', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('Insulin Pump Output (uU/min)');
title('Control Input (Insulin Delivery)');