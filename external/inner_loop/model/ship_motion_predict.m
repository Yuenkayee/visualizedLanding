function ship = ship_motion_predict(t, param)
%% 这是船体运动的模型
%SHIP_MOTION_MODEL  基于 RAO 与波浪谱合成的六自由度舰船运动模型
%
%   ship = SHIP_MOTION_MODEL(t)
%   ship = SHIP_MOTION_MODEL(t, param)
%
%   该函数按照 O'Reilly (1987) 附录中 “Condensed explanation of ship
%   motion model” 的思路实现：
%       1) 用波浪能量谱 Sw(w) 描述海况；
%       2) 通过 RAO 得到舰船响应谱 S_DOF(w) = Sw(w) * RAO_DOF(w)；
%       3) 将响应谱离散为若干正弦分量；
%       4) 对正弦分量求和，生成 surge/sway/heave/roll/pitch/yaw 的
%          位移、速度和加速度时间历程。
%
%   输入：
%       t     : 时间，标量或列/行向量，单位 s
%       param : 可选参数结构体。可包含字段：
%           .Hs           显著波高，单位 m，默认 3.0
%           .g            重力加速度，单位 m/s^2，默认 9.80665
%           .spectrumType 波浪谱类型，当前支持 'PM'，默认 'PM'
%           .wMin         波浪圆频率下限，rad/s，默认 0.2
%           .wMax         波浪圆频率上限，rad/s，默认 2.5
%           .nComp        正弦分量个数，默认 48
%           .seed         随机相位种子，默认 1
%           .phaseMode    'random' 或 'zero'，默认 'random'
%           .RAO          RAO 结构体，可选。若不提供，则使用内置简化 RAO。
%                         param.RAO 需要包含：
%                           .w     : RAO 频率网格，rad/s
%                           .amp   : nFreq x 6，按
%                                    [surge sway heave roll pitch yaw]
%                                    排列的 RAO 幅值平方；
%                                    平移 DOF 单位 m^2/m^2，
%                                    转动 DOF 单位 rad^2/m^2
%                           .phase : nFreq x 6，相位，rad，可选
%           .doWrapAngle  是否将角位移限制到 [-pi, pi]，默认 false
%
%   输出 ship 为结构体：
%       .t       : 时间列向量，s
%       .eta     : 六自由度位移/角位移，N x 6，
%                  [surge sway heave roll pitch yaw]，单位 [m m m rad rad rad]
%       .eta_dot : 六自由度速度/角速度，N x 6，单位 [m/s m/s m/s rad/s rad/s rad/s]
%       .eta_ddot: 六自由度加速度/角加速度，N x 6，单位 [m/s^2 ... rad/s^2]
%       .x, .y, .z, .phi, .theta, .psi  : 对应通道的便捷输出
%       .xdot, .ydot, ...               : 速度便捷输出
%       .xddot, .yddot, ...             : 加速度便捷输出
%       .component                      : 用于复现实验的频率、幅值和相位
%       .param                          : 实际采用的参数
%
%   注意：真实舰型应使用由船模试验或水动力程序得到的 RAO。
%   本文件中的默认 RAO 只是用于仿真框架调试的平滑近似，不代表特定舰型。

    if nargin < 1 || isempty(t)
        t = (0:0.01:300).';
    end
    if nargin < 2 || isempty(param)
        param = struct();
    end

    param = set_default_param(param);

    t = t(:);
    nTime = numel(t);
    dofNum = 6;

    % 频率离散。论文附录中当前实践为每个自由度使用 48 个正弦分量，
    % 这里将该数量作为默认值，同时允许用户通过 param.nComp 修改。
    wEdge = linspace(param.wMin, param.wMax, param.nComp + 1);
    w = 0.5 * (wEdge(1:end-1) + wEdge(2:end));
    dw = diff(wEdge);

    % 波浪能量谱 Sw(w)。默认采用 Pierson-Moskowitz 谱。
    Sw = wave_spectrum_pm(w, param.Hs, param.g);

    % 读取或生成 RAO，并插值到正弦分量中心频率。
    [raoAmp, raoPhase] = get_rao_on_grid(w, param);

    % 舰船响应谱：S_DOF(w) = Sw(w) * RAO_DOF(w)
    Sresp = Sw(:) .* raoAmp;
    Sresp = max(Sresp, 0);

    % 将响应谱直方图面积转换为正弦分量幅值。
    % 对单边谱有 A_n = sqrt(2 * S(w_n) * Delta_w)。
    compAmp = sqrt(2 * Sresp .* dw(:));

    % 初始相位。
    switch lower(param.phaseMode)
        case 'random'
            rng(param.seed, 'twister');
            randPhase = 2 * pi * rand(param.nComp, dofNum);
        case 'zero'
            randPhase = zeros(param.nComp, dofNum);
        otherwise
            error('ship_motion_model:BadPhaseMode', ...
                  'param.phaseMode must be ''random'' or ''zero''.');
    end
    compPhase = randPhase + raoPhase;

    eta = zeros(nTime, dofNum);
    eta_dot = zeros(nTime, dofNum);
    eta_ddot = zeros(nTime, dofNum);

    for k = 1:param.nComp
        arg = w(k) * t + compPhase(k, :);
        amp = compAmp(k, :);

        eta = eta + amp .* sin(arg);
        eta_dot = eta_dot + amp .* w(k) .* cos(arg);
        eta_ddot = eta_ddot - amp .* (w(k)^2) .* sin(arg);
    end

    if param.doWrapAngle
        eta(:, 4:6) = wrap_to_pi_local(eta(:, 4:6));
    end

    ship = struct();
    ship.t = t;
    ship.eta = eta;
    ship.eta_dot = eta_dot;
    ship.eta_ddot = eta_ddot;

    ship.x = eta(:, 1);
    ship.y = eta(:, 2);
    ship.z = eta(:, 3);
    ship.phi = eta(:, 4);
    ship.theta = eta(:, 5);
    ship.psi = eta(:, 6);

    ship.xdot = eta_dot(:, 1);
    ship.ydot = eta_dot(:, 2);
    ship.zdot = eta_dot(:, 3);
    ship.phidot = eta_dot(:, 4);
    ship.thetadot = eta_dot(:, 5);
    ship.psidot = eta_dot(:, 6);

    ship.xddot = eta_ddot(:, 1);
    ship.yddot = eta_ddot(:, 2);
    ship.zddot = eta_ddot(:, 3);
    ship.phiddot = eta_ddot(:, 4);
    ship.thetaddot = eta_ddot(:, 5);
    ship.psiddot = eta_ddot(:, 6);

    ship.component = struct();
    ship.component.w = w(:);
    ship.component.dw = dw(:);
    ship.component.Sw = Sw(:);
    ship.component.raoAmp = raoAmp;
    ship.component.raoPhase = raoPhase;
    ship.component.Sresp = Sresp;
    ship.component.amp = compAmp;
    ship.component.phase = compPhase;

    ship.param = param;
end

function param = set_default_param(param)
    param = set_field_if_missing(param, 'Hs', 3.0);
    param = set_field_if_missing(param, 'g', 9.80665);
    param = set_field_if_missing(param, 'spectrumType', 'PM');
    param = set_field_if_missing(param, 'wMin', 0.2);
    param = set_field_if_missing(param, 'wMax', 2.5);
    param = set_field_if_missing(param, 'nComp', 48);
    param = set_field_if_missing(param, 'seed', 1);
    param = set_field_if_missing(param, 'phaseMode', 'random');
    param = set_field_if_missing(param, 'doWrapAngle', false);

    if param.wMin <= 0 || param.wMax <= param.wMin
        error('ship_motion_model:BadFrequencyRange', ...
              'Require 0 < param.wMin < param.wMax.');
    end
    if param.nComp < 1 || fix(param.nComp) ~= param.nComp
        error('ship_motion_model:BadNComp', ...
              'param.nComp must be a positive integer.');
    end
    if ~strcmpi(param.spectrumType, 'PM')
        error('ship_motion_model:BadSpectrumType', ...
              'Current implementation only supports param.spectrumType = ''PM''.');
    end
end

function s = set_field_if_missing(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end

function Sw = wave_spectrum_pm(w, Hs, g)
    % Pierson-Moskowitz 谱。
    % 为了适配 SI 单位，使用与论文附录同型的形式：
    %   Sw(w) = alpha*g^2/w^5 * exp(-4*alpha*g^2/(Hs^2*w^4))
    % 其中 alpha = 0.0081。
    alpha = 0.0081;
    Sw = alpha * g^2 ./ (w.^5) .* exp(-4 * alpha * g^2 ./ (Hs^2 .* w.^4));
    Sw(~isfinite(Sw)) = 0;
end

function [raoAmp, raoPhase] = get_rao_on_grid(w, param)
    dofNum = 6;
    if isfield(param, 'RAO') && ~isempty(param.RAO)
        check_rao_struct(param.RAO);
        raoAmp = interp1(param.RAO.w(:), param.RAO.amp, w(:), 'linear', 0);
        if isfield(param.RAO, 'phase') && ~isempty(param.RAO.phase)
            raoPhase = interp1(param.RAO.w(:), param.RAO.phase, w(:), 'linear', 0);
        else
            raoPhase = zeros(numel(w), dofNum);
        end
    else
        [raoAmp, raoPhase] = default_smooth_rao(w);
    end

    if size(raoAmp, 2) ~= dofNum || size(raoPhase, 2) ~= dofNum
        error('ship_motion_model:BadRAODimension', ...
              'RAO amplitude and phase must have 6 columns.');
    end
    raoAmp = max(raoAmp, 0);
end

function check_rao_struct(RAO)
    if ~isfield(RAO, 'w') || ~isfield(RAO, 'amp')
        error('ship_motion_model:BadRAO', ...
              'param.RAO must contain fields .w and .amp.');
    end
    if numel(RAO.w) ~= size(RAO.amp, 1)
        error('ship_motion_model:BadRAO', ...
              'numel(param.RAO.w) must equal size(param.RAO.amp, 1).');
    end
    if size(RAO.amp, 2) ~= 6
        error('ship_motion_model:BadRAO', ...
              'param.RAO.amp must be nFreq x 6.');
    end
    if isfield(RAO, 'phase') && ~isempty(RAO.phase)
        if any(size(RAO.phase) ~= size(RAO.amp))
            error('ship_motion_model:BadRAO', ...
                  'param.RAO.phase must have the same size as param.RAO.amp.');
        end
    end
end

function [raoAmp, raoPhase] = default_smooth_rao(w)
    % 内置简化 RAO。
    % 六列顺序：[surge sway heave roll pitch yaw]。
    % 该近似只用于没有真实 RAO 数据时的程序调试。
    w = w(:);
    n = numel(w);
    raoAmp = zeros(n, 6);

    raoAmp(:, 1) = 0.12 * gaussian_rao(w, 0.45, 0.28) + 0.02 * gaussian_rao(w, 1.20, 0.50); % surge
    raoAmp(:, 2) = 0.16 * gaussian_rao(w, 0.75, 0.35);                                      % sway
    raoAmp(:, 3) = 1.00 * gaussian_rao(w, 0.95, 0.32) + 0.18 * gaussian_rao(w, 1.65, 0.45); % heave
    raoAmp(:, 4) = 0.030 * gaussian_rao(w, 0.75, 0.24);                                    % roll, rad^2/m^2
    raoAmp(:, 5) = 0.020 * gaussian_rao(w, 1.05, 0.26);                                    % pitch, rad^2/m^2
    raoAmp(:, 6) = 0.012 * gaussian_rao(w, 0.55, 0.30);                                    % yaw, rad^2/m^2

    % 简化相位：真实应用中应由 RAO 数据给出。
    raoPhase = zeros(n, 6);
    raoPhase(:, 1) = deg2rad_local(20);
    raoPhase(:, 2) = deg2rad_local(-15);
    raoPhase(:, 3) = deg2rad_local(0);
    raoPhase(:, 4) = deg2rad_local(35);
    raoPhase(:, 5) = deg2rad_local(-25);
    raoPhase(:, 6) = deg2rad_local(10);
end

function y = gaussian_rao(w, wc, bw)
    y = exp(-0.5 * ((w - wc) ./ bw).^2);
end

function rad = deg2rad_local(deg)
    rad = deg * pi / 180;
end

function ang = wrap_to_pi_local(ang)
    ang = mod(ang + pi, 2 * pi) - pi;
end