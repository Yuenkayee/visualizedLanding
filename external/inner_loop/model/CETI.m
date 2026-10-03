

classdef CETI < matlab.System
    % CETI  Control-equivalent turbulence input model for helicopter matched disturbance.
    %
    % This MATLAB System block implements the CETI transfer functions for
    % the four helicopter input channels
    %
    %   du_CETI = [theta_c; theta_s; theta_0; theta_T]
    %           = [cyc_lat; cyc_lon; coll; ped]
    %
    % where all four equivalent control inputs are output in deg. The input
    % omega_n is assumed to be a zero-mean unit-variance white-noise signal.
    %
    % Continuous CETI model:
    %   theta_c / omega_n = Kc / (s + 2*Uinf/Rm)
    %   theta_s / omega_n = Ks / (s + 2*Uinf/Rm)
    %   theta_0 / omega_n = K0*(s + 33.91*Uinf/Rm) /
    %       ((s + 1.46*Uinf/Rm)*(s + 9.45*Uinf/Rm))
    %   theta_T / omega_n = KT / (s + Uinf/Rt)
    %
    % The continuous transfer functions are discretized internally by the
    % Tustin transform, so this block does not require the Control System
    % Toolbox and can be used directly in Simulink as a MATLAB System block.

    %#codegen

    properties
        % Turbulence intensity [m/s]
        sigma_w = 3

        % Mean wind speed [m/s]
        Uinf = 15

        % Main rotor radius [m]
        Rm = 8.18

        % Tail rotor radius [m]
        Rt = 1.68

        % Sample time [s]
        Ts = 0.01
    end

    properties(DiscreteState)
        % Previous white-noise input for first-order main-rotor filters
        wnPrev_m

        % Previous white-noise input for tail-rotor filter
        wnPrev_t

        % Previous collective-filter inputs
        wnPrev_0_1
        wnPrev_0_2

        % Previous collective-filter outputs
        theta0Prev1
        theta0Prev2

        % Previous first-order filter outputs
        thetaCPrev
        thetaSPrev
        thetaTPrev
    end

    properties(Access = private)
        % First-order filter coefficients
        am
        at
        alpha_m
        alpha_t
        beta_c
        beta_s
        beta_t

        % Second-order collective filter coefficients
        den0_1
        den0_2
        num0_0
        num0_1
        num0_2
    end

    methods(Access = protected)
        function setupImpl(obj)
            updateCoefficients(obj);
        end

        function du_CETI = stepImpl(obj, omega_n)
            % stepImpl returns the matched disturbance vector in deg.
            % Input:
            %   omega_n : zero-mean unit-variance white-noise sample
            % Output:
            %   du_CETI : [4 x 1] control-equivalent turbulence input, deg

            wn = double(omega_n);

            theta_c = obj.alpha_m * obj.thetaCPrev + ...
                obj.beta_c * (wn + obj.wnPrev_m);

            theta_s = obj.alpha_m * obj.thetaSPrev + ...
                obj.beta_s * (wn + obj.wnPrev_m);

            theta_T = obj.alpha_t * obj.thetaTPrev + ...
                obj.beta_t * (wn + obj.wnPrev_t);

            theta_0 = obj.num0_0 * wn + obj.num0_1 * obj.wnPrev_0_1 + ...
                obj.num0_2 * obj.wnPrev_0_2 - ...
                obj.den0_1 * obj.theta0Prev1 - ...
                obj.den0_2 * obj.theta0Prev2;

            obj.wnPrev_m = wn;
            obj.wnPrev_t = wn;

            obj.wnPrev_0_2 = obj.wnPrev_0_1;
            obj.wnPrev_0_1 = wn;

            obj.thetaCPrev = theta_c;
            obj.thetaSPrev = theta_s;
            obj.thetaTPrev = theta_T;

            obj.theta0Prev2 = obj.theta0Prev1;
            obj.theta0Prev1 = theta_0;

            du_CETI = [theta_c; theta_s; theta_0; theta_T];
        end

        function resetImpl(obj)
            obj.wnPrev_m = 0;
            obj.wnPrev_t = 0;
            obj.wnPrev_0_1 = 0;
            obj.wnPrev_0_2 = 0;

            obj.thetaCPrev = 0;
            obj.thetaSPrev = 0;
            obj.thetaTPrev = 0;
            obj.theta0Prev1 = 0;
            obj.theta0Prev2 = 0;
        end

        function processTunedPropertiesImpl(obj)
            updateCoefficients(obj);
        end

        function sts = getSampleTimeImpl(obj)
            sts = createSampleTime(obj, 'Type', 'Discrete', ...
                'SampleTime', obj.Ts);
        end

        function out = getOutputSizeImpl(~)
            out = [4 1];
        end

        function out = getOutputDataTypeImpl(~)
            out = 'double';
        end

        function out = isOutputComplexImpl(~)
            out = false;
        end

        function out = isOutputFixedSizeImpl(~)
            out = true;
        end

        function [sz, dt, cp] = getDiscreteStateSpecificationImpl(~, name)
            % All DiscreteState properties are real scalar double states.
            switch name
                case {'wnPrev_m', 'wnPrev_t', ...
                        'wnPrev_0_1', 'wnPrev_0_2', ...
                        'theta0Prev1', 'theta0Prev2', ...
                        'thetaCPrev', 'thetaSPrev', 'thetaTPrev'}
                    sz = [1 1];
                    dt = 'double';
                    cp = false;
                otherwise
                    sz = [1 1];
                    dt = 'double';
                    cp = false;
            end
        end

        function icon = getIconImpl(~)
            icon = sprintf('CETI');
        end
    end

    methods(Access = private)
        function updateCoefficients(obj)
            validateattributes(obj.sigma_w, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, '', 'sigma_w');
            validateattributes(obj.Uinf, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, '', 'Uinf');
            validateattributes(obj.Rm, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, '', 'Rm');
            validateattributes(obj.Rt, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, '', 'Rt');
            validateattributes(obj.Ts, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, '', 'Ts');

            sig = double(obj.sigma_w);
            U = double(obj.Uinf);
            Rm_ = double(obj.Rm);
            Rt_ = double(obj.Rt);
            Ts_ = double(obj.Ts);

            c = 2 / Ts_;

            obj.am = 2 * U / Rm_;
            obj.at = U / Rt_;

            Kc = 0.837 * sig^(-0.6265) * sqrt(sig^2 * U / (pi * Rm_));
            Ks = 1.702 * sig^(-0.6265) * sqrt(sig^2 * U / (pi * Rm_));
            KT = 1.573 * sig^(-0.6493) * sqrt(sig^2 * U / (pi * Rt_));

            obj.alpha_m = (c - obj.am) / (c + obj.am);
            obj.alpha_t = (c - obj.at) / (c + obj.at);

            obj.beta_c = Kc / (c + obj.am);
            obj.beta_s = Ks / (c + obj.am);
            obj.beta_t = KT / (c + obj.at);

            K0 = 0.1486 * sig^(-0.7069) * sqrt(3 * sig^2 * U / (pi * Rm_));
            b0 = 33.91 * U / Rm_;
            a1 = 1.46 * U / Rm_;
            a2 = 9.45 * U / Rm_;

            % Continuous collective filter:
            %   G0(s) = K0*(s + b0) / (s^2 + d1*s + d0)
            d1 = a1 + a2;
            d0 = a1 * a2;

            % Tustin substitution s = c*(z - 1)/(z + 1).
            % After multiplying by (z + 1)^2:
            %   numerator   = K0*((c + b0)z^2 + 2*b0*z + (b0 - c))
            %   denominator = (c^2 + d1*c + d0)z^2 + ...
            %                 2*(d0 - c^2)z + (c^2 - d1*c + d0)
            A0 = c^2 + d1 * c + d0;
            A1 = 2 * (d0 - c^2);
            A2 = c^2 - d1 * c + d0;

            B0 = K0 * (c + b0);
            B1 = K0 * (2 * b0);
            B2 = K0 * (b0 - c);

            obj.den0_1 = A1 / A0;
            obj.den0_2 = A2 / A0;
            obj.num0_0 = B0 / A0;
            obj.num0_1 = B1 / A0;
            obj.num0_2 = B2 / A0;
        end
    end
end