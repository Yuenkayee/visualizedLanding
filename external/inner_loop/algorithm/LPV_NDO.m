

classdef LPV_NDO < matlab.System
    % LPV_NDO
    % MATLAB System implementation of an LPV nonlinear disturbance observer
    % for matched input-channel disturbances in the multi-trim-point
    % helicopter linear model.
    %
    % LPV plant model:
    %   x_dot = A(v_down)*x + B(v_down)*(u + d)
    %
    % State definition:
    %   x = [u, w, q, theta, v, p, r, phi, psi]'
    %
    % Input definition:
    %   u = [cyc_lat, cyc_lon, coll, ped]'
    %
    % Scheduling variable:
    %   v_down : NED down-direction velocity, in knots
    %
    % Matched disturbance:
    %   d = [d_cyc_lat, d_cyc_lon, d_coll, d_ped]'
    %
    % Disturbance observer:
    %   d_hat = z + L(v_down)*x
    %   z_dot = -L(v_down)*(A(v_down)*x + ...
    %             B(v_down)*u_comp + B(v_down)*d_hat)
    %   u_comp = u0 - d_hat
    %
    % This module estimates matched input disturbance. It is not a full
    % state observer. The full state x, or an externally estimated x_hat,
    % must be provided as an input.
    %
    % MATLAB System block inputs:
    %   input 1: u0,     4x1 nominal control input
    %   input 2: x,      9x1 measured or estimated full state vector
    %   input 3: v_down, scalar scheduling variable in knots
    %
    % MATLAB System block outputs:
    %   output 1: u_comp, 4x1 compensated input sent to the helicopter
    %   output 2: d_hat,  4x1 estimated matched input disturbance

    properties (Nontunable)
        % Trim-point plant matrices.


        A_10 = zeros(9,9)
        B_10 = zeros(9,4)
        A_5  = zeros(9,9)
        B_5  = zeros(9,4)
        A_0  = zeros(9,9)
        B_0  = zeros(9,4)

        % Discrete sample time of the observer.

        SampleTime = 0.02

        % Observer bandwidth. If observer gains are not provided, the
        % gain at each step is constructed as
        %   L(v_down) = ObserverBandwidth * pinv(B(v_down))
        % so that L(v_down)*B(v_down) is approximately
        % ObserverBandwidth*I.

        ObserverBandwidth = 5

        % Optional user-specified observer gains at trim points.
        % Each nonempty gain must have size 4-by-9.
        % Leave all as [] to automatically construct L from B(v_down).

        ObserverGain_10 = []
        ObserverGain_5  = []
        ObserverGain_0  = []

        % Optional first-order low-pass filtering on disturbance estimate.
        % Use 0 to disable filtering.

        DisturbanceFilterBandwidth = 0

        % Optional limits for the estimated disturbance.

        EnableDisturbanceLimit = true
        DisturbanceLowerLimit = [-0.05; -0.05; -0.05; -0.05]
        DisturbanceUpperLimit = [ 0.05;  0.05;  0.05;  0.05]

        % Optional limits for the final compensated control input.


        EnableInputSaturation = false
        InputLowerLimit = [-inf; -inf; -inf; -inf]
        InputUpperLimit = [ inf;  inf;  inf;  inf]

        % If true, v_down is clamped to the interpolation range [0,10].
        % If false, linear extrapolation is used outside the range.
        
        ClampSchedulingVariable = true
    end

    properties (Access = private)
        z
        dFiltered
        UseScheduledObserverGain
    end

    methods (Access = protected)
        function setupImpl(obj)
            obj.validateParameters();
            obj.UseScheduledObserverGain = ...
                ~isempty(obj.ObserverGain_0) || ...
                ~isempty(obj.ObserverGain_5) || ...
                ~isempty(obj.ObserverGain_10);

            if obj.UseScheduledObserverGain
                if isempty(obj.ObserverGain_0) || ...
                        isempty(obj.ObserverGain_5) || ...
                        isempty(obj.ObserverGain_10)
                    error('LPV_NDO:ObserverGainSet', ...
                        ['ObserverGain_0, ObserverGain_5 and ' ...
                         'ObserverGain_10 must either all be empty or ' ...
                         'all be specified.']);
                end
            end

            obj.z = zeros(4,1);
            obj.dFiltered = zeros(4,1);
        end

        function [u_comp,d_hat] = stepImpl(obj,u0,x,v_down)
            u0 = double(u0(:));
            x = double(x(:));
            v_down = double(v_down);

            if numel(u0) ~= 4
                error('LPV_NDO:InputDimension', ...
                    'Input u0 must be a 4-by-1 vector.');
            end

            if numel(x) ~= 9
                error('LPV_NDO:StateDimension', ...
                    'Input x must be a 9-by-1 vector.');
            end

            if ~isscalar(v_down) || ~isfinite(v_down)
                error('LPV_NDO:SchedulingVariable', ...
                    'v_down must be a finite scalar in knots.');
            end

            [A_now,B_now] = obj.interpolateMatrices(v_down);
            L_now = obj.getObserverGain(v_down,B_now);
            Ts = obj.SampleTime;

            % LPV disturbance estimate from the NDO algebraic output
            % equation.
            d_raw = obj.z + L_now*x;

            % Optional low-pass filtering. This reduces noise amplification
            % at the cost of adding disturbance-estimation lag.
            if obj.DisturbanceFilterBandwidth > 0
                omega_f = obj.DisturbanceFilterBandwidth;
                alpha_f = Ts*omega_f/(1 + Ts*omega_f);
                obj.dFiltered = obj.dFiltered + ...
                    alpha_f*(d_raw - obj.dFiltered);
                d_hat = obj.dFiltered;
            else
                d_hat = d_raw;
            end

            if obj.EnableDisturbanceLimit
                d_lower = obj.DisturbanceLowerLimit(:);
                d_upper = obj.DisturbanceUpperLimit(:);
                d_hat = min(max(d_hat,d_lower),d_upper);
            end

            % Matched-disturbance compensation.
            u_comp = u0 - d_hat;

            if obj.EnableInputSaturation
                u_lower = obj.InputLowerLimit(:);
                u_upper = obj.InputUpperLimit(:);
                u_comp = min(max(u_comp,u_lower),u_upper);
            end

            % Observer state update using the scheduled A, B and L.
            % With u_comp = u0 - d_hat, the term
            % B_now*u_comp + B_now*d_hat is equivalent to B_now*u0,
            % but it is kept in this form to make the compensation
            % structure explicit.
            z_dot = -L_now*(A_now*x + B_now*u_comp + B_now*d_hat);
            obj.z = obj.z + Ts*z_dot;
        end

        function resetImpl(obj)
            obj.z = zeros(4,1);
            obj.dFiltered = zeros(4,1);
        end

        function num = getNumInputsImpl(~)
            num = 3;
        end

        function num = getNumOutputsImpl(~)
            num = 2;
        end

        function varargout = getOutputSizeImpl(~)
            varargout{1} = [4,1];
            varargout{2} = [4,1];
        end

        function varargout = getOutputDataTypeImpl(~)
            varargout{1} = 'double';
            varargout{2} = 'double';
        end

        function varargout = isOutputComplexImpl(~)
            varargout{1} = false;
            varargout{2} = false;
        end

        function varargout = isOutputFixedSizeImpl(~)
            varargout{1} = true;
            varargout{2} = true;
        end

        function sts = getSampleTimeImpl(obj)
            sts = createSampleTime(obj, ...
                'Type','Discrete', ...
                'SampleTime',obj.SampleTime);
        end

        function flag = supportsMultipleInstanceImpl(~)
            flag = true;
        end
    end

    methods (Access = private)
        function validateParameters(obj)
            if ~isequal(size(obj.A_10),[9,9]) || ...
                    ~isequal(size(obj.A_5),[9,9]) || ...
                    ~isequal(size(obj.A_0),[9,9])
                error('LPV_NDO:StateMatrixDimension', ...
                    'A_10, A_5 and A_0 must all have size 9-by-9.');
            end

            if ~isequal(size(obj.B_10),[9,4]) || ...
                    ~isequal(size(obj.B_5),[9,4]) || ...
                    ~isequal(size(obj.B_0),[9,4])
                error('LPV_NDO:InputMatrixDimension', ...
                    'B_10, B_5 and B_0 must all have size 9-by-4.');
            end

            validateattributes(obj.SampleTime,{'numeric'}, ...
                {'scalar','real','finite','positive'});

            validateattributes(obj.ObserverBandwidth,{'numeric'}, ...
                {'scalar','real','finite','positive'});

            validateattributes(obj.DisturbanceFilterBandwidth,{'numeric'}, ...
                {'scalar','real','finite','nonnegative'});

            obj.validateObserverGain(obj.ObserverGain_0,'ObserverGain_0');
            obj.validateObserverGain(obj.ObserverGain_5,'ObserverGain_5');
            obj.validateObserverGain(obj.ObserverGain_10,'ObserverGain_10');

            if obj.EnableDisturbanceLimit
                if numel(obj.DisturbanceLowerLimit) ~= 4 || ...
                        numel(obj.DisturbanceUpperLimit) ~= 4
                    error('LPV_NDO:DisturbanceLimit', ...
                        ['DisturbanceLowerLimit and ' ...
                         'DisturbanceUpperLimit must each contain ' ...
                         'four elements.']);
                end
            end

            if obj.EnableInputSaturation
                if numel(obj.InputLowerLimit) ~= 4 || ...
                        numel(obj.InputUpperLimit) ~= 4
                    error('LPV_NDO:InputLimit', ...
                        ['InputLowerLimit and InputUpperLimit must each ' ...
                         'contain four elements.']);
                end
            end

            if rank(obj.B_0) < 4 && isempty(obj.ObserverGain_0)
                warning('LPV_NDO:B0NotFullColumnRank', ...
                    ['B_0 is not full column rank. The automatic gain ' ...
                     'may not independently observe all four matched ' ...
                     'input disturbances near v_down = 0 kt.']);
            end

            if rank(obj.B_5) < 4 && isempty(obj.ObserverGain_5)
                warning('LPV_NDO:B5NotFullColumnRank', ...
                    ['B_5 is not full column rank. The automatic gain ' ...
                     'may not independently observe all four matched ' ...
                     'input disturbances near v_down = 5 kt.']);
            end

            if rank(obj.B_10) < 4 && isempty(obj.ObserverGain_10)
                warning('LPV_NDO:B10NotFullColumnRank', ...
                    ['B_10 is not full column rank. The automatic gain ' ...
                     'may not independently observe all four matched ' ...
                     'input disturbances near v_down = 10 kt.']);
            end
        end

        function validateObserverGain(~,gain,name)
            if ~isempty(gain) && ~isequal(size(gain),[4,9])
                error('LPV_NDO:ObserverGain', ...
                    '%s must be empty or have size 4-by-9.',name);
            end
        end

        function [A_now,B_now] = interpolateMatrices(obj,v_down)
            v = obj.normalizeSchedulingVariable(v_down);

            if v <= 5
                lambda = v/5;
                A_now = (1-lambda)*obj.A_0 + lambda*obj.A_5;
                B_now = (1-lambda)*obj.B_0 + lambda*obj.B_5;
            else
                lambda = (v-5)/5;
                A_now = (1-lambda)*obj.A_5 + lambda*obj.A_10;
                B_now = (1-lambda)*obj.B_5 + lambda*obj.B_10;
            end
        end

        function L_now = getObserverGain(obj,v_down,B_now)
            if obj.UseScheduledObserverGain
                L_now = obj.interpolateObserverGain(v_down);
            else
                L_now = obj.ObserverBandwidth * pinv(B_now);
            end
        end

        function L_now = interpolateObserverGain(obj,v_down)
            v = obj.normalizeSchedulingVariable(v_down);

            if v <= 5
                lambda = v/5;
                L_now = (1-lambda)*obj.ObserverGain_0 + ...
                    lambda*obj.ObserverGain_5;
            else
                lambda = (v-5)/5;
                L_now = (1-lambda)*obj.ObserverGain_5 + ...
                    lambda*obj.ObserverGain_10;
            end
        end

        function v = normalizeSchedulingVariable(obj,v_down)
            if obj.ClampSchedulingVariable
                v = min(max(v_down,0),10);
            else
                v = v_down;
            end
        end
    end
end