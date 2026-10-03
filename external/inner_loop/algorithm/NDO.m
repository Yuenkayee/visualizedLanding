

classdef NDO < matlab.System
    % NDO
    % MATLAB System implementation of a nonlinear disturbance observer
    % for matched input-channel disturbances in the helicopter model
    %
    % Plant model:
    %   x_dot = A*x + B*(u + d)
    %
    % State definition:
    %   x = [u, w, q, theta, v, p, r, phi, psi]'
    %
    % Input definition:
    %   u = [cyc_lat, cyc_lon, coll, ped]'
    %
    % Matched disturbance:
    %   d = [d_cyc_lat, d_cyc_lon, d_coll, d_ped]'
    %
    % Disturbance observer:
    %   d_hat = z + L*x
    %   z_dot = -L*(A*x + B*u_comp + B*d_hat)
    %   u_comp = u0 - d_hat
    %
    % Inputs
    %   x  : 9x1 measured or estimated full state vector
    %   u0 : 4x1 nominal control input from H_inf
    %
    % Outputs
    %   u_comp : 4x1 compensated input sent to the helicopter
    %   d_hat  : 4x1 estimated matched input disturbance

    properties (Nontunable)
        % Continuous-time linear plant matrices.

        PlantA = zeros(9,9)
        PlantB = zeros(9,4)

        % Discrete sample time of the observer.
        
        SampleTime = 0.02

        % Observer bandwidth. If ObserverGain is empty, L is constructed as
        %   L = ObserverBandwidth * pinv(B)
        % so that L*B is approximately ObserverBandwidth*I.

        ObserverBandwidth = 5

        % Optional user-specified observer gain.
        % Expected size: 4-by-9.
        % Leave it as [] to automatically construct L from PlantB.

        ObserverGain = []

        % Optional first-order low-pass filtering on disturbance estimate.
        % Use 0 to disable filtering. A value in [2,20] is often useful
        % when state measurements contain noise.

        DisturbanceFilterBandwidth = 0

        % Optional limits for the estimated disturbance.

        EnableDisturbanceLimit = true
        DisturbanceLowerLimit = [-0.05; -0.05; -0.05; -0.05]
        DisturbanceUpperLimit = [ 0.05;  0.05;  0.05;  0.05]

        % Optional limits for the final compensated control input.

        EnableInputSaturation = false
        InputLowerLimit = [-inf; -inf; -inf; -inf]
        InputUpperLimit = [ inf;  inf;  inf;  inf]
    end

    properties (Access = private)
        L
        z
        dFiltered
    end

    methods (Access = protected)
        function setupImpl(obj)
            obj.validateParameters();

            if isempty(obj.ObserverGain)
                % Use pinv instead of (B'*B)\B' so the observer can still
                % initialize when B is close to rank-deficient. A warning is
                % issued below if B does not have full column rank.
                obj.L = obj.ObserverBandwidth * pinv(obj.PlantB);
            else
                obj.L = obj.ObserverGain;
            end

            obj.z = zeros(4,1);
            obj.dFiltered = zeros(4,1);
        end

        function [u_comp,d_hat] = stepImpl(obj,u0,x)
            x = double(x(:));
            u0 = double(u0(:));

            if numel(x) ~= 9
                error('NDO:StateDimension', ...
                    'Input x must be a 9-by-1 vector.');
            end

            if numel(u0) ~= 4
                error('NDO:InputDimension', ...
                    'Input u0 must be a 4-by-1 vector.');
            end

            A = obj.PlantA;
            B = obj.PlantB;
            L = obj.L;
            Ts = obj.SampleTime;

            % Disturbance estimate from the NDO algebraic output equation.
            d_raw = obj.z + L*x;

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

            % Observer state update. The equation is written using the
            % compensated command u_comp and the estimated disturbance
            % d_hat. Under x_dot = A*x + B*(u_comp + d), the disturbance
            % estimation error approximately follows e_d_dot = -L*B*e_d
            % for slowly varying matched disturbances.
            z_dot = -L*(A*x + B*u_comp + B*d_hat);
            obj.z = obj.z + Ts*z_dot;
        end

        function resetImpl(obj)
            obj.z = zeros(4,1);
            obj.dFiltered = zeros(4,1);
        end

        function num = getNumInputsImpl(~)
            num = 2;
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
            if ~isequal(size(obj.PlantA),[9,9])
                error('NDO:PlantA', ...
                    'PlantA must have size 9-by-9.');
            end

            if ~isequal(size(obj.PlantB),[9,4])
                error('NDO:PlantB', ...
                    'PlantB must have size 9-by-4.');
            end

            validateattributes(obj.SampleTime,{'numeric'}, ...
                {'scalar','real','finite','positive'});

            validateattributes(obj.ObserverBandwidth,{'numeric'}, ...
                {'scalar','real','finite','positive'});

            validateattributes(obj.DisturbanceFilterBandwidth,{'numeric'}, ...
                {'scalar','real','finite','nonnegative'});

            if ~isempty(obj.ObserverGain) && ...
                    ~isequal(size(obj.ObserverGain),[4,9])
                error('NDO:ObserverGain', ...
                    'ObserverGain must be empty or have size 4-by-9.');
            end

            if obj.EnableDisturbanceLimit
                if numel(obj.DisturbanceLowerLimit) ~= 4 || ...
                        numel(obj.DisturbanceUpperLimit) ~= 4
                    error('NDO:DisturbanceLimit', ...
                        ['DisturbanceLowerLimit and ' ...
                         'DisturbanceUpperLimit must each contain ' ...
                         'four elements.']);
                end
            end

            if obj.EnableInputSaturation
                if numel(obj.InputLowerLimit) ~= 4 || ...
                        numel(obj.InputUpperLimit) ~= 4
                    error('NDO:InputLimit', ...
                        ['InputLowerLimit and InputUpperLimit must each ' ...
                         'contain four elements.']);
                end
            end

            if rank(obj.PlantB) < 4 && isempty(obj.ObserverGain)
                warning('NDO:BNotFullColumnRank', ...
                    ['PlantB is not full column rank. The automatic gain ' ...
                     'L = ObserverBandwidth*pinv(B) may not provide ' ...
                     'independent observation of all four matched ' ...
                     'input disturbances. Consider specifying ' ...
                     'ObserverGain manually.']);
            end
        end
    end
end