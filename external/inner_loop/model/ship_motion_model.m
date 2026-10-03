classdef ship_motion_model < matlab.System
    %SHIP_MOTION_MODEL Six-degree-of-freedom ship-motion System object.
    %
    %   model = ship_motion_model
    %   model = ship_motion_model(paramStruct)
    %   model = ship_motion_model('Hs', 4, 'seed', 10)
    %
    %   [eta, eta_dot, eta_ddot] = model(t) evaluates the motion at the
    %   scalar simulation time t. Each output is a 1-by-6 row vector in
    %   the order [surge sway heave roll pitch yaw]. This interface can be
    %   used directly in a MATLAB System block in Simulink.
    %
    %   ship = simulate(model, tVector) evaluates a complete time vector
    %   and returns the structure produced by the earlier function version.
    %
    %   The default RAO is only a smooth approximation for framework
    %   testing. Use measured or hydrodynamically calculated RAO data for
    %   a real vessel.

    properties (Nontunable)
        Hs = 3.0                  % Significant wave height, m
        g = 9.80665               % Gravitational acceleration, m/s^2
        spectrumType = 'PM'       % Currently only 'PM' is supported
        wMin = 0.2                % Minimum angular frequency, rad/s
        wMax = 2.5                % Maximum angular frequency, rad/s
        nComp = 48                % Number of sinusoidal components
        seed = 1                  % Random-phase seed
        phaseMode = 'random'      % 'random' or 'zero'
        RAO = struct()            % Optional fields: w, amp and phase
        doWrapAngle = false       % Wrap angular positions to [-pi, pi]
    end

    properties (Access = private)
        Component
    end

    methods
        function obj = ship_motion_model(varargin)
            % Accept either the old parameter structure or name-value pairs.
            if nargin == 1 && isstruct(varargin{1})
                param = varargin{1};
                names = fieldnames(param);
                for k = 1:numel(names)
                    name = names{k};
                    if ~isprop(obj, name)
                        error('ship_motion_model:UnknownParameter', ...
                            'Unknown parameter field ''%s''.', name);
                    end
                    obj.(name) = param.(name);
                end
            elseif nargin > 0
                setProperties(obj, nargin, varargin{:});
            end
        end

        function ship = simulate(obj, t)
            %SIMULATE Evaluate a time vector without locking the object.
            obj.validateParameters();
            validateattributes(t, {'numeric'}, ...
                {'real', 'finite', 'vector'}, mfilename, 't');

            t = double(t(:));
            component = obj.buildComponents();
            [eta, etaDot, etaDDot] = obj.evaluateTime(t, component);
            ship = obj.makeOutputStructure(t, eta, etaDot, etaDDot, component);
        end

        function component = getComponent(obj)
            %GETCOMPONENT Return the frequencies, spectra, amplitudes and phases.
            obj.validateParameters();
            component = obj.buildComponents();
        end
    end

    methods (Access = protected)
        function setupImpl(obj, ~)
            obj.Component = obj.buildComponents();
        end

        function [eta, etaDot, etaDDot] = stepImpl(obj, t)
            [eta, etaDot, etaDDot] = obj.evaluateTime(double(t), obj.Component);
        end

        function resetImpl(~)
            % The model is an explicit function of time and has no state.
        end

        function validatePropertiesImpl(obj)
            obj.validateParameters();
        end

        function validateInputsImpl(~, t)
            validateattributes(t, {'numeric'}, ...
                {'real', 'finite', 'scalar'}, mfilename, 't');
        end

        function n = getNumInputsImpl(~)
            n = 1;
        end

        function n = getNumOutputsImpl(~)
            n = 3;
        end

        function [s1, s2, s3] = getOutputSizeImpl(~)
            s1 = [1, 6];
            s2 = [1, 6];
            s3 = [1, 6];
        end

        function [d1, d2, d3] = getOutputDataTypeImpl(~)
            d1 = 'double';
            d2 = 'double';
            d3 = 'double';
        end

        function [c1, c2, c3] = isOutputComplexImpl(~)
            c1 = false;
            c2 = false;
            c3 = false;
        end

        function [f1, f2, f3] = isOutputFixedSizeImpl(~)
            f1 = true;
            f2 = true;
            f3 = true;
        end

        function name = getInputNamesImpl(~)
            name = 't';
        end

        function [n1, n2, n3] = getOutputNamesImpl(~)
            n1 = 'eta';
            n2 = 'eta_dot';
            n3 = 'eta_ddot';
        end

        function icon = getIconImpl(~)
            icon = sprintf('Ship Motion\n6-DOF');
        end
    end

    methods (Access = private)
        function validateParameters(obj)
            validateattributes(obj.Hs, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, mfilename, 'Hs');
            validateattributes(obj.g, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, mfilename, 'g');
            validateattributes(obj.wMin, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive'}, mfilename, 'wMin');
            validateattributes(obj.wMax, {'numeric'}, ...
                {'real', 'finite', 'scalar', '>', obj.wMin}, mfilename, 'wMax');
            validateattributes(obj.nComp, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
                mfilename, 'nComp');
            validateattributes(obj.seed, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'integer', 'nonnegative'}, ...
                mfilename, 'seed');
            validateattributes(obj.doWrapAngle, {'logical', 'numeric'}, ...
                {'real', 'finite', 'scalar'}, mfilename, 'doWrapAngle');

            if ~(ischar(obj.spectrumType) || ...
                    (isstring(obj.spectrumType) && isscalar(obj.spectrumType))) || ...
                    ~strcmpi(char(obj.spectrumType), 'PM')
                error('ship_motion_model:BadSpectrumType', ...
                    'spectrumType must be ''PM''.');
            end
            if ~(ischar(obj.phaseMode) || ...
                    (isstring(obj.phaseMode) && isscalar(obj.phaseMode))) || ...
                    ~any(strcmpi(char(obj.phaseMode), {'random', 'zero'}))
                error('ship_motion_model:BadPhaseMode', ...
                    'phaseMode must be ''random'' or ''zero''.');
            end
            if ~isstruct(obj.RAO) || ~isscalar(obj.RAO)
                error('ship_motion_model:BadRAO', ...
                    'RAO must be a scalar structure.');
            end
            if ~isempty(fieldnames(obj.RAO))
                obj.checkRAO(obj.RAO);
            end
        end

        function component = buildComponents(obj)
            wEdge = linspace(obj.wMin, obj.wMax, obj.nComp + 1);
            w = 0.5 * (wEdge(1:end-1) + wEdge(2:end));
            dw = diff(wEdge);
            Sw = obj.waveSpectrumPM(w);
            [raoAmp, raoPhase] = obj.getRAOOnGrid(w);

            Sresp = max(Sw(:) .* raoAmp, 0);
            compAmp = sqrt(2 * Sresp .* dw(:));

            switch lower(char(obj.phaseMode))
                case 'random'
                    stream = RandStream('mt19937ar', 'Seed', double(obj.seed));
                    randPhase = 2 * pi * rand(stream, obj.nComp, 6);
                case 'zero'
                    randPhase = zeros(obj.nComp, 6);
            end

            component = struct( ...
                'w', w(:), ...
                'dw', dw(:), ...
                'Sw', Sw(:), ...
                'raoAmp', raoAmp, ...
                'raoPhase', raoPhase, ...
                'Sresp', Sresp, ...
                'amp', compAmp, ...
                'phase', randPhase + raoPhase);
        end

        function [eta, etaDot, etaDDot] = evaluateTime(obj, t, component)
            t = t(:);
            arg = t * component.w.';
            nTime = numel(t);
            eta = zeros(nTime, 6);
            etaDot = zeros(nTime, 6);
            etaDDot = zeros(nTime, 6);

            for dof = 1:6
                phase = component.phase(:, dof).';
                amp = component.amp(:, dof);
                sinArg = sin(arg + phase);
                cosArg = cos(arg + phase);
                eta(:, dof) = sinArg * amp;
                etaDot(:, dof) = cosArg * (amp .* component.w);
                etaDDot(:, dof) = -sinArg * (amp .* component.w.^2);
            end

            if logical(obj.doWrapAngle)
                eta(:, 4:6) = mod(eta(:, 4:6) + pi, 2 * pi) - pi;
            end
        end

        function Sw = waveSpectrumPM(obj, w)
            alpha = 0.0081;
            Sw = alpha * obj.g^2 ./ w.^5 .* ...
                exp(-4 * alpha * obj.g^2 ./ (obj.Hs^2 .* w.^4));
            Sw(~isfinite(Sw)) = 0;
        end

        function [raoAmp, raoPhase] = getRAOOnGrid(obj, w)
            if ~isempty(fieldnames(obj.RAO))
                raoAmp = interp1(obj.RAO.w(:), obj.RAO.amp, w(:), 'linear', 0);
                if isfield(obj.RAO, 'phase') && ~isempty(obj.RAO.phase)
                    raoPhase = interp1(obj.RAO.w(:), obj.RAO.phase, ...
                        w(:), 'linear', 0);
                else
                    raoPhase = zeros(numel(w), 6);
                end
            else
                [raoAmp, raoPhase] = obj.defaultSmoothRAO(w);
            end
            raoAmp = max(raoAmp, 0);
        end

        function checkRAO(~, rao)
            if ~isfield(rao, 'w') || ~isfield(rao, 'amp')
                error('ship_motion_model:BadRAO', ...
                    'RAO must contain fields .w and .amp.');
            end
            if ~isnumeric(rao.w) || ~isnumeric(rao.amp) || ...
                    numel(rao.w) ~= size(rao.amp, 1) || size(rao.amp, 2) ~= 6
                error('ship_motion_model:BadRAO', ...
                    'RAO.amp must be nFreq-by-6 and match RAO.w.');
            end
            if any(~isfinite(rao.w(:))) || any(diff(rao.w(:)) <= 0) || ...
                    any(~isfinite(rao.amp(:)))
                error('ship_motion_model:BadRAO', ...
                    'RAO.w must be finite and strictly increasing; RAO.amp must be finite.');
            end
            if isfield(rao, 'phase') && ~isempty(rao.phase)
                if ~isnumeric(rao.phase) || ~isequal(size(rao.phase), size(rao.amp)) || ...
                        any(~isfinite(rao.phase(:)))
                    error('ship_motion_model:BadRAO', ...
                        'RAO.phase must be finite and the same size as RAO.amp.');
                end
            end
        end

        function [raoAmp, raoPhase] = defaultSmoothRAO(~, w)
            w = w(:);
            gaussian = @(wc, bw) exp(-0.5 * ((w - wc) ./ bw).^2);
            raoAmp = zeros(numel(w), 6);
            raoAmp(:, 1) = 0.12 * gaussian(0.45, 0.28) + 0.02 * gaussian(1.20, 0.50);
            raoAmp(:, 2) = 0.16 * gaussian(0.75, 0.35);
            raoAmp(:, 3) = 1.00 * gaussian(0.95, 0.32) + 0.18 * gaussian(1.65, 0.45);
            raoAmp(:, 4) = 0.030 * gaussian(0.75, 0.24);
            raoAmp(:, 5) = 0.020 * gaussian(1.05, 0.26);
            raoAmp(:, 6) = 0.012 * gaussian(0.55, 0.30);

            raoPhase = zeros(numel(w), 6);
            raoPhase(:, 1) = 20 * pi / 180;
            raoPhase(:, 2) = -15 * pi / 180;
            raoPhase(:, 4) = 35 * pi / 180;
            raoPhase(:, 5) = -25 * pi / 180;
            raoPhase(:, 6) = 10 * pi / 180;
        end

        function ship = makeOutputStructure(obj, t, eta, etaDot, etaDDot, component)
            ship = struct();
            ship.t = t;
            ship.eta = eta;
            ship.eta_dot = etaDot;
            ship.eta_ddot = etaDDot;

            positionNames = {'x', 'y', 'z', 'phi', 'theta', 'psi'};
            velocityNames = {'xdot', 'ydot', 'zdot', 'phidot', 'thetadot', 'psidot'};
            accelerationNames = {'xddot', 'yddot', 'zddot', ...
                'phiddot', 'thetaddot', 'psiddot'};
            for k = 1:6
                ship.(positionNames{k}) = eta(:, k);
                ship.(velocityNames{k}) = etaDot(:, k);
                ship.(accelerationNames{k}) = etaDDot(:, k);
            end
            ship.component = component;
            ship.param = struct( ...
                'Hs', obj.Hs, 'g', obj.g, 'spectrumType', obj.spectrumType, ...
                'wMin', obj.wMin, 'wMax', obj.wMax, 'nComp', obj.nComp, ...
                'seed', obj.seed, 'phaseMode', obj.phaseMode, ...
                'RAO', obj.RAO, 'doWrapAngle', obj.doWrapAngle);
        end
    end
end
