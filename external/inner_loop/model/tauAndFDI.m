classdef tauAndFDI < matlab.System
    %TAUANDFDI Control and wave-excitation loads for ship_motion_model_2.
    %
    % Inputs:
    %   t   - simulation time, s
    %   eta - [x;y;z;phi;theta;psi], 6-by-1
    %   nu  - [u;v;w;p;q;r], 6-by-1
    %   velocityCommandNED - [V_N;V_E;V_D], 3-by-1, m/s
    % Outputs:
    %   tau  - generalized cruise/stabilizer control load, 6-by-1
    %   F_DI - first-order incident/diffraction wave loads, 6-by-1
    %
    % Default sea case: Hs = 5 m for a Fassmer OPV 70 reference ship. A relative
    % wave direction of 180 deg
    % denotes head seas, 90 deg starboard beam seas, and 0 deg following
    % seas.  The default is 150 deg oblique-head seas.
    % The published ship speed range is 20-26 kn; NED speed commands above
    % 26 kn are accepted for stress testing but are outside the reference
    % vessel's published operating envelope.
    %
    % The wave elevation follows a discretized Pierson-Moskowitz spectrum.
    % Deep-water dispersion and the vessel speed give
    %   k = omega^2/g
    %   omega_encounter = omega - k*U*cos(beta).
    % In the absence of panel/strip-theory excitation-force RAOs, F_DI is
    % estimated with Froude-Krylov translational loads, hydrostatic wave-
    % slope moments, and finite-hull spatial averaging.  The coefficients
    % below are deliberately exposed for calibration.
    %
    % tau uses nominal-model dynamic inversion plus bounded state feedback.
    % This is an ideal cruise-control allocation: tau(2:6) represents the
    % combined action of rudder, fins, trim devices and other stabilizers,
    % rather than a detailed actuator model.

    properties (Nontunable)
        % Vessel and environment

        ShipLength = 70.2

        ShipBeam = 11.0

        ShipDraught = 3.5
        DisplacementMass = 1.2e6
        WaterplaneCoefficient = 0.75
        TransverseGM = 1.5
        LongitudinalGM = 200
        WaterDensity = 1025
        Gravity = 9.80665

        % Operating point and model-based cruise controller


        DesiredHeading = 0
        DesiredRollAngle = 0
        DesiredPitchAngle = 0
        HeadingCommandSpeedThreshold = 0.10
        TranslationTimeConstant = [3; 4; 2.5]
        NominalRestoringEquilibrium = zeros(6, 1)

        % Rotational closed-loop target dynamics. Entries 1:3 are retained
        % as zeros so the vector follows the six-DOF state ordering.

        ControlNaturalFrequency = [0; 0; 0; 0; 0; 0.10]
        ControlDampingRatio = [0; 0; 0; 0; 0; 1.0]

        % Fraction of each measured/modelled wave load cancelled by tau.
        % Values below one retain realistic residual wave motion.

        WaveFeedforwardGain = [0.95; 0.85; 0.50; 0; 0; 0.80]

        % Generalized actuator limits [X;Y;Z;K;M;N].  Negative surge load
        % represents braking/reverse thrust required for strict speed hold.

        % Roll and pitch limits of +/-1 N*m effectively disable active
        % stabilization so the passive hull response is visible.
        MinimumControlLoad = [-0.25e6; -0.5e6; -8e6; -1; -1; -15e6]
        MaximumControlLoad = [ 0.60e6;  0.5e6;  8e6;  1;  1;  15e6]

        % Irregular wave definition


        SignificantWaveHeight = 5
        PeakPeriod = 11.2
        RelativeWaveDirectionDeg = 150
        MinimumWaveFrequency = 0.20
        MaximumWaveFrequency = 2.00
        NumberOfWaveComponents = 64
        RandomSeed = 7

        % Approximate incident+diffraction excitation coefficients

        DiffractionScale = 1.10
        SurgeExcitationCoefficient = 0.25
        SwayExcitationCoefficient = 0.55
        HeaveExcitationCoefficient = 0.70
        RollExcitationCoefficient = 1.00
        PitchExcitationCoefficient = 1.00
        YawExcitationCoefficient = 0.18
    end

    properties (Access = private)
        WaveComponent
    end

    methods
        function obj = tauAndFDI(varargin)
            if nargin == 1 && isstruct(varargin{1})
                parameters = varargin{1};
                names = fieldnames(parameters);
                for k = 1:numel(names)
                    name = names{k};
                    if ~isprop(obj, name)
                        error('tauAndFDI:UnknownParameter', ...
                            'Unknown parameter field ''%s''.', name);
                    end
                    obj.(name) = parameters.(name);
                end
            elseif nargin > 0
                setProperties(obj, nargin, varargin{:});
            end
        end

        function component = getWaveComponent(obj)
            %GETWAVECOMPONENT Return the deterministic random-wave grid.
            obj.validateParameters();
            component = obj.buildWaveComponents();
        end

        function thrust = steadyThrust(obj, velocityCommandNED)
            %STEADYTHRUST Nominal surge thrust for an NED velocity command.
            obj.validateParameters();
            obj.validateVector3(velocityCommandNED, 'velocityCommandNED');
            speed = norm(double(velocityCommandNED(1:2)));
            nominalVelocity = [speed; zeros(5, 1)];
            [~, ~, damping, ~] = obj.nominalDynamics(nominalVelocity);
            thrust = damping(1, 1) * speed;
        end
    end

    methods (Access = protected)
        function setupImpl(obj, ~, ~, ~, ~)
            obj.validateParameters();
            obj.WaveComponent = obj.buildWaveComponents();
        end

        function [tau, F_DI] = stepImpl(obj, t, eta, nu, velocityCommandNED)
            eta = double(eta(:));
            nu = double(nu(:));
            velocityCommandNED = double(velocityCommandNED(:));
            F_DI = obj.waveLoad(double(t), obj.WaveComponent, ...
                velocityCommandNED);
            tau = obj.controlLoad(eta, nu, F_DI, velocityCommandNED);
        end

        function resetImpl(~)
            % Random phases are fixed by RandomSeed; no dynamic state.
        end

        function validatePropertiesImpl(obj)
            obj.validateParameters();
        end

        function validateInputsImpl(obj, t, eta, nu, velocityCommandNED)
            validateattributes(t, {'numeric'}, ...
                {'real', 'finite', 'scalar'}, mfilename, 't');
            obj.validateVector6(eta, 'eta');
            obj.validateVector6(nu, 'nu');
            obj.validateVector3(velocityCommandNED, 'velocityCommandNED');
        end

        function n = getNumInputsImpl(~)
            n = 4;
        end

        function n = getNumOutputsImpl(~)
            n = 2;
        end

        function [s1, s2] = getOutputSizeImpl(~)
            s1 = [6, 1];
            s2 = [6, 1];
        end

        function [d1, d2] = getOutputDataTypeImpl(~)
            d1 = 'double';
            d2 = 'double';
        end

        function [c1, c2] = isOutputComplexImpl(~)
            c1 = false;
            c2 = false;
        end

        function [f1, f2] = isOutputFixedSizeImpl(~)
            f1 = true;
            f2 = true;
        end

        function [n1, n2, n3, n4] = getInputNamesImpl(~)
            n1 = 't';
            n2 = 'eta';
            n3 = 'nu';
            n4 = 'velocity_command_NED';
        end

        function [n1, n2] = getOutputNamesImpl(~)
            n1 = 'tau';
            n2 = 'F_DI';
        end

        function icon = getIconImpl(~)
            icon = sprintf('Control + Wave Loads\nNED velocity command');
        end

        function flag = supportsMultipleInstanceImpl(~)
            flag = true;
        end
    end

    methods (Access = private)
        function tau = controlLoad(obj, eta, nu, F_DI, velocityCommandNED)
            [totalMass, C, damping, restoring] = obj.nominalDynamics(nu);

            rotationBodyToNED = obj.bodyToNEDRotation(eta);
            desiredBodyTranslation = rotationBodyToNED.' * ...
                velocityCommandNED;
            horizontalCommandSpeed = norm(velocityCommandNED(1:2));
            desiredHeading = obj.DesiredHeading;
            if horizontalCommandSpeed >= obj.HeadingCommandSpeedThreshold
                desiredHeading = atan2(velocityCommandNED(2), ...
                    velocityCommandNED(1));
            end

            desiredAcceleration = zeros(6, 1);
            translationTimeConstant = double(obj.TranslationTimeConstant(:));
            desiredAcceleration(1:3) = ...
                (desiredBodyTranslation - nu(1:3)) ./ translationTimeConstant;

            positionError = [0; ...
                0; ...
                0; ...
                eta(4) - obj.DesiredRollAngle; ...
                eta(5) - obj.DesiredPitchAngle; ...
                obj.wrapToPi(eta(6) - desiredHeading)];
            omega = double(obj.ControlNaturalFrequency(:));
            zeta = double(obj.ControlDampingRatio(:));
            desiredAcceleration(4:6) = ...
                -2 .* zeta(4:6) .* omega(4:6) .* nu(4:6) - ...
                omega(4:6).^2 .* positionError(4:6);

            % Dynamic inversion of the same nominal matrices used by
            % ship_motion_model_2.nominal70mParameters(). At the target
            % state this automatically produces D(1,1)*U steady thrust.
            restoringError = eta - ...
                double(obj.NominalRestoringEquilibrium(:));
            tauUnsaturated = totalMass * desiredAcceleration + C * nu + ...
                damping * nu + restoring * restoringError - ...
                double(obj.WaveFeedforwardGain(:)) .* F_DI;

            lowerLimit = double(obj.MinimumControlLoad(:));
            upperLimit = double(obj.MaximumControlLoad(:));
            tau = min(max(tauUnsaturated, lowerLimit), upperLimit);
        end

        function F_DI = waveLoad(obj, t, component, velocityCommandNED)
            horizontalCommandSpeed = norm(velocityCommandNED(1:2));
            omegaEncounter = component.omega - component.waveNumber * ...
                horizontalCommandSpeed .* cos(component.beta);
            argument = omegaEncounter * t + component.phase;
            cosine = cos(argument);
            sine = sin(argument);

            commonHorizontal = obj.DiffractionScale * obj.DisplacementMass .* ...
                component.omega.^2 .* component.amplitude .* ...
                component.spatialFilter;
            surgeAmplitude = obj.SurgeExcitationCoefficient .* ...
                commonHorizontal .* cos(component.beta);
            swayAmplitude = obj.SwayExcitationCoefficient .* ...
                commonHorizontal .* sin(component.beta);

            waterplaneArea = obj.WaterplaneCoefficient * ...
                obj.ShipLength * obj.ShipBeam;
            heaveAmplitude = obj.DiffractionScale * ...
                obj.HeaveExcitationCoefficient * obj.WaterDensity * ...
                obj.Gravity * waterplaneArea .* component.amplitude .* ...
                component.spatialFilter;

            rollStiffness = obj.DisplacementMass * obj.Gravity * ...
                obj.TransverseGM;
            pitchStiffness = obj.DisplacementMass * obj.Gravity * ...
                obj.LongitudinalGM;
            transverseSlope = component.waveNumber .* component.amplitude .* ...
                sin(component.beta) .* component.spatialFilter;
            longitudinalSlope = component.waveNumber .* component.amplitude .* ...
                cos(component.beta) .* component.spatialFilter;
            rollAmplitude = obj.DiffractionScale * ...
                obj.RollExcitationCoefficient * rollStiffness .* transverseSlope;
            pitchAmplitude = obj.DiffractionScale * ...
                obj.PitchExcitationCoefficient * pitchStiffness .* longitudinalSlope;

            longitudinalPhase = sin(0.5 * component.waveNumber * ...
                obj.ShipLength .* cos(component.beta));
            yawAmplitude = obj.YawExcitationCoefficient * swayAmplitude * ...
                (0.25 * obj.ShipLength) .* longitudinalPhase;

            F_DI = [sum(surgeAmplitude .* cosine); ...
                    sum(swayAmplitude .* cosine); ...
                    sum(heaveAmplitude .* cosine); ...
                   -sum(rollAmplitude .* sine); ...
                   -sum(pitchAmplitude .* sine); ...
                    sum(yawAmplitude .* sine)];
        end

        function [totalMass, C, damping, restoring] = nominalDynamics(obj, nu)
            % Matrices must match ship_motion_model_2.nominal70mParameters.
            mass = double(obj.DisplacementMass);
            kRoll = 0.37 * double(obj.ShipBeam);
            kPitch = 0.25 * double(obj.ShipLength);
            kYaw = 0.25 * double(obj.ShipLength);
            rigidInertia = mass * [kRoll^2, kPitch^2, kYaw^2];
            rigidMass = diag([mass, mass, mass, rigidInertia]);

            addedRatio = [0.05, 1.00, 3.40, 0.45, 3.00, 0.35];
            addedMass = diag(addedRatio .* ...
                [mass, mass, mass, rigidInertia]);
            totalMass = rigidMass + addedMass;

            linearMomentum = totalMass(1:3, 1:3) * nu(1:3) + ...
                totalMass(1:3, 4:6) * nu(4:6);
            angularMomentum = totalMass(4:6, 1:3) * nu(1:3) + ...
                totalMass(4:6, 4:6) * nu(4:6);
            zero3 = zeros(3);
            momentumSkew = obj.skew(linearMomentum);
            C = [zero3,             -momentumSkew; ...
                 -momentumSkew, -obj.skew(angularMomentum)];

            waterplaneArea = obj.WaterplaneCoefficient * ...
                obj.ShipLength * obj.ShipBeam;
            heaveStiffness = obj.WaterDensity * obj.Gravity * waterplaneArea;
            rollStiffness = mass * obj.Gravity * obj.TransverseGM;
            pitchStiffness = mass * obj.Gravity * obj.LongitudinalGM;
            restoring = diag( ...
                [0, 0, heaveStiffness, rollStiffness, pitchStiffness, 0]);

            effectiveInertia = diag(totalMass);
            dampingDiagonal = [ ...
                effectiveInertia(1) / 60; ...
                effectiveInertia(2) / 25; ...
                2*0.25*sqrt(effectiveInertia(3)*heaveStiffness); ...
                2*0.08*sqrt(effectiveInertia(4)*rollStiffness); ...
                2*0.15*sqrt(effectiveInertia(5)*pitchStiffness); ...
                effectiveInertia(6) / 30];
            damping = diag(dampingDiagonal);
        end

        function component = buildWaveComponents(obj)
            edge = linspace(obj.MinimumWaveFrequency, ...
                obj.MaximumWaveFrequency, obj.NumberOfWaveComponents + 1).';
            omega = 0.5 * (edge(1:end-1) + edge(2:end));
            deltaOmega = diff(edge);
            spectrum = obj.piersonMoskowitzSpectrum(omega);
            % Preserve the requested Hs despite truncating the frequency
            % range: m0 = integral(S d omega) = Hs^2/16.
            targetVariance = obj.SignificantWaveHeight^2 / 16;
            discreteVariance = sum(spectrum .* deltaOmega);
            spectrum = spectrum * (targetVariance / discreteVariance);
            amplitude = sqrt(2 * spectrum .* deltaOmega);

            stream = RandStream('mt19937ar', 'Seed', double(obj.RandomSeed));
            phase = 2 * pi * rand(stream, obj.NumberOfWaveComponents, 1);

            beta = deg2rad(obj.RelativeWaveDirectionDeg) * ...
                ones(obj.NumberOfWaveComponents, 1);
            waveNumber = omega.^2 / obj.Gravity;

            xLongitudinal = 0.5 * waveNumber * obj.ShipLength .* cos(beta);
            xTransverse = 0.5 * waveNumber * obj.ShipBeam .* sin(beta);
            spatialFilter = obj.sincUnnormalized(xLongitudinal) .* ...
                obj.sincUnnormalized(xTransverse);

            component = struct( ...
                'omega', omega, ...
                'deltaOmega', deltaOmega, ...
                'spectrum', spectrum, ...
                'amplitude', amplitude, ...
                'phase', phase, ...
                'beta', beta, ...
                'waveNumber', waveNumber, ...
                'spatialFilter', spatialFilter);
        end

        function spectrum = piersonMoskowitzSpectrum(obj, omega)
            % Normalized PM spectrum: integral S(omega)domega = Hs^2/16.
            omegaPeak = 2 * pi / obj.PeakPeriod;
            spectrum = (5/16) * obj.SignificantWaveHeight^2 * ...
                omegaPeak^4 ./ omega.^5 .* ...
                exp(-(5/4) * (omegaPeak ./ omega).^4);
            spectrum(~isfinite(spectrum)) = 0;
        end

        function validateParameters(obj)
            positiveScalar = {'real', 'finite', 'scalar', 'positive'};
            validateattributes(obj.ShipLength, {'numeric'}, positiveScalar, ...
                mfilename, 'ShipLength');
            validateattributes(obj.ShipBeam, {'numeric'}, positiveScalar, ...
                mfilename, 'ShipBeam');
            validateattributes(obj.ShipDraught, {'numeric'}, positiveScalar, ...
                mfilename, 'ShipDraught');
            validateattributes(obj.DisplacementMass, {'numeric'}, positiveScalar, ...
                mfilename, 'DisplacementMass');
            validateattributes(obj.WaterDensity, {'numeric'}, positiveScalar, ...
                mfilename, 'WaterDensity');
            validateattributes(obj.Gravity, {'numeric'}, positiveScalar, ...
                mfilename, 'Gravity');
            validateattributes(obj.SignificantWaveHeight, {'numeric'}, positiveScalar, ...
                mfilename, 'SignificantWaveHeight');
            validateattributes(obj.PeakPeriod, {'numeric'}, positiveScalar, ...
                mfilename, 'PeakPeriod');
            validateattributes(obj.DesiredHeading, {'numeric'}, ...
                {'real', 'finite', 'scalar'}, mfilename, 'DesiredHeading');
            validateattributes(obj.HeadingCommandSpeedThreshold, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'nonnegative'}, ...
                mfilename, 'HeadingCommandSpeedThreshold');
            referenceNames = {'DesiredRollAngle', 'DesiredPitchAngle'};
            for k = 1:numel(referenceNames)
                validateattributes(obj.(referenceNames{k}), {'numeric'}, ...
                    {'real', 'finite', 'scalar'}, mfilename, referenceNames{k});
            end
            obj.validateVector3(obj.TranslationTimeConstant, ...
                'TranslationTimeConstant');
            if any(obj.TranslationTimeConstant(:) <= 0)
                error('tauAndFDI:InvalidTranslationTimeConstant', ...
                    'All translation time constants must be positive.');
            end
            obj.validateVector6(obj.NominalRestoringEquilibrium, ...
                'NominalRestoringEquilibrium');
            obj.validateVector6(obj.ControlNaturalFrequency, ...
                'ControlNaturalFrequency');
            obj.validateVector6(obj.ControlDampingRatio, ...
                'ControlDampingRatio');
            obj.validateVector6(obj.WaveFeedforwardGain, ...
                'WaveFeedforwardGain');
            obj.validateVector6(obj.MinimumControlLoad, 'MinimumControlLoad');
            obj.validateVector6(obj.MaximumControlLoad, 'MaximumControlLoad');
            if any(obj.ControlNaturalFrequency(:) < 0) || ...
                    any(obj.ControlDampingRatio(:) < 0) || ...
                    any(obj.WaveFeedforwardGain(:) < 0) || ...
                    any(obj.WaveFeedforwardGain(:) > 1)
                error('tauAndFDI:InvalidControlTuning', ...
                    ['Control frequencies/damping must be nonnegative, ', ...
                     'and WaveFeedforwardGain must lie in [0,1].']);
            end
            if any(obj.MinimumControlLoad(:) >= obj.MaximumControlLoad(:))
                error('tauAndFDI:InvalidControlLimits', ...
                    'Each minimum control load must be below its maximum.');
            end
            validateattributes(obj.RelativeWaveDirectionDeg, {'numeric'}, ...
                {'real', 'finite', 'scalar'}, ...
                mfilename, 'RelativeWaveDirectionDeg');
            validateattributes(obj.MinimumWaveFrequency, {'numeric'}, ...
                positiveScalar, mfilename, 'MinimumWaveFrequency');
            validateattributes(obj.MaximumWaveFrequency, {'numeric'}, ...
                {'real', 'finite', 'scalar', '>', obj.MinimumWaveFrequency}, ...
                mfilename, 'MaximumWaveFrequency');
            validateattributes(obj.NumberOfWaveComponents, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
                mfilename, 'NumberOfWaveComponents');
            validateattributes(obj.RandomSeed, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'integer', 'nonnegative'}, ...
                mfilename, 'RandomSeed');

            fractionNames = {'WaterplaneCoefficient', 'DiffractionScale', ...
                'SurgeExcitationCoefficient', 'SwayExcitationCoefficient', ...
                'HeaveExcitationCoefficient', 'RollExcitationCoefficient', ...
                'PitchExcitationCoefficient', 'YawExcitationCoefficient'};
            for k = 1:numel(fractionNames)
                validateattributes(obj.(fractionNames{k}), {'numeric'}, ...
                    {'real', 'finite', 'scalar', 'nonnegative'}, ...
                    mfilename, fractionNames{k});
            end
        end

        function angle = wrapToPi(~, angle)
            angle = mod(angle + pi, 2*pi) - pi;
        end

        function R = bodyToNEDRotation(~, eta)
            phi = eta(4);
            theta = eta(5);
            psi = eta(6);
            cPhi = cos(phi);   sPhi = sin(phi);
            cTheta = cos(theta); sTheta = sin(theta);
            cPsi = cos(psi);   sPsi = sin(psi);
            R = [cPsi*cTheta, ...
                 -sPsi*cPhi + cPsi*sTheta*sPhi, ...
                  sPsi*sPhi + cPsi*cPhi*sTheta; ...
                 sPsi*cTheta, ...
                  cPsi*cPhi + sPhi*sTheta*sPsi, ...
                 -cPsi*sPhi + sTheta*sPsi*cPhi; ...
                 -sTheta, cTheta*sPhi, cTheta*cPhi];
        end

        function value = sincUnnormalized(~, x)
            value = ones(size(x));
            index = abs(x) > 1.0e-10;
            value(index) = sin(x(index)) ./ x(index);
        end

        function S = skew(~, vector)
            S = [0,         -vector(3),  vector(2); ...
                 vector(3),  0,         -vector(1); ...
                -vector(2),  vector(1),  0];
        end

        function validateVector6(~, value, name)
            validateattributes(value, {'numeric'}, ...
                {'real', 'finite', 'vector', 'numel', 6}, mfilename, name);
        end

        function validateVector3(~, value, name)
            validateattributes(value, {'numeric'}, ...
                {'real', 'finite', 'vector', 'numel', 3}, mfilename, name);
        end
    end
end
